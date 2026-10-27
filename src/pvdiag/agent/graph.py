"""Bounded diagnosis with optional human-approved local inspection orders."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from time import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from pvdiag.agent.ablation import resolve_ablation, summary_calls, validate_ablation
from pvdiag.agent.guards import check_report, degrade
from pvdiag.agent.state import (
    ActionOutcome,
    ApprovalDecision,
    DiagnosisReport,
    DiagRequest,
    DiagState,
    ModelReview,
    PendingAction,
    Plan,
    RunResult,
    Scope,
    ToolCall,
    VerificationIssue,
    VerificationRecord,
    WorkOrderDraft,
)
from pvdiag.agent.verification import check_domain
from pvdiag.observability.tracing import current_session

DEFAULT_PROMPT = Path(__file__).resolve().parents[3] / "harness" / "system_prompt.md"
DEFAULT_REVIEW_PROMPT = DEFAULT_PROMPT.with_name("verification_prompt.md")


def build_graph(
    gateway,
    reasoner,
    *,
    max_calls=30,
    max_loops=3,
    checkpointer=None,
    prompt_path=DEFAULT_PROMPT,
    verification_mode=None,
    review_prompt_path=DEFAULT_REVIEW_PROMPT,
    ops_gateway=None,
    ablation=None,
    memory_view=None,
):
    policy, verification_mode = resolve_ablation(ablation, verification_mode)
    validate_ablation(policy, reasoner, gateway.knowledge_source_sha256, ops_gateway, memory_view)
    if policy and gateway.allowed_tools != policy.model_tools:
        raise ValueError("gateway permissions differ from the ablation policy")
    if not 2 <= max_calls <= 100 or not 1 <= max_loops <= 3:
        raise ValueError("invalid execution budget")
    prompt = prompt_path.read_text()
    if policy:
        prompt += policy.prompt()
    if verification_mode not in {"structural", "domain", "domain_llm"}:
        raise ValueError("invalid verification mode")
    if verification_mode == "domain_llm" and not callable(getattr(reasoner, "review", None)):
        raise ValueError("domain_llm requires an explicit model reviewer")
    review_prompt = review_prompt_path.read_text() if verification_mode == "domain_llm" else ""
    harness_sha = hashlib.sha256((prompt + review_prompt).encode()).hexdigest()
    policy_sha = hashlib.sha256(
        b"\0".join(
            Path(__file__).with_name(name).read_bytes() for name in ("guards.py", "verification.py")
        )
    ).hexdigest()
    ops_policy_sha = None
    if ops_gateway is not None:
        root = Path(__file__).resolve().parents[1]
        ops_policy_sha = hashlib.sha256(
            b"\0".join(
                (root / name).read_bytes()
                for name in (
                    "agent/graph.py",
                    "agent/__main__.py",
                    "agent/state.py",
                    "agent/runtime.py",
                    "agent/persistence.py",
                    "agent/sessions.py",
                    "ops/store.py",
                    "ops/gateway.py",
                    "mcp_servers/pv_ops.py",
                )
            )
        ).hexdigest()

    async def guard_in(state: DiagState):
        started = time()
        request = DiagRequest.model_validate(state["request"])
        if not any(
            word in request.question.lower()
            for word in (
                "光伏",
                "逆变器",
                "辐照",
                "发电",
                "故障",
                "诊断",
                "photovoltaic",
                "inverter",
                "solar",
                "pv",
            )
        ):
            raise ValueError("question is outside PV diagnosis scope")
        responses = await asyncio.gather(
            *(
                gateway.call(
                    ToolCall(name=name, arguments={"system_id": request.system_id}),
                    request,
                    None,
                    "guard_in",
                )
                for name in ("list_devices", "get_site_info")
            )
        )
        if any(item[0] is None for item in responses):
            raise ValueError("cannot establish observation scope")
        evidence = {ev["evidence_id"]: ev for ev, _ in responses}
        if len({ev["source_sha256"] for ev in evidence.values()}) != 1:
            raise ValueError("bootstrap observation snapshots differ")
        catalog = responses[0][0]["result"]["devices"]
        inverters = {d["device_id"] for d in catalog if d["kind"] == "inverter"}
        if not set(request.device_ids) <= inverters:
            raise ValueError("requested inverter outside observation scope")
        poa = next((d["device_id"] for d in catalog if "poa_wm2" in d["metrics"]), None)
        ambient = next((d["device_id"] for d in catalog if "ambient_c" in d["metrics"]), None)
        scope = Scope(
            source_sha256=responses[0][0]["source_sha256"],
            device_ids=request.device_ids,
            poa_device_id=poa,
            ambient_device_id=ambient,
        )
        return {
            "request": request,
            "scope": scope,
            "evidence": evidence,
            "trace": [trace for _, trace in responses],
            "loops": 0,
            "critique": [],
            "started_at": started,
            "verification": [],
            "pending_action": None,
            "approval": None,
            "action_outcome": None,
            "memory": None,
        }

    async def fixed_summary(state):
        calls = summary_calls(state["request"], state["scope"])
        if len(calls) + len(state["trace"]) > max_calls:
            raise ValueError("tool budget cannot cover the complete fixed summary")
        responses = await asyncio.gather(
            *(
                gateway.call(call, state["request"], state["scope"], "fixed_summary")
                for call in calls
            )
        )
        if any(item is None for item, _ in responses):
            raise ValueError("cannot establish the complete fixed summary")
        return {
            "evidence": state["evidence"] | {ev["evidence_id"]: ev for ev, _ in responses},
            "trace": [*state["trace"], *(trace for _, trace in responses)],
            "plan": Plan(calls=[]),
            "loops": 1 if policy.level == "A0" else 0,
        }

    def model_state(state):
        if policy and not policy.feedback:
            # No output-check findings, prior verdicts or reviewer guidance can
            # become an implicit verification agent in A0–A2.
            return {
                key: state[key] for key in ("request", "scope", "evidence", "trace", "loops")
            } | {"critique": [], "verification": []}
        return state

    async def recall_memory(state):
        return {"memory": await memory_view.recall(state["request"], state["evidence"])}

    async def plan(state):
        try:
            candidate = await reasoner.plan(model_state(state), gateway.tools, prompt)
            candidate = Plan.model_validate(
                candidate.model_dump() if isinstance(candidate, Plan) else candidate
            )
            return {"plan": candidate, "critique": []}
        except (ValueError, TypeError, RuntimeError):
            return {"plan": Plan(calls=[]), "critique": ["plan_failed"]}

    async def investigate(state):
        calls = state["plan"].calls[: max(0, max_calls - len(state["trace"]))]
        replies = await asyncio.gather(
            *(gateway.call(call, state["request"], state["scope"], "investigate") for call in calls)
        )
        evidence = dict(state["evidence"])
        for item, _ in replies:
            if item is not None:
                evidence[item["evidence_id"]] = item
        critique = list(state.get("critique", []))
        if len(calls) < len(state["plan"].calls):
            critique.append("tool_budget_exhausted")
        return {
            "evidence": evidence,
            "trace": [*state["trace"], *(trace for _, trace in replies)],
            "loops": state["loops"] + 1,
            "critique": critique,
        }

    async def hypothesize(state):
        try:
            report = await reasoner.hypothesize(model_state(state), prompt)
            return {
                "verdict": DiagnosisReport.model_validate(
                    report.model_dump() if isinstance(report, DiagnosisReport) else report
                )
            }
        except (ValueError, TypeError, RuntimeError):
            return {"verdict": degrade(state["evidence"]), "critique": ["hypothesis_failed"]}

    async def verify(state):
        structural = check_report(
            state["verdict"], state["evidence"], state["request"], state["scope"]
        )
        issues = [
            VerificationIssue(code=code, message="结构化输出或引用校验失败。")
            for code in structural
        ]
        # Invalid citations must be repaired before domain logic dereferences them.
        if not structural and verification_mode != "structural":
            issues.extend(
                check_domain(state["verdict"], state["evidence"], state["request"], state["scope"])
            )
        reviewed = False
        ready = state["plan"].ready_for_verdict and state["verdict"].status in {"fault", "normal"}
        if verification_mode == "domain_llm" and not structural and ready:
            try:
                review = await reasoner.review(state, issues, review_prompt)
                review = ModelReview.model_validate(
                    review.model_dump() if isinstance(review, ModelReview) else review
                )
                reviewed = True
                for issue in review.issues:
                    if set(issue.evidence_ids) - state["evidence"].keys() or (
                        issue.finding_index is not None
                        and issue.finding_index >= len(state["verdict"].findings)
                    ):
                        raise ValueError("invalid reviewer citations")
                issues.extend(review.issues)
            except (ValueError, TypeError, RuntimeError):
                issues.append(
                    VerificationIssue(
                        code="model_review_failed", message="模型自评不可用，不能视为通过。"
                    )
                )
                reviewed = False
        critique = sorted(
            set(state.get("critique", []) + [f"{issue.code}: {issue.message}" for issue in issues])
        )
        record = VerificationRecord(
            loop=state["loops"],
            mode=verification_mode,
            passed=not critique,
            issues=issues,
            model_reviewed=reviewed,
            checked_evidence_ids=list(state["evidence"]),
            report_status=state["verdict"].status,
            ready_for_verdict=state["plan"].ready_for_verdict,
        )
        return {"critique": critique, "verification": [*state["verification"], record]}

    def route(state):
        if policy and policy.level == "A0":
            return "propose_action"
        repeat = state["critique"] or not state["plan"].ready_for_verdict
        if policy and not policy.feedback:
            repeat = not state["plan"].ready_for_verdict
        return (
            "plan"
            if repeat and state["loops"] < max_loops and len(state["trace"]) < max_calls
            else "propose_action"
        )

    def action_draft(state):
        if state["critique"] or not state["plan"].ready_for_verdict:
            raise ValueError("diagnosis not ready for an action")
        verdict = state["verdict"]
        if (
            verdict.status != "fault"
            or check_report(verdict, state["evidence"], state["request"], state["scope"])
            or check_domain(verdict, state["evidence"], state["request"], state["scope"])
        ):
            raise ValueError("inspection action requires structurally and domain-supported faults")
        return WorkOrderDraft(
            request=state["request"],
            report=verdict,
            observation_source_sha256=state["scope"].source_sha256,
            harness_sha256=harness_sha,
            ops_policy_sha256=ops_policy_sha,
            knowledge_source_sha256=gateway.knowledge_source_sha256,
            evidence={key: state["evidence"][key] for key in verdict.evidence_ids},
        )

    def propose_action(state, config):
        if ops_gateway is None:
            return {}
        if state["verdict"].status != "fault":
            return {"action_outcome": ActionOutcome(status="not_needed")}
        try:
            draft = action_draft(state)
            proposal = PendingAction.propose(config["configurable"]["thread_id"], draft)
            ops_gateway.store.register(proposal)
            return {"pending_action": proposal, "action_outcome": ActionOutcome(status="pending")}
        except (ValueError, KeyError, RuntimeError):
            return {"action_outcome": ActionOutcome(status="blocked", error="action_not_supported")}

    def approval(state):
        proposal = state["pending_action"]
        response = interrupt(
            {
                "kind": "work_order_approval",
                "message": "请核对具体报告与证据。批准仅创建本地人工核查工单，不控制设备。",
                "proposal": proposal.model_dump(mode="json"),
            }
        )
        try:
            decision = ApprovalDecision.model_validate(response)
            record = ops_gateway.store.decide(proposal, decision)
            return {
                "approval": record,
                "action_outcome": ActionOutcome(
                    status="pending" if record.approved else "rejected"
                ),
            }
        except (ValueError, TypeError, RuntimeError):
            return {
                "approval": None,
                "action_outcome": ActionOutcome(
                    status="invalid_approval", error="approval_invalid_or_mismatched"
                ),
            }

    async def act(state):
        try:
            proposal = PendingAction.model_validate(state["pending_action"].model_dump())
            current = PendingAction.propose(proposal.run_id, action_draft(state))
            if current != proposal:
                raise ValueError("approved action no longer matches the current diagnosis")
        except (ValueError, KeyError, TypeError):
            return {
                "action_outcome": ActionOutcome(status="blocked", error="approved_payload_changed")
            }
        receipt, trace = await ops_gateway.create(proposal, state["approval"])
        return {
            "action_outcome": ActionOutcome(status="created", receipt=receipt)
            if receipt
            else ActionOutcome(status="failed", error="work_order_call_failed"),
            "trace": [*state["trace"], trace],
        }

    def report(state):
        verdict = (
            degrade(state["evidence"])
            if state["critique"] or not state["plan"].ready_for_verdict
            else state["verdict"]
        )
        return {"verdict": verdict}

    async def guard_out(state):
        if callable(getattr(reasoner, "sync_accounting", None)):
            await reasoner.sync_accounting()
        verdict = state["verdict"]
        if check_report(verdict, state["evidence"], state["request"], state["scope"]):
            verdict = degrade(state["evidence"])
        if verification_mode != "structural" and check_domain(
            verdict, state["evidence"], state["request"], state["scope"]
        ):
            verdict = degrade(state["evidence"])
        ablation_record = None
        if policy:
            summary_ids = [
                trace.evidence_id
                for trace in state["trace"]
                if trace.node in {"guard_in", "fixed_summary"} and trace.success
            ]
            ablation_record = policy.metadata() | {
                "max_tool_calls": max_calls,
                "max_loops": 1 if policy.level == "A0" else max_loops,
                "summary_evidence_ids": summary_ids,
                "summary_sha256": hashlib.sha256(
                    json.dumps(
                        {key: state["evidence"][key] for key in summary_ids},
                        sort_keys=True,
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
            }
        result = RunResult(
            request=state["request"],
            report=verdict,
            evidence=state["evidence"],
            trace=state["trace"],
            loops=state["loops"],
            model=deepcopy(reasoner.metadata),
            model_usage=deepcopy(reasoner.usage),
            harness_sha256=harness_sha,
            duration_seconds=max(0.0, time() - state["started_at"]),
            verification_mode=verification_mode,
            verification=state["verification"],
            verification_policy_sha256=policy_sha,
            knowledge_source_sha256=gateway.knowledge_source_sha256,
            ops_enabled=ops_gateway is not None,
            pending_action=state.get("pending_action"),
            approval=state.get("approval"),
            action_outcome=state.get("action_outcome"),
            ops_policy_sha256=ops_policy_sha,
            ablation=ablation_record,
            memory=state.get("memory"),
        )
        if session := current_session():
            result.observability = session.receipt()
            session.root.update(
                output={"report": result.report, "model_usage": result.model_usage},
                metadata={
                    "model": result.model,
                    "ablation": result.ablation,
                    "harness_sha256": result.harness_sha256,
                    "knowledge_source_sha256": result.knowledge_source_sha256,
                    "phase": "completed",
                    "usd_cost": None,
                },
            )
        return {"verdict": verdict, "result": result}

    graph = StateGraph(DiagState)
    for name, node in (
        ("guard_in", guard_in),
        ("fixed_summary", fixed_summary),
        ("recall_memory", recall_memory),
        ("plan", plan),
        ("investigate", investigate),
        ("hypothesize", hypothesize),
        ("verify", verify),
        ("propose_action", propose_action),
        ("approval", approval),
        ("act", act),
        ("report", report),
        ("guard_out", guard_out),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "guard_in")
    graph.add_edge("guard_in", "fixed_summary" if policy else "plan")
    graph.add_edge("fixed_summary", "recall_memory" if memory_view else "hypothesize" if policy and policy.level == "A0" else "plan")
    graph.add_edge("recall_memory", "plan")
    graph.add_edge("plan", "investigate")
    graph.add_edge("investigate", "hypothesize")
    graph.add_edge("hypothesize", "verify")
    graph.add_conditional_edges(
        "verify", route, {"plan": "plan", "propose_action": "propose_action"}
    )
    graph.add_conditional_edges(
        "propose_action", lambda state: "approval" if state.get("pending_action") else "report"
    )
    graph.add_conditional_edges(
        "approval",
        lambda state: "act" if state.get("approval") and state["approval"].approved else "report",
    )
    graph.add_edge("act", "report")
    graph.add_edge("report", "guard_out")
    graph.add_edge("guard_out", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
