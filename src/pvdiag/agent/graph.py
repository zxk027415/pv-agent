"""M1 LangGraph: scope → plan → parallel evidence → hypothesis → checks → report."""

from __future__ import annotations

import asyncio
import hashlib
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from pvdiag.agent.guards import check_report, degrade
from pvdiag.agent.state import (
    DiagnosisReport,
    DiagRequest,
    DiagState,
    Plan,
    RunResult,
    Scope,
    ToolCall,
)

DEFAULT_PROMPT = Path(__file__).resolve().parents[3] / "harness" / "system_prompt.md"


def build_graph(
    gateway, reasoner, *, max_calls=30, max_loops=3, checkpointer=None, prompt_path=DEFAULT_PROMPT
):
    if not 2 <= max_calls <= 100 or not 1 <= max_loops <= 3:
        raise ValueError("invalid execution budget")
    prompt = prompt_path.read_text()
    harness_sha = hashlib.sha256(prompt.encode()).hexdigest()

    async def guard_in(state: DiagState):
        started = perf_counter()
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
        }

    async def plan(state):
        try:
            candidate = await reasoner.plan(state, gateway.tools, prompt)
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
            report = await reasoner.hypothesize(state, prompt)
            return {
                "verdict": DiagnosisReport.model_validate(
                    report.model_dump() if isinstance(report, DiagnosisReport) else report
                )
            }
        except (ValueError, TypeError, RuntimeError):
            return {"verdict": degrade(state["evidence"]), "critique": ["hypothesis_failed"]}

    def verify(state):
        issues = check_report(state["verdict"], state["evidence"], state["request"], state["scope"])
        return {"critique": sorted(set(state.get("critique", []) + issues))}

    def route(state):
        repeat = state["critique"] or not state["plan"].ready_for_verdict
        return (
            "plan"
            if repeat and state["loops"] < max_loops and len(state["trace"]) < max_calls
            else "report"
        )

    def report(state):
        verdict = (
            degrade(state["evidence"])
            if state["critique"] or not state["plan"].ready_for_verdict
            else state["verdict"]
        )
        return {"verdict": verdict}

    def guard_out(state):
        verdict = state["verdict"]
        if check_report(verdict, state["evidence"], state["request"], state["scope"]):
            verdict = degrade(state["evidence"])
        result = RunResult(
            request=state["request"],
            report=verdict,
            evidence=state["evidence"],
            trace=state["trace"],
            loops=state["loops"],
            model=deepcopy(reasoner.metadata),
            model_usage=deepcopy(reasoner.usage),
            harness_sha256=harness_sha,
            duration_seconds=perf_counter() - state["started_at"],
        )
        return {"verdict": verdict, "result": result}

    graph = StateGraph(DiagState)
    for name, node in (
        ("guard_in", guard_in),
        ("plan", plan),
        ("investigate", investigate),
        ("hypothesize", hypothesize),
        ("verify", verify),
        ("report", report),
        ("guard_out", guard_out),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "guard_in")
    graph.add_edge("guard_in", "plan")
    graph.add_edge("plan", "investigate")
    graph.add_edge("investigate", "hypothesize")
    graph.add_edge("hypothesize", "verify")
    graph.add_conditional_edges("verify", route, {"plan": "plan", "report": "report"})
    graph.add_edge("report", "guard_out")
    graph.add_edge("guard_out", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
