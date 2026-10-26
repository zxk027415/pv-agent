"""Run one diagnosis; model calls and human-approved local orders are opt-in."""

import argparse
import asyncio
import getpass
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values
from langgraph.types import Command

from pvdiag.agent.ablation import LEVELS, resolve_ablation
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.runtime import open_runtime
from pvdiag.agent.state import ApprovalDecision, DiagRequest
from pvdiag.mcp_servers.pv_knowledge import (
    add_knowledge_arguments,
    knowledge_from_args,
)
from pvdiag.mcp_servers.store import ObservationStore
from pvdiag.observability.store import TraceStore
from pvdiag.observability.tracing import trace_session


def reasoner_from_args(args):
    policy, _ = resolve_ablation(
        getattr(args, "ablation", None), getattr(args, "verification", None)
    )
    if args.mode == "b0":
        if policy:
            raise ValueError("A0–A3 require --mode llm")
        return RuleReasoner()
    if not args.allow_model_calls:
        raise ValueError("LLM mode requires --allow-model-calls; no model request was sent")
    from pvdiag.agent.llm import LLMReasoner

    settings = dotenv_values(Path.cwd() / ".env", interpolate=False)
    field = getattr(args, "credential_env", None)
    if field is None:
        field = {"api.deepseek.com": "DEEPSEEK_API_KEY", "api.openai.com": "OPENAI_API_KEY"}.get(
            urlsplit(args.base_url).hostname, "PVDIAG_LLM_API_KEY"
        )
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", field):
        raise ValueError("invalid credential environment field name")
    key = os.environ.get(field) or settings.get(field)
    if not key:
        raise ValueError(f"set {field} in the environment or local .env for this endpoint")
    return LLMReasoner(
        args.model,
        provider="openai",
        base_url=args.base_url,
        api_key=key,
        max_model_calls=getattr(args, "max_model_calls", 6),
    )


def add_model_arguments(parser):
    parser.add_argument("--mode", choices=["b0", "llm"], default="b0")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--max-model-calls", type=int, default=6)
    parser.add_argument(
        "--credential-env", help="explicit credential field for the selected endpoint"
    )
    parser.add_argument(
        "--verification",
        choices=["structural", "domain", "domain_llm"],
        help="default: structural; A3 fixes this to domain_llm",
    )
    parser.add_argument("--ablation", choices=LEVELS, help="enforce an A0–A3 experiment policy")
    parser.add_argument(
        "--trace-db", type=Path, help="explicitly record a local Langfuse trace outbox"
    )
    parser.add_argument(
        "--allow-model-calls",
        action="store_true",
        help="explicitly permit potentially billable model requests",
    )


async def run_diagnosis(
    store,
    request,
    reasoner,
    *,
    max_calls=30,
    max_loops=3,
    verification_mode=None,
    knowledge_store=None,
    ops_store=None,
    approval_handler=None,
    ablation=None,
    trace_store=None,
):
    if (ops_store is None) != (approval_handler is None):
        raise ValueError("operations require an explicit trusted human approval handler")
    async with open_runtime(
        store,
        reasoner,
        max_calls=max_calls,
        max_loops=max_loops,
        verification_mode=verification_mode,
        knowledge_store=knowledge_store,
        ops_store=ops_store,
        ablation=ablation,
        trace_store=trace_store,
    ) as runtime:
        graph = runtime.graph
        config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 40}
        with trace_session(
            trace_store, config["configurable"]["thread_id"], request, runtime.contract
        ):
            state = await graph.ainvoke({"request": request}, config=config)
            if state.get("__interrupt__"):
                decision = await approval_handler(state["__interrupt__"][0].value)
                if isinstance(decision, ApprovalDecision):
                    decision = decision.model_dump(mode="json")
                state = await graph.ainvoke(Command(resume=decision), config=config)
        return state["result"]


async def console_approval(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        answer = await asyncio.to_thread(input, "输入 approve 批准此本地核查工单；其他输入拒绝：")
    except EOFError:
        answer = "reject"
    proposal = payload["proposal"]
    return ApprovalDecision(
        action_id=proposal["action_id"],
        payload_sha256=proposal["payload_sha256"],
        approved=answer.strip() == "approve",
        reviewer=getpass.getuser(),
        reason="本地 CLI 人工确认" if answer.strip() == "approve" else "本地 CLI 未批准",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_id", type=int)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/runs/diagnosis.json"))
    add_model_arguments(parser)
    add_knowledge_arguments(parser)
    parser.add_argument(
        "--enable-ops",
        action="store_true",
        help="explicitly enable simulated local work orders with interactive human approval",
    )
    parser.add_argument(
        "--ops-db", type=Path, help="local work-order ledger; requires --enable-ops"
    )
    args = parser.parse_args(argv)
    if args.ops_db and not args.enable_ops:
        parser.error("--ops-db requires --enable-ops")
    request = DiagRequest(
        system_id=args.system_id, device_ids=args.devices, start=args.start, end=args.end
    )
    reasoner = reasoner_from_args(args)
    ops_store = None
    if args.enable_ops:
        from pvdiag.ops.store import WorkOrderStore

        ops_store = WorkOrderStore(args.ops_db or Path("data/ops/work_orders.sqlite3"))
    result = asyncio.run(
        run_diagnosis(
            ObservationStore(args.observations, args.metadata),
            request,
            reasoner,
            verification_mode=args.verification,
            knowledge_store=knowledge_from_args(args),
            ops_store=ops_store,
            approval_handler=console_approval if args.enable_ops else None,
            ablation=args.ablation,
            trace_store=TraceStore(args.trace_db) if args.trace_db else None,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n")
    print(
        f"{args.output}: mode={result.model['mode']}, status={result.report.status}, tools={len(result.trace)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
