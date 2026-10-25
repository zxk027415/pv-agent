"""Run a single read-only diagnosis; network model calls require an explicit flag."""

import argparse
import asyncio
import os
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

from pvdiag.agent.gateway import MCPGateway
from pvdiag.agent.graph import build_graph
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.state import DiagRequest
from pvdiag.mcp_servers.pv_analytics import create_server as analytics_server
from pvdiag.mcp_servers.pv_data import create_server as data_server
from pvdiag.mcp_servers.store import ObservationStore


def reasoner_from_args(args):
    if args.mode == "b0":
        return RuleReasoner()
    if not args.allow_model_calls:
        raise ValueError("LLM mode requires --allow-model-calls; no model request was sent")
    from pvdiag.agent.llm import LLMReasoner

    settings = dotenv_values(Path.cwd() / ".env", interpolate=False)
    key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or settings.get("DEEPSEEK_API_KEY")
        or settings.get("OPENAI_API_KEY")
    )
    if not key:
        raise ValueError("set DEEPSEEK_API_KEY or OPENAI_API_KEY in the environment or local .env")
    return LLMReasoner(args.model, provider="openai", base_url=args.base_url, api_key=key)


def add_model_arguments(parser):
    parser.add_argument("--mode", choices=["b0", "llm"], default="b0")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument(
        "--allow-model-calls",
        action="store_true",
        help="explicitly permit potentially billable model requests",
    )


async def run_diagnosis(store, request, reasoner, *, max_calls=30, max_loops=3):
    async with MCPGateway.connect(data_server(store), analytics_server(store)) as gateway:
        graph = build_graph(gateway, reasoner, max_calls=max_calls, max_loops=max_loops)
        state = await graph.ainvoke(
            {"request": request},
            config={"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 40},
        )
        return state["result"]


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
    args = parser.parse_args(argv)
    request = DiagRequest(
        system_id=args.system_id, device_ids=args.devices, start=args.start, end=args.end
    )
    reasoner = reasoner_from_args(args)
    result = asyncio.run(
        run_diagnosis(ObservationStore(args.observations, args.metadata), request, reasoner)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n")
    print(
        f"{args.output}: mode={result.model['mode']}, status={result.report.status}, tools={len(result.trace)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
