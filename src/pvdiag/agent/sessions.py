"""Persistent diagnosis CLI. Schema initialization and interactive approval are explicit."""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import dotenv_values

from pvdiag.agent.__main__ import add_model_arguments, console_approval, reasoner_from_args
from pvdiag.agent.persistence import PostgresStore
from pvdiag.agent.runtime import SessionReply, open_runtime
from pvdiag.agent.state import DiagRequest, RunResult
from pvdiag.mcp_servers.pv_knowledge import add_knowledge_arguments, knowledge_from_args
from pvdiag.mcp_servers.store import ObservationStore
from pvdiag.observability.store import TraceStore


def connection_setting(field):
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", field):
        raise ValueError("invalid connection environment field")
    value = os.getenv(field) or dotenv_values(".env", interpolate=False).get(field)
    if not value:
        raise ValueError(f"set {field} in the environment or ignored local .env")
    return value


async def inspect_run(store, run_id):
    await store.get_run(run_id)
    checkpoint = await store.saver.aget_tuple({"configurable": {"thread_id": run_id}})
    if checkpoint is None:
        return SessionReply(
            run_id=run_id, status="running" if await store.is_active(run_id) else "interrupted"
        )
    values = checkpoint.checkpoint["channel_values"]
    if values.get("result"):
        return SessionReply(
            run_id=run_id, status="completed", result=RunResult.model_validate(values["result"])
        )
    for _, channel, value in checkpoint.pending_writes or []:
        if channel == "__interrupt__":
            interruption = value[0] if isinstance(value, (list, tuple)) else value
            return SessionReply(
                run_id=run_id, status="awaiting_approval", approval_request=interruption.value
            )
    return SessionReply(
        run_id=run_id, status="running" if await store.is_active(run_id) else "interrupted"
    )


async def execute(args):
    async with PostgresStore.open(
        connection_setting(args.dsn_env), schema=args.schema, initialize=args.command == "init"
    ) as persistence:
        if args.command == "init":
            return {"status": "initialized", "schema": args.schema}
        if args.command == "show":
            return (await inspect_run(persistence, args.run_id)).model_dump(mode="json")
        ops = None
        if args.ops_db and not args.enable_ops:
            raise ValueError("--ops-db requires --enable-ops")
        if args.enable_ops:
            from pvdiag.ops.store import WorkOrderStore

            ops = WorkOrderStore(args.ops_db or Path("data/ops/work_orders.sqlite3"))
        from pvdiag.memory.cli import memory_from_args

        async with memory_from_args(args) as memory_view, open_runtime(
            ObservationStore(args.observations, args.metadata),
            reasoner_from_args(args),
            knowledge_store=knowledge_from_args(args),
            ops_store=ops,
            persistence=persistence,
            verification_mode=args.verification,
            ablation=args.ablation,
            trace_store=TraceStore(args.trace_db) if args.trace_db else None,
            memory_view=memory_view,
        ) as runtime:
            if args.command == "start":
                request = DiagRequest(
                    system_id=args.system_id,
                    device_ids=args.devices,
                    start=args.start,
                    end=args.end,
                )
                reply = await runtime.start(args.run_id, request)
            else:
                reply = await runtime.resume(args.run_id)
            if args.interactive and reply.status == "awaiting_approval":
                decision = await console_approval(reply.approval_request)
                reply = await runtime.resume(args.run_id, decision.model_dump(mode="json"))
            return reply.model_dump(mode="json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "start", "resume", "show"):
        command = commands.add_parser(name)
        command.add_argument("--dsn-env", default="PVDIAG_POSTGRES_DSN")
        command.add_argument("--schema", default="pvdiag")
        command.add_argument("--output", type=Path)
        if name == "init":
            continue
        command.add_argument("run_id")
        if name == "show":
            continue
        command.add_argument("--observations", required=True, type=Path)
        command.add_argument("--metadata", required=True, type=Path)
        add_model_arguments(command)
        add_knowledge_arguments(command)
        command.add_argument("--enable-ops", action="store_true")
        command.add_argument("--ops-db", type=Path)
        command.add_argument(
            "--interactive",
            action="store_true",
            help="review the concrete pending action in the terminal",
        )
        if name == "start":
            command.add_argument("system_id", type=int)
            command.add_argument("--devices", nargs="+", required=True)
            command.add_argument("--start", required=True)
            command.add_argument("--end", required=True)
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(execute(args))
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
        print(f"{args.output}: {result['status']}")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
