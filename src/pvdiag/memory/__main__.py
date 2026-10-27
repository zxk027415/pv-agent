"""Memory CLI. Initialization and writes are explicit; lessons need interactive human review."""

import argparse
import asyncio
import getpass
import json
from pathlib import Path

from pvdiag.agent.state import RunResult


def review_lesson(draft):
    print(json.dumps(draft, ensure_ascii=False, indent=2))
    try:
        answer = input("输入 accept 接受此经验，reject 记录为拒绝；其他输入放弃：").strip()
        if answer not in {"accept", "reject"}:
            return None
        reason = input("审核理由：").strip()
    except EOFError:
        return None
    return draft | {
        "outcome": "accepted" if answer == "accept" else "rejected",
        "reviewer": getpass.getuser(),
        "review_reason": reason,
        "review_method": "local_interactive",
    }


async def execute(args, review=review_lesson):
    from pvdiag.agent.sessions import connection_setting
    from pvdiag.memory.embedding import MemoryEmbedding
    from pvdiag.memory.store import PgVectorStore

    lesson = None
    if args.command == "add-lesson":
        draft = json.loads(args.file.read_text())
        if set(draft) != {"title", "text", "source_case_ids"}:
            raise ValueError("lesson drafts contain only title, text and source_case_ids")
        lesson = review(draft)
        if lesson is None:
            return {"status": "not_reviewed"}
    async with PgVectorStore.open(
        connection_setting(args.dsn_env),
        MemoryEmbedding(args.embedding_path),
        schema=args.schema,
        initialize=args.command == "init",
        read_only=args.command == "show",
    ) as store:
        if args.command == "add-case":
            run = RunResult.model_validate_json(args.run.read_text())
            return await store.add_case(
                json.loads(args.manifest.read_text()),
                args.scenario_id,
                run,
                review_bundle=args.review_bundle,
            )
        if args.command == "add-lesson":
            return await store.add_lesson(lesson)
        view = await store.view(revision=getattr(args, "revision", None))
        return view.snapshot.model_dump(mode="json") | {"sha256": view.snapshot.sha256}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "add-case", "add-lesson", "show"):
        command = commands.add_parser(name)
        command.add_argument("--dsn-env", default="PVDIAG_MEMORY_DSN")
        command.add_argument("--schema", default="pvdiag_memory")
        command.add_argument("--embedding-path", type=Path, required=True)
        if name == "add-case":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--scenario-id", required=True)
            command.add_argument("--run", type=Path, required=True, help="RunResult JSON")
            command.add_argument("--review-bundle", type=Path)
        elif name == "add-lesson":
            command.add_argument(
                "--file", type=Path, required=True, help="title, text and source_case_ids"
            )
        elif name == "show":
            command.add_argument("--revision", type=int)
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(execute(args))
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
