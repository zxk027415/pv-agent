"""Inspect/prepare a local trace outbox, or explicitly upload it to Langfuse."""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import dotenv_values

from pvdiag.observability.langfuse import LangfuseExporter, prepare
from pvdiag.observability.store import TraceStore, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "prepare", "upload"])
    parser.add_argument("--trace-db", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prepared", type=Path, help="the reviewed prepare artifact; required for upload"
    )
    parser.add_argument(
        "--base-url", help="explicit Langfuse Cloud HTTPS origin, or LANGFUSE_BASE_URL"
    )
    parser.add_argument(
        "--allow-upload",
        action="store_true",
        help="explicitly permit remote dataset, trace and score writes",
    )
    args = parser.parse_args(argv)
    if not args.trace_db.is_file():
        parser.error("the configured local trace database does not exist")
    if args.command == "upload" and not args.allow_upload:
        parser.error("upload requires --allow-upload; no network request was sent")
    if args.command == "upload" and args.prepared is None:
        parser.error("upload requires --prepared to bind the reviewed local snapshot")
    if args.command == "prepare" and args.output is None:
        parser.error("prepare requires --output for the reviewable export artifact")
    try:
        store = TraceStore(args.trace_db)
        snapshot = store.snapshot()
        if args.command == "prepare":
            result = prepare(snapshot)
        elif args.command == "inspect":
            result = {
                "version": "local-otlp-v1",
                "objects": {k: len(v) for k, v in snapshot.items()},
                "incomplete_observations": sum(
                    s["end_ns"] is None for s in snapshot.get("span", {}).values()
                ),
                "snapshot_sha256": digest(snapshot),
                "deliveries": store.delivery_report(),
            }
        else:
            reviewed = json.loads(args.prepared.read_text())
            if reviewed != prepare(snapshot):
                raise ValueError("langfuse_reviewed_snapshot_changed")
            settings = dotenv_values(".env", interpolate=False)

            def setting(name):
                return os.getenv(name) or settings.get(name)

            async def upload():
                exporter = LangfuseExporter(
                    store,
                    base_url=args.base_url or setting("LANGFUSE_BASE_URL") or "",
                    public_key=setting("LANGFUSE_PUBLIC_KEY"),
                    secret_key=setting("LANGFUSE_SECRET_KEY"),
                )
                try:
                    return await exporter.upload(
                        allow_upload=True,
                        expected_snapshot_sha256=reviewed["source_snapshot_sha256"],
                    )
                finally:
                    await exporter.close()

            result = asyncio.run(upload())
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )
            print(f"{args.output}: {args.command}, sha256={digest(result)}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError) as error:
        code = (
            str(error)
            if re.fullmatch(r"(?:langfuse|trace)_[a-z0-9_]{1,100}", str(error))
            else type(error).__name__
        )
        parser.exit(
            2,
            f"observability operation failed: {code}; inspect local configuration and delivery status\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
