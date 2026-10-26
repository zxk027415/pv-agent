"""Explicit API table initialization and authenticated loopback server startup."""

import argparse
import asyncio
import json
from pathlib import Path

from pvdiag.agent.persistence import PostgresStore
from pvdiag.agent.sessions import connection_setting
from pvdiag.api.app import create_app
from pvdiag.api.config import Authentication, load_profiles
from pvdiag.api.store import ApiStore
from pvdiag.observability.store import TraceStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "serve"])
    parser.add_argument("--dsn-env", default="PVDIAG_POSTGRES_DSN")
    parser.add_argument("--schema", default="pvdiag")
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--identities-env", default="PVDIAG_API_TOKENS")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-active", type=int, default=2)
    parser.add_argument("--trace-db", type=Path)
    args = parser.parse_args(argv)
    try:
        dsn = connection_setting(args.dsn_env)
        if args.command == "init":

            async def initialize():
                async with PostgresStore.open(
                    dsn, schema=args.schema, initialize=True
                ) as persistence:
                    await ApiStore(persistence).initialize()

            asyncio.run(initialize())
            print("API persistence initialized")
            return 0
        if args.profiles is None:
            parser.error("serve requires --profiles")
        authentication = Authentication(json.loads(connection_setting(args.identities_env)))
        profiles = load_profiles(args.profiles, allow_model_calls=args.allow_model_calls)
        app = create_app(
            profiles,
            authentication,
            dsn,
            schema=args.schema,
            max_active=args.max_active,
            trace_store=TraceStore(args.trace_db) if args.trace_db else None,
        )
    except (ValueError, RuntimeError, OSError):
        parser.exit(
            2,
            "invalid API configuration or unavailable storage; check the configured environment fields and profiles\n",
        )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, access_log=False, timeout_graceful_shutdown=10)
    return 0


if __name__ == "__main__":
    main()
