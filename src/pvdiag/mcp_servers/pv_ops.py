"""Local simulated work-order server; no MCP endpoint can grant an approval."""

import argparse
from pathlib import Path

from fastmcp import FastMCP

from pvdiag.agent.state import CreateWorkOrderRequest, WorkOrderReceipt
from pvdiag.ops.store import WorkOrderStore


def create_server(store: WorkOrderStore) -> FastMCP:
    server = FastMCP(
        "pv-ops",
        instructions="Local simulated inspection orders only. Requires a previously recorded human decision bound to immutable contents.",
    )

    @server.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    def create_work_order(request: CreateWorkOrderRequest) -> WorkOrderReceipt:
        """Create one local inspection work order for an approved proposal; retries are idempotent."""
        return store.create_work_order(request)

    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args(argv)
    server = create_server(WorkOrderStore(args.db))
    if args.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(transport="http", host="127.0.0.1", port=args.port, show_banner=False)


if __name__ == "__main__":
    main()
