"""Explicit observation/metadata paths; no automatic discovery of bench truth."""

import argparse
from pathlib import Path

from pvdiag.mcp_servers.store import ObservationStore

READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


def serve(factory, name: str, port: int):
    parser = argparse.ArgumentParser(description=f"Read-only {name} MCP server")
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument(
        "--parameters", type=Path, help="optional sourced per-device physical parameters"
    )
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    server = factory(ObservationStore(args.observations, args.metadata, args.parameters))
    if args.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(transport="http", host="127.0.0.1", port=args.port, show_banner=False)
