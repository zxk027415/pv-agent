"""Run with python -m pvdiag.mcp_servers.pv_data --observations ... --metadata ..."""

from fastmcp import FastMCP

from pvdiag.mcp_servers.cli import READ_ONLY, serve
from pvdiag.mcp_servers.data_tools import DataTools
from pvdiag.mcp_servers.models import (
    DevicesResult,
    Evidence,
    QualityRequest,
    QualityResult,
    SiteRequest,
    SiteResult,
    TimeseriesRequest,
    TimeseriesResult,
)
from pvdiag.mcp_servers.store import ObservationStore


def create_server(store: ObservationStore) -> FastMCP:
    server = FastMCP(
        "pv-data",
        instructions="Read-only PV observation summaries. Never infer missing capacity or read benchmark truth.",
    )
    tools = DataTools(store)

    @server.tool(annotations=READ_ONLY)
    def list_devices(request: SiteRequest) -> Evidence[DevicesResult]:
        """List only devices visible in this observation snapshot; absent ratings remain null."""
        return tools.list_devices(request)

    @server.tool(annotations=READ_ONLY)
    def get_site_info(request: SiteRequest) -> Evidence[SiteResult]:
        """Get source metadata coordinates, timezone, mounting geometry and equipment models."""
        return tools.get_site_info(request)

    @server.tool(annotations=READ_ONLY)
    def get_timeseries(request: TimeseriesRequest) -> Evidence[TimeseriesResult]:
        """Get bounded statistics and sampled means in UTC [start,end); never a raw table."""
        return tools.get_timeseries(request)

    @server.tool(annotations=READ_ONLY)
    def data_quality(request: QualityRequest) -> Evidence[QualityResult]:
        """Distinguish absent/NaN bins from zeros and report gaps, zero spans and robust jumps."""
        return tools.data_quality(request)

    return server


if __name__ == "__main__":
    serve(create_server, "pv-data", 8001)
