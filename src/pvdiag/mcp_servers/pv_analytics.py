"""Run with python -m pvdiag.mcp_servers.pv_analytics --observations ... --metadata ..."""

from fastmcp import FastMCP

from pvdiag.mcp_servers.analytics_tools import AnalyticsTools
from pvdiag.mcp_servers.cli import READ_ONLY, serve
from pvdiag.mcp_servers.models import (
    AnomaliesResult,
    CompareRequest,
    DetectRequest,
    Evidence,
    ExpectedRequest,
    ExpectedResult,
    PatternsResult,
    PeerRequest,
    PeersResult,
    PeriodsResult,
)
from pvdiag.mcp_servers.store import ObservationStore


def create_server(store: ObservationStore) -> FastMCP:
    server = FastMCP(
        "pv-analytics",
        instructions="Read-only statistical and physical evidence. Use representative reference windows. Shapes are not root-cause verdicts.",
    )
    tools = AnalyticsTools(store)

    @server.tool(annotations=READ_ONLY)
    def peer_compare(request: PeerRequest) -> Evidence[PeersResult]:
        """Normalize each inverter against independent peers using an explicit reference window."""
        return tools.peer_compare(request)

    @server.tool(annotations=READ_ONLY)
    def detect_anomalies(request: DetectRequest) -> Evidence[AnomaliesResult]:
        """Find daylight zero power, missing data and rolling z-score anomalies in peer ratios."""
        return tools.detect_anomalies(request)

    @server.tool(annotations=READ_ONLY)
    def profile_pattern(request: PeerRequest) -> Evidence[PatternsResult]:
        """Describe an anomaly interval: zeros, relative loss, slope and local-hour repetition."""
        return tools.profile_pattern(request)

    @server.tool(annotations=READ_ONLY)
    def compare_periods(request: CompareRequest) -> Evidence[PeriodsResult]:
        """Compare disjoint windows' observed energy, coverage and peer-normalized performance."""
        return tools.compare_periods(request)

    @server.tool(annotations=READ_ONLY)
    def expected_power(request: ExpectedRequest) -> Evidence[ExpectedResult]:
        """Use pvlib PVWatts/Faiman with sourced ratings; return unavailable when parameters are absent."""
        return tools.expected_power(request)

    return server


if __name__ == "__main__":
    serve(create_server, "pv-analytics", 8002)
