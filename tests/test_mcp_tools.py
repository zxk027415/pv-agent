import asyncio
import json
import socket
import sys
import time

import pandas as pd
import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from pydantic import ValidationError

from pvdiag.mcp_servers.analytics_tools import AnalyticsTools
from pvdiag.mcp_servers.data_tools import DataTools
from pvdiag.mcp_servers.models import (
    CompareRequest,
    DetectRequest,
    ExpectedRequest,
    PeerRequest,
    QualityRequest,
    SiteRequest,
    TimeseriesRequest,
)
from pvdiag.mcp_servers.pv_analytics import create_server as analytics_server
from pvdiag.mcp_servers.pv_data import create_server as data_server
from pvdiag.mcp_servers.store import ObservationStore, segments


@pytest.fixture
def files(tmp_path):
    ts = pd.date_range("2019-01-01", periods=96 * 7, freq="15min", tz="UTC")
    day = (ts.hour >= 16) & (ts.hour < 23)
    pieces = []
    for device, metric, values in (
        ("inv01", "ac_power_kw", day.astype(float) * 10),
        ("inv02", "ac_power_kw", day.astype(float) * 20),
        ("inv03", "ac_power_kw", day.astype(float) * 30),
        ("poa", "poa_wm2", day.astype(float) * 600),
        ("ambient", "ambient_c", 20.0),
    ):
        pieces.append(
            pd.DataFrame(
                {
                    "ts": ts,
                    "system_id": 2107,
                    "device_id": device,
                    "metric": metric,
                    "value": values,
                }
            )
        )
    frame = pd.concat(pieces, ignore_index=True)
    observation = tmp_path / "FAULT_LABEL_MUST_NOT_LEAK.parquet"
    frame.to_parquet(observation, index=False)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "System": {"system_id": 2107, "timezone_code": "PST8PDT"},
                "Site": {"latitude": 39.0, "longitude": -122.0},
                "Metrics": {
                    f"inv_{i:02d}_ac_power_inv_{i}": {
                        "common_name": "AC power",
                        "units": "kw",
                        "source_type": "INVERTER",
                        "metric_id": i,
                        "source_id": None,
                    }
                    for i in range(1, 5)
                },
            }
        )
    )
    # Adjacent files deliberately cannot be parsed; store must never touch them.
    (tmp_path / "truth.json").write_text("DO NOT READ")
    (tmp_path / "audit.json").write_text("DO NOT READ")
    return observation, metadata


def window(**updates):
    return {
        "system_id": 2107,
        "device_ids": ["inv01", "inv02", "inv03"],
        "start": "2019-01-01",
        "end": "2019-01-08",
    } | updates


def peer(**updates):
    return (
        window(poa_device_id="poa", reference_start="2019-01-01", reference_end="2019-01-03")
        | updates
    )


def change_power(files, fraction=None, missing=False):
    frame = pd.read_parquet(files[0])
    target = (frame.device_id == "inv01") & (frame.ts >= "2019-01-03")
    if fraction is not None:
        frame.loc[target, "value"] *= fraction
    if missing:
        target &= frame.ts.between(
            "2019-01-03 18:00+00:00", "2019-01-03 19:00+00:00", inclusive="left"
        )
        frame = frame.loc[~target]
    frame.to_parquet(files[0], index=False)


def test_catalog_scope_unknown_capacity_and_stable_evidence(files):
    service = DataTools(ObservationStore(*files))
    request = SiteRequest(system_id=2107)
    first = service.list_devices(request)
    assert first.evidence_id == service.list_devices(request).evidence_id
    assert {d.device_id for d in first.result.devices} == {
        "inv01",
        "inv02",
        "inv03",
        "poa",
        "ambient",
    }
    assert all(d.rated_ac_kw is None for d in first.result.devices)
    assert "FAULT_LABEL" not in first.model_dump_json()
    assert service.get_site_info(request).result.latitude == 39.0
    with pytest.raises(ValueError, match="system_id"):
        service.list_devices(SiteRequest(system_id=9069))
    with pytest.raises(ValueError, match="outside observation scope"):
        service.get_timeseries(TimeseriesRequest(**window(device_ids=["inv04"])))


@pytest.mark.parametrize(
    "updates",
    [
        {"end": "2020-01-01"},
        {"end": "2018-01-01"},
        {"device_ids": ["inv01", "inv01"]},
        {"start": "2019-01-01 00:01"},
        {"metrics": ["fake"]},
        {"max_points": 241},
        {"unknown": "forbidden"},
    ],
)
def test_request_guards(updates):
    with pytest.raises(ValidationError):
        TimeseriesRequest(**window(**updates))


def test_timeseries_global_budget_missing_metric_and_utc(files):
    tools = DataTools(ObservationStore(*files))
    request = TimeseriesRequest(**window(metrics=["ac_power_kw", "dc_power_kw"], max_points=5))
    result = tools.get_timeseries(request).result
    assert result.returned_points == sum(len(s.points) for s in result.series) <= 5
    assert result.truncated
    missing = [s for s in result.series if s.metric == "dc_power_kw"]
    assert all(s.coverage == 0 and s.mean is None for s in missing)
    assert all(point.ts.endswith("+00:00") for s in result.series for point in s.points)
    json.dumps(result.model_dump(), allow_nan=False)


def test_quality_distinguishes_missing_from_zero_and_reports_exclusive_end(files):
    change_power(files, fraction=0, missing=True)
    request = QualityRequest(
        **window(device_ids=["inv01"], start="2019-01-03 16:00", end="2019-01-03 20:00")
    )
    quality = DataTools(ObservationStore(*files)).data_quality(request).result.series[0]
    assert quality.missing_bins == 4
    assert quality.zero_bins == 12
    assert quality.coverage == 0.75
    assert quality.missing_spans[0].start == "2019-01-03T18:00:00+00:00"
    assert quality.missing_spans[0].end == "2019-01-03T19:00:00+00:00"


def test_peer_reference_preserves_f4_loss_and_periods_do_not_overlap(files):
    change_power(files, fraction=0.875)
    tools = AnalyticsTools(ObservationStore(*files))
    request = PeerRequest(**peer(start="2019-01-03"))
    result = tools.peer_compare(request).result
    target = result.devices[0]
    assert target.median_ratio == pytest.approx(0.875)
    assert target.low_ratio_fraction == 1
    assert target.low_spans
    assert all(d.median_ratio > 1 for d in result.devices[1:])
    compare = CompareRequest(**peer(start="2019-01-03", end="2019-01-05"))
    energy = tools.compare_periods(compare).result.devices[0]
    assert energy.energy_ratio == pytest.approx(0.875)
    assert energy.reference_coverage == energy.current_coverage == 1
    assert energy.median_peer_ratio == pytest.approx(0.875)
    with pytest.raises(ValidationError, match="must not overlap"):
        CompareRequest(**peer())


def test_absent_reference_returns_unavailable_ratios_without_guessing(files):
    tools = AnalyticsTools(ObservationStore(*files))
    result = tools.peer_compare(
        PeerRequest(**peer(reference_start="2018-01-01", reference_end="2018-01-02"))
    ).result
    assert all(d.reference_ratio is None and d.median_ratio is None for d in result.devices)
    assert all(d.low_spans == [] for d in result.devices)


def test_anomalies_and_profile_find_step_and_zero_without_fault_truth(files):
    change_power(files, fraction=0.75)
    tools = AnalyticsTools(ObservationStore(*files))
    anomalies = tools.detect_anomalies(DetectRequest(**peer())).result
    assert any(a.device_id == "inv01" and a.kind == "rolling_zscore" for a in anomalies.anomalies)
    pattern = tools.profile_pattern(PeerRequest(**peer(start="2019-01-03"))).result.devices[0]
    assert pattern.shape == "level_shift"
    assert pattern.median_peer_ratio == pytest.approx(0.75)
    change_power(files, fraction=0)
    tools = AnalyticsTools(ObservationStore(*files))
    anomalies = tools.detect_anomalies(DetectRequest(**peer())).result
    assert any(a.device_id == "inv01" and a.kind == "zero_power" for a in anomalies.anomalies)
    pattern = tools.profile_pattern(PeerRequest(**peer(start="2019-01-03"))).result.devices[0]
    assert pattern.shape == "zero_output"
    assert pattern.zero_fraction == 1


def test_expected_power_requires_sourced_parameters_and_clips_to_ac_rating(files, tmp_path):
    request = ExpectedRequest(
        **window(device_ids=["inv01"], poa_device_id="poa", ambient_device_id="ambient")
    )
    tools = AnalyticsTools(ObservationStore(*files))
    assert tools.expected_power(request).result.devices[0].status == "unavailable"
    parameters = tmp_path / "parameters.json"
    parameters.write_text(
        json.dumps(
            {
                "inv01": {
                    "rated_dc_kw": 1000,
                    "rated_ac_kw": 10,
                    "gamma_pdc": -0.004,
                    "source": "synthetic clipping test",
                }
            }
        )
    )
    modeled = AnalyticsTools(ObservationStore(*files, parameters)).expected_power(request)
    result = modeled.result.devices[0]
    assert result.status == "available"
    assert result.parameters.source == "synthetic clipping test"
    assert result.parameters.gamma_pdc == -0.004
    assert result.median_performance_ratio == pytest.approx(1.0)
    assert result.expected_known_energy_kwh == pytest.approx(7 * 7 * 10)
    assert len(result.performance_ratio_points) == 120
    assert result.points_truncated
    assert all(p.value == pytest.approx(1) for p in result.performance_ratio_points)
    assert modeled.source_sha256 != tools.expected_power(request).source_sha256


def test_snapshot_is_immutable_and_evidence_changes_with_new_observations(files):
    old = DataTools(ObservationStore(*files))
    request = TimeseriesRequest(**window())
    before = old.get_timeseries(request)
    change_power(files, fraction=0.75)
    assert old.get_timeseries(request).model_dump() == before.model_dump()
    assert (
        DataTools(ObservationStore(*files)).get_timeseries(request).evidence_id
        != before.evidence_id
    )


def test_segments_break_at_gaps_and_bounded_responses_mark_truncation(files):
    index = pd.to_datetime(["2019-01-01 00:00Z", "2019-01-01 00:30Z"])
    assert len(segments(pd.Series(True, index=index))) == 2
    frame = pd.read_parquet(files[0])
    rows = frame[frame.device_id == "inv01"].index
    frame.loc[rows[::2], "value"] = float("nan")
    frame.to_parquet(files[0], index=False)
    result = DataTools(ObservationStore(*files)).data_quality(QualityRequest(**window())).result
    assert result.spans_truncated
    assert (
        sum(len(s.missing_spans) + len(s.zero_spans) + len(s.jump_spans) for s in result.series)
        <= 64
    )


def test_noisy_first_device_cannot_consume_all_later_device_spans(files):
    frame = pd.read_parquet(files[0])
    rows = frame[frame.device_id == "inv01"].index
    frame.loc[rows[::8], "value"] += 100
    frame.to_parquet(files[0], index=False)
    result = DataTools(ObservationStore(*files)).data_quality(QualityRequest(**window())).result
    assert result.spans_truncated
    assert all(s.zero_spans for s in result.series)
    assert (
        sum(len(s.missing_spans) + len(s.zero_spans) + len(s.jump_spans) for s in result.series)
        <= 64
    )


def test_mcp_tools_publish_schemas_readonly_annotations_and_protocol_results(files):
    async def verify():
        store = ObservationStore(*files)
        for server, expected in (
            (
                data_server(store),
                {"list_devices", "get_site_info", "get_timeseries", "data_quality"},
            ),
            (
                analytics_server(store),
                {
                    "peer_compare",
                    "detect_anomalies",
                    "profile_pattern",
                    "compare_periods",
                    "expected_power",
                },
            ),
        ):
            async with Client(server) as client:
                listed = await client.list_tools()
                assert {tool.name for tool in listed} == expected
                assert all(
                    tool.annotations.read_only_hint and not tool.annotations.destructive_hint
                    for tool in listed
                )
                assert all(tool.output_schema and tool.input_schema for tool in listed)
                if "list_devices" in expected:
                    response = await client.call_tool(
                        "list_devices", {"request": {"system_id": 2107}}
                    )
                else:
                    response = await client.call_tool("peer_compare", {"request": peer()})
                assert response.structured_content["evidence_id"].startswith("ev_")

    asyncio.run(verify())


def test_stdio_transport_reads_real_process_without_mutating_inputs(files):
    async def verify():
        observation_bytes, metadata_bytes = files[0].read_bytes(), files[1].read_bytes()
        transport = StdioTransport(
            sys.executable,
            [
                "-m",
                "pvdiag.mcp_servers.pv_data",
                "--observations",
                str(files[0]),
                "--metadata",
                str(files[1]),
            ],
        )
        async with Client(transport) as client:
            response = await client.call_tool("data_quality", {"request": window()})
            assert response.structured_content["result"]["series"][0]["coverage"] == 1
        assert files[0].read_bytes() == observation_bytes
        assert files[1].read_bytes() == metadata_bytes

    asyncio.run(verify())


def test_pattern_missing_readings_are_not_counted_as_nonzero(files):
    change_power(files, fraction=0, missing=True)
    tools = AnalyticsTools(ObservationStore(*files))
    pattern = tools.profile_pattern(
        PeerRequest(**peer(start="2019-01-03 16:00", end="2019-01-03 20:00"))
    ).result.devices[0]
    assert pattern.daylight_bins == 16
    assert pattern.observed_daylight_bins == 12
    assert pattern.zero_fraction == 1


def test_pattern_loss_hours_require_recurrence_and_off_loss_excludes_them(files):
    frame = pd.read_parquet(files[0])
    inv01 = (frame.device_id == "inv01") & (frame.ts >= "2019-01-03")
    frame.loc[inv01 & (frame.ts.dt.hour == 17), "value"] *= 0.5
    sporadic = inv01 & frame.ts.between(
        "2019-01-04 21:00+00:00", "2019-01-04 22:00+00:00", inclusive="left"
    )
    frame.loc[sporadic, "value"] *= 0.5
    frame.to_parquet(files[0], index=False)
    tools = AnalyticsTools(ObservationStore(*files))
    pattern = tools.profile_pattern(PeerRequest(**peer(start="2019-01-03"))).result.devices[0]
    # A single-day dip is not a recurring loss hour, and off-loss is not floored at 0.9.
    assert pattern.loss_hour_count == 1
    assert pattern.off_loss_median_ratio == pytest.approx(1.0)


def test_resample_labels_stay_inside_the_requested_window(files):
    tools = DataTools(ObservationStore(*files))
    result = tools.get_timeseries(
        TimeseriesRequest(**window(start="2019-01-01 00:15", end="2019-01-01 01:15"))
    ).result
    assert all(s.expected_bins == 1 for s in result.series)
    assert all(p.ts == "2019-01-01T00:15:00+00:00" for s in result.series for p in s.points)


def test_streamable_http_transport_is_callable(files, tmp_path):
    async def verify():
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        log = tmp_path / "http-server.log"
        with log.open("w") as output:
            process = await asyncio.create_subprocess_exec(
                *[
                    sys.executable,
                    "-m",
                    "pvdiag.mcp_servers.pv_analytics",
                    "--observations",
                    str(files[0]),
                    "--metadata",
                    str(files[1]),
                    "--transport",
                    "http",
                    "--port",
                    str(port),
                ],
                stdout=output,
                stderr=output,
            )
            try:
                deadline = time.monotonic() + 10
                while True:
                    assert process.returncode is None, log.read_text()
                    try:
                        _, writer = await asyncio.open_connection("127.0.0.1", port)
                        writer.close()
                        await writer.wait_closed()
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise
                        await asyncio.sleep(0.1)
                async with Client(f"http://127.0.0.1:{port}/mcp", timeout=2) as client:
                    result = await client.call_tool("peer_compare", {"request": peer()})
                    assert result.structured_content["result"]["devices"][0][
                        "median_ratio"
                    ] == pytest.approx(1)
            finally:
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5)

    asyncio.run(verify())


def test_timeseries_and_quality_skip_channels_absent_from_the_snapshot(tmp_path):
    import json as _json

    import pandas as _pd

    from pvdiag.mcp_servers.data_tools import DataTools
    from pvdiag.mcp_servers.models import QualityRequest, TimeseriesRequest
    from pvdiag.mcp_servers.store import ObservationStore

    ts = _pd.date_range("2019-01-01", periods=96, freq="15min", tz="UTC")
    rows = [
        _pd.DataFrame({"ts": ts, "system_id": 2107, "device_id": d, "metric": m, "value": v})
        for d, m, v in (("inv01", "ac_power_kw", 1.0), ("poa", "poa_wm2", 500.0), ("inv01", "module_c", float("nan")))
    ]
    path = tmp_path / "o.parquet"
    _pd.concat(rows).to_parquet(path, index=False)
    meta = tmp_path / "m.json"
    meta.write_text(_json.dumps({"System": {"system_id": 2107, "timezone_code": "PST8PDT"}, "Metrics": {}}))
    tools = DataTools(ObservationStore(path, meta))
    window = {"system_id": 2107, "device_ids": ["inv01", "poa"], "start": "2019-01-01", "end": "2019-01-02"}
    series = tools.get_timeseries(TimeseriesRequest(**window, metrics=["ac_power_kw", "poa_wm2", "module_c"])).result
    assert [(s.device_id, s.metric) for s in series.series] == [
        ("inv01", "ac_power_kw"), ("inv01", "module_c"), ("poa", "poa_wm2")
    ]
    assert series.unavailable == ["inv01:poa_wm2", "poa:ac_power_kw", "poa:module_c"]
    # A catalogued but entirely missing channel is still reported, with zero coverage.
    assert next(s for s in series.series if s.metric == "module_c").coverage == 0.0
    quality = tools.data_quality(QualityRequest(**window, metrics=["poa_wm2"])).result
    assert [(s.device_id, s.metric) for s in quality.series] == [("poa", "poa_wm2")]
    assert quality.unavailable == ["inv01:poa_wm2"]
