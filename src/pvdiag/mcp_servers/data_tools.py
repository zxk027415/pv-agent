"""Bounded read-only data summaries backed by an immutable observation snapshot."""

from __future__ import annotations

import pandas as pd

from pvdiag.mcp_servers.models import (
    DeviceInfo,
    DevicesResult,
    Evidence,
    Point,
    QualityRequest,
    QualityResult,
    QualitySummary,
    SeriesSummary,
    SiteRequest,
    SiteResult,
    TimeseriesRequest,
    TimeseriesResult,
)
from pvdiag.mcp_servers.store import ObservationStore, finite, segments


class DataTools:
    def __init__(self, store: ObservationStore):
        self.store = store

    def list_devices(self, request: SiteRequest) -> Evidence[DevicesResult]:
        self.store.check(request)
        devices = []
        for device, metrics in sorted(self.store.catalog.items()):
            params = self.store.parameters.get(device)
            kind = (
                "inverter"
                if device in self.store.inverters
                else ("meter" if device.startswith("meter_") else "sensor")
            )
            devices.append(
                DeviceInfo(
                    device_id=device,
                    kind=kind,
                    metrics=sorted(metrics),
                    rated_ac_kw=params.rated_ac_kw if params else None,
                    rated_dc_kw=params.rated_dc_kw if params else None,
                    capacity_source=params.source if params else None,
                )
            )
        return self.store.evidence("list_devices", request, DevicesResult(devices=devices))

    def get_site_info(self, request: SiteRequest) -> Evidence[SiteResult]:
        self.store.check(request)
        metadata = self.store.metadata
        result = SiteResult(
            system_id=self.store.system_id,
            timezone=metadata["System"]["timezone_code"],
            latitude=metadata.get("Site", {}).get("latitude"),
            longitude=metadata.get("Site", {}).get("longitude"),
            mounts=[
                {k: m.get(k) for k in ("tilt", "azimuth", "tracking")}
                for m in metadata.get("Mount", {}).values()
            ],
            inverter_models=sorted(
                {m["model"] for m in metadata.get("Inverters", {}).values() if m.get("model")}
            ),
            module_models=sorted(
                {m["model"] for m in metadata.get("Modules", {}).values() if m.get("model")}
            ),
        )
        return self.store.evidence("get_site_info", request, result)

    def get_timeseries(self, request: TimeseriesRequest) -> Evidence[TimeseriesResult]:
        self.store.check(request)
        summaries = []
        remaining = request.max_points
        pairs = [(d, m) for d in request.device_ids for m in dict.fromkeys(request.metrics)]
        truncated = False
        for index, (device, metric) in enumerate(pairs):
            series = self.store.series(device, metric, request.start, request.end, request.resample)
            # Share a single total response budget across all requested series.
            allowance = remaining // (len(pairs) - index)
            count = min(allowance, len(series))
            positions = (
                []
                if count == 0
                else (
                    [0]
                    if count == 1
                    else [i * (len(series) - 1) // (count - 1) for i in range(count)]
                )
            )
            points = [
                Point(ts=series.index[i].isoformat(), value=finite(series.iloc[i]))
                for i in positions
            ]
            remaining -= len(points)
            truncated |= len(points) < len(series)
            summaries.append(
                SeriesSummary(
                    device_id=device,
                    metric=metric,
                    observed_bins=int(series.notna().sum()),
                    expected_bins=len(series),
                    coverage=float(series.notna().mean()),
                    mean=finite(series.mean()),
                    minimum=finite(series.min()),
                    maximum=finite(series.max()),
                    points=points,
                )
            )
        result = TimeseriesResult(
            resample=request.resample,
            series=summaries,
            truncated=truncated,
            returned_points=request.max_points - remaining,
        )
        return self.store.evidence(
            "get_timeseries",
            request,
            result,
            ["Points are evenly sampled from resampled means; use analytics for detection."],
        )

    def data_quality(self, request: QualityRequest) -> Evidence[QualityResult]:
        self.store.check(request)
        summaries = []
        remaining, truncated = 64, False
        pairs = [(d, m) for d in request.device_ids for m in dict.fromkeys(request.metrics)]
        for pair_index, (device, metric) in enumerate(pairs):
            series = self.store.series(device, metric, request.start, request.end)
            difference = series.diff().abs()
            median = difference.median()
            mad = (difference - median).abs().median()
            threshold = max(1.0, float(median + 6 * mad)) if pd.notna(mad) else float("inf")
            found = [
                segments(mask) for mask in (series.isna(), series.eq(0), difference > threshold)
            ]
            spans = [[], [], []]
            allowance = min(remaining, max(1, remaining // (len(pairs) - pair_index)))
            for offset in range(max(map(len, found), default=0)):
                for kind, candidates in enumerate(found):
                    if allowance and offset < len(candidates):
                        spans[kind].append(candidates[offset])
                        allowance -= 1
                        remaining -= 1
                if allowance == 0:
                    break
            truncated |= any(
                len(selected) < len(candidates)
                for selected, candidates in zip(spans, found, strict=True)
            )
            summaries.append(
                QualitySummary(
                    device_id=device,
                    metric=metric,
                    coverage=float(series.notna().mean()),
                    missing_bins=int(series.isna().sum()),
                    zero_bins=int(series.eq(0).sum()),
                    negative_bins=int(series.lt(0).sum()),
                    missing_spans=spans[0],
                    zero_spans=spans[1],
                    jump_spans=spans[2],
                )
            )
        result = QualityResult(series=summaries, spans_truncated=truncated)
        return self.store.evidence(
            "data_quality",
            request,
            result,
            ["Jumps use max(1 unit, median absolute change + 6 MAD); solar ramps can be flagged."],
        )
