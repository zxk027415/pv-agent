"""Read a single observation dataset; never open scenario truth or audit files."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from pvdiag.data.normalize import channels_from_metadata
from pvdiag.mcp_servers.models import (
    DeviceParameters,
    Evidence,
    Model,
    SiteRequest,
    Span,
    WindowRequest,
)
from pvdiag.sim.baseline import validate_frame


def finite(value) -> float | None:
    return float(value) if pd.notna(value) and abs(value) != float("inf") else None


def segments(mask: pd.Series) -> list[Span]:
    """Contiguous half-open 15-minute spans; a missing bin breaks a span."""
    spans = []
    selected = mask.fillna(False)
    start = previous = None
    for ts in selected.index[selected]:
        if previous is None or ts - previous != pd.Timedelta(minutes=15):
            if start is not None:
                spans.append(
                    Span(
                        start=start.isoformat(),
                        end=(previous + pd.Timedelta(minutes=15)).isoformat(),
                        bins=int((previous - start) / pd.Timedelta(minutes=15)) + 1,
                    )
                )
            start = ts
        previous = ts
    if start is not None:
        spans.append(
            Span(
                start=start.isoformat(),
                end=(previous + pd.Timedelta(minutes=15)).isoformat(),
                bins=int((previous - start) / pd.Timedelta(minutes=15)) + 1,
            )
        )
    return spans


class ObservationStore:
    def __init__(self, observations: Path, metadata: Path, parameters: Path | None = None):
        observation_bytes, metadata_bytes = observations.read_bytes(), metadata.read_bytes()
        parameter_bytes = parameters.read_bytes() if parameters else b""
        self.frame = pd.read_parquet(io.BytesIO(observation_bytes))
        self.metadata = json.loads(metadata_bytes)
        self.system_id = int(self.metadata["System"]["system_id"])
        validate_frame(self.frame, self.system_id)
        # Hash exactly the bytes loaded into this read-only snapshot, plus model
        # settings. No path or scenario ID (which can contain a fault label) is exposed.
        digest = hashlib.sha256()
        for content in (observation_bytes, metadata_bytes, parameter_bytes):
            digest.update(hashlib.sha256(content).digest())
        self.source_sha256 = digest.hexdigest()
        settings = json.loads(parameter_bytes) if parameter_bytes else {}
        self.parameters = {
            device: DeviceParameters.model_validate(params) for device, params in settings.items()
        }
        self.catalog = self.frame.groupby("device_id").metric.unique().to_dict()
        self.inverters = {
            c.device_id
            for c in channels_from_metadata(self.metadata)
            if c.metric == "ac_power_kw" and c.device_id.startswith("inv")
        }

    def check(self, request: SiteRequest) -> None:
        if request.system_id != self.system_id:
            raise ValueError("system_id is outside the loaded observation scope")
        if isinstance(request, WindowRequest):
            unknown = set(request.device_ids) - set(self.catalog)
            if unknown:
                raise ValueError(f"devices outside observation scope: {sorted(unknown)}")

    def sensor(self, device: str, metric: str) -> None:
        if metric not in self.catalog.get(device, []):
            raise ValueError(f"{device} has no observed {metric} channel")

    def series(self, device: str, metric: str, start, end, freq="15min") -> pd.Series:
        lower, upper = pd.Timestamp(start), pd.Timestamp(end)
        rows = self.frame[
            (self.frame.device_id == device)
            & (self.frame.metric == metric)
            & (self.frame.ts >= lower)
            & (self.frame.ts < upper)
        ]
        grid = pd.date_range(lower, upper, freq=freq, inclusive="left")
        return rows.set_index("ts").value.resample(freq, origin=lower).mean().reindex(grid)

    def evidence(
        self, tool: str, request: Model, result: Model, warnings=(), *, version="m1-v1"
    ) -> Evidence:
        payload = {
            "tool": tool,
            "version": version,
            "source_sha256": self.source_sha256,
            "request": request.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "warnings": list(warnings),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        return Evidence[type(result)](
            evidence_id=f"ev_{digest[:24]}", **(payload | {"result": result})
        )
