"""Validated tool contracts; summaries are bounded and never contain raw tables."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pvdiag.data.normalize import utc_bound

Metric = Literal["ac_power_kw", "dc_power_kw", "poa_wm2", "ambient_c", "module_c"]
T = TypeVar("T")


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SiteRequest(Model):
    system_id: int = Field(gt=0)


class WindowRequest(SiteRequest):
    device_ids: list[str] = Field(min_length=1, max_length=24)
    start: datetime
    end: datetime

    @field_validator("start", "end", mode="before")
    @classmethod
    def utc(cls, value):
        return utc_bound(value).to_pydatetime()

    @field_validator("device_ids")
    @classmethod
    def unique_devices(cls, value):
        if len(value) != len(set(value)) or any(not device for device in value):
            raise ValueError("device_ids must be nonempty and unique")
        return value

    @model_validator(mode="after")
    def valid_window(self):
        validate_window(self.start, self.end)
        return self


def validate_window(start: datetime, end: datetime) -> None:
    if not timedelta(0) < end - start <= timedelta(days=90):
        raise ValueError("window must increase and not exceed 90 days")
    for ts in (start, end):
        if ts.minute % 15 or ts.second or ts.microsecond:
            raise ValueError("window bounds must align to UTC 15-minute bins")


class TimeseriesRequest(WindowRequest):
    metrics: list[Metric] = Field(
        default_factory=lambda: ["ac_power_kw"], min_length=1, max_length=5
    )
    resample: Literal["15min", "1h", "1D"] = "1h"
    max_points: int = Field(default=120, ge=1, le=240)


class QualityRequest(WindowRequest):
    metrics: list[Metric] = Field(
        default_factory=lambda: ["ac_power_kw"], min_length=1, max_length=5
    )


class PeerRequest(WindowRequest):
    poa_device_id: str
    reference_start: datetime
    reference_end: datetime

    @field_validator("reference_start", "reference_end", mode="before")
    @classmethod
    def reference_utc(cls, value):
        return utc_bound(value).to_pydatetime()

    @model_validator(mode="after")
    def valid_reference(self):
        validate_window(self.reference_start, self.reference_end)
        if len(self.device_ids) < 3:
            raise ValueError("peer analysis requires at least three inverters")
        return self


class DetectRequest(PeerRequest):
    method: Literal["rolling_zscore"] = "rolling_zscore"


class CompareRequest(PeerRequest):
    @model_validator(mode="after")
    def disjoint_periods(self):
        if max(self.start, self.reference_start) < min(self.end, self.reference_end):
            raise ValueError("comparison periods must not overlap")
        return self


class ExpectedRequest(WindowRequest):
    poa_device_id: str
    ambient_device_id: str
    max_points: int = Field(default=120, ge=1, le=240)


class DeviceParameters(Model):
    rated_dc_kw: float = Field(gt=0)
    rated_ac_kw: float = Field(gt=0)
    gamma_pdc: float = Field(lt=0, gt=-0.02)
    source: str = Field(min_length=1)


class Evidence(Model, Generic[T]):
    evidence_id: str
    tool: str
    version: str = "m1-v1"
    source_sha256: str
    request: dict
    result: T
    warnings: list[str] = Field(default_factory=list)


class DeviceInfo(Model):
    device_id: str
    kind: Literal["inverter", "meter", "sensor"]
    metrics: list[str]
    rated_ac_kw: float | None = None
    rated_dc_kw: float | None = None
    capacity_source: str | None = None


class DevicesResult(Model):
    devices: list[DeviceInfo]


class SiteResult(Model):
    system_id: int
    timezone: str
    latitude: float | None
    longitude: float | None
    mounts: list[dict]
    inverter_models: list[str]
    module_models: list[str]


class Point(Model):
    ts: str
    value: float | None


class SeriesSummary(Model):
    device_id: str
    metric: str
    observed_bins: int
    expected_bins: int
    coverage: float
    mean: float | None
    minimum: float | None
    maximum: float | None
    points: list[Point]


class TimeseriesResult(Model):
    resample: str
    series: list[SeriesSummary]
    truncated: bool
    returned_points: int


class Span(Model):
    start: str
    end: str
    bins: int


class QualitySummary(Model):
    device_id: str
    metric: str
    coverage: float
    missing_bins: int
    zero_bins: int
    negative_bins: int
    missing_spans: list[Span]
    zero_spans: list[Span]
    jump_spans: list[Span]


class QualityResult(Model):
    resample: str = "15min"
    series: list[QualitySummary]
    spans_truncated: bool


class PeerSummary(Model):
    device_id: str
    reference_bins: int
    reference_ratio: float | None
    comparable_daylight_bins: int
    median_ratio: float | None
    low_ratio_fraction: float | None
    low_spans: list[Span]
    high_spans: list[Span]


class PeersResult(Model):
    normalization: str = "leave-one-out peer ratio / reference-period median ratio"
    devices: list[PeerSummary]
    spans_truncated: bool


class Anomaly(Model):
    device_id: str
    kind: Literal["data_gap", "zero_power", "rolling_zscore"]
    span: Span
    score: float | None = None


class AnomaliesResult(Model):
    method: str
    anomalies: list[Anomaly]
    truncated: bool


class Pattern(Model):
    device_id: str
    daylight_bins: int
    observed_daylight_bins: int
    zero_fraction: float | None
    median_peer_ratio: float | None
    ratio_iqr: float | None
    slope_per_day: float | None
    anomalous_days: int
    peak_loss_local_hour: int | None
    shape: Literal["zero_output", "level_shift", "gradual", "time_localized", "no_clear_pattern"]
    loss_hour_count: int = 0
    off_loss_median_ratio: float | None = None
    largest_daily_recovery: float | None = None
    recovery_ts: str | None = None
    pre_recovery_slope_per_day: float | None = None


class PatternsResult(Model):
    local_timezone: str
    devices: list[Pattern]


class PeriodSummary(Model):
    device_id: str
    reference_known_energy_kwh: float
    current_known_energy_kwh: float
    reference_coverage: float
    current_coverage: float
    energy_ratio: float | None
    median_peer_ratio: float | None


class PeriodsResult(Model):
    integration: str = "sum of observed UTC 15min mean kW * 0.25h; missing bins excluded"
    devices: list[PeriodSummary]


class ExpectedSummary(Model):
    device_id: str
    status: Literal["available", "unavailable"]
    parameters: DeviceParameters | None = None
    reason: str | None = None
    comparable_bins: int = 0
    expected_known_energy_kwh: float | None = None
    median_performance_ratio: float | None = None
    mean_deviation_kw: float | None = None
    performance_ratio_points: list[Point] = Field(default_factory=list)
    points_truncated: bool = False


class ExpectedResult(Model):
    model: str = "pvlib PVWatts DC/AC + Faiman cell temperature"
    assumptions: list[str]
    devices: list[ExpectedSummary]
