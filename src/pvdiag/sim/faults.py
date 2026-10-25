"""Six deterministic fault primitives, without input mutation or interpolation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from zoneinfo import ZoneInfo

import pandas as pd

from pvdiag.data.normalize import utc_bound
from pvdiag.sim.baseline import DAYLIGHT_WM2, FREQUENCY, validate_frame, window_bounds


@dataclass(frozen=True)
class FaultSpec:
    type: str
    device_id: str
    start: str
    end: str
    loss_fraction: float | None = None
    poa_device_id: str | None = None
    soiling_rate_per_day: float | None = None
    cleaning_times: tuple[str, ...] = ()
    severity: float | None = None
    daily_window: tuple[str, str] | None = None
    timezone: str | None = None
    drift_per_day: float | None = None

    def __post_init__(self):
        lower, upper = window_bounds(self.start, self.end)
        object.__setattr__(self, "start", lower.isoformat())
        object.__setattr__(self, "end", upper.isoformat())
        if not self.device_id:
            raise ValueError("fault requires a target device")
        provided = {
            name
            for name in (
                "loss_fraction",
                "poa_device_id",
                "soiling_rate_per_day",
                "severity",
                "daily_window",
                "timezone",
                "drift_per_day",
            )
            if getattr(self, name) is not None
        }
        if self.cleaning_times:
            provided.add("cleaning_times")
        allowed = {
            "F1": {"soiling_rate_per_day", "cleaning_times"},
            "F2": {"severity", "daily_window", "timezone"},
            "F3": {"poa_device_id"},
            "F4": {"loss_fraction"},
            "F5": {"drift_per_day"},
            "F6": set(),
        }
        if self.type not in allowed or provided - allowed[self.type]:
            raise ValueError("unsupported fault type or incompatible parameters")
        if self.type == "F3":
            if not self.poa_device_id or self.loss_fraction is not None:
                raise ValueError("F3 requires a POA sensor and no loss fraction")
        elif self.type == "F4":
            if self.loss_fraction not in (0.125, 0.25) or self.poa_device_id is not None:
                raise ValueError("F4 loss fraction must be 1/8 or 1/4, without a POA sensor")
        elif self.type == "F1":
            rate = self.soiling_rate_per_day
            if rate is None or not isfinite(rate) or not 0.001 <= rate <= 0.005:
                raise ValueError("F1 rate must be 0.001–0.005 per day")
            clean = [utc_bound(t) for t in self.cleaning_times]
            if len(set(clean)) != len(clean) or any(
                not lower < t < upper or t.floor(FREQUENCY) != t for t in clean
            ):
                raise ValueError("cleaning times must be distinct, aligned and inside fault window")
            clean.sort()
            object.__setattr__(self, "cleaning_times", tuple(t.isoformat() for t in clean))
            resets = [lower, *clean, upper]
            longest = max((b - a).total_seconds() / 86400 for a, b in pairwise(resets))
            if longest * rate >= 1:
                raise ValueError("F1 loss would reach nonpositive power multiplier")
        elif self.type == "F2":
            if (
                self.severity is None
                or not isfinite(self.severity)
                or not 0.3 <= self.severity <= 0.7
            ):
                raise ValueError("F2 severity must be 0.3–0.7")
            if self.daily_window is None or len(self.daily_window) != 2 or not self.timezone:
                raise ValueError("F2 requires a daily window and explicit site timezone")
            object.__setattr__(self, "daily_window", tuple(self.daily_window))
            for clock in self.daily_window:
                if not re.fullmatch(r"(?:[01]\d|2[0-3]):(?:00|15|30|45)", clock):
                    raise ValueError("F2 daily window must use HH:MM quarter-hours")
            if self.daily_window[0] == self.daily_window[1]:
                raise ValueError("F2 daily window must have nonzero length")
            try:
                ZoneInfo(self.timezone)
            except (KeyError, ValueError) as exc:
                raise ValueError("invalid site timezone") from exc
        elif self.type == "F5" and (
            self.drift_per_day is None
            or not isfinite(self.drift_per_day)
            or not 0 < self.drift_per_day <= 0.02
        ):
            raise ValueError("F5 positive drift must be at most 0.02 per day")

    def to_dict(self) -> dict:
        if self.type == "F3":
            params = {
                "poa_device_id": self.poa_device_id,
                "irradiance_threshold_wm2": DAYLIGHT_WM2,
                "alignment": "UTC 15min bin mean; no interpolation",
            }
        elif self.type == "F4":
            params = {"loss_fraction": self.loss_fraction}
        elif self.type == "F1":
            params = {
                "rate_per_day": self.soiling_rate_per_day,
                "cleaning_times": list(self.cleaning_times),
            }
        elif self.type == "F2":
            params = {
                "severity": self.severity,
                "window": list(self.daily_window),
                "timezone": self.timezone,
            }
        elif self.type == "F5":
            params = {"drift_per_day": self.drift_per_day}
        else:
            params = {}
        return {
            "type": self.type,
            "device_id": self.device_id,
            "start": self.start,
            "end": self.end,
            "params": params,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FaultSpec:
        if set(data) != {"type", "device_id", "start", "end", "params"}:
            raise ValueError("unsupported fault fields")
        params = data["params"]
        if not isinstance(params, dict):
            raise TypeError("fault params must be an object")
        if data["type"] == "F3":
            allowed = {"poa_device_id", "irradiance_threshold_wm2", "alignment"}
            if set(params) - allowed or params.get("irradiance_threshold_wm2") != DAYLIGHT_WM2:
                raise ValueError("unsupported F3 parameters")
            if params.get("alignment") != "UTC 15min bin mean; no interpolation":
                raise ValueError("unsupported F3 alignment")
            return cls(
                data["type"],
                data["device_id"],
                data["start"],
                data["end"],
                poa_device_id=params["poa_device_id"],
            )
        if data["type"] == "F4" and set(params) == {"loss_fraction"}:
            return cls(
                data["type"],
                data["device_id"],
                data["start"],
                data["end"],
                loss_fraction=params["loss_fraction"],
            )
        common = (data["type"], data["device_id"], data["start"], data["end"])
        if data["type"] == "F1" and set(params) == {"rate_per_day", "cleaning_times"}:
            return cls(
                *common,
                soiling_rate_per_day=params["rate_per_day"],
                cleaning_times=tuple(params["cleaning_times"]),
            )
        if data["type"] == "F2" and set(params) == {"severity", "window", "timezone"}:
            return cls(
                *common,
                severity=params["severity"],
                daily_window=tuple(params["window"]),
                timezone=params["timezone"],
            )
        if data["type"] == "F5" and set(params) == {"drift_per_day"}:
            return cls(*common, drift_per_day=params["drift_per_day"])
        if data["type"] == "F6" and not params:
            return cls(*common)
        raise ValueError("unsupported fault parameters")


def inject_fault(
    frame: pd.DataFrame, system_id: int, fault: FaultSpec
) -> tuple[pd.DataFrame, dict]:
    """Apply a primitive within [start, end), preserving all timestamps/rows.

    F3 zeroes AC when the chosen sensor's same-bin POA mean is strictly >200.
    Missing POA for a target bin fails instead of guessing. F4 multiplies AC by
    1-loss_fraction, including night-time values. NaNs stay NaN. Other metrics,
    F1 resets a continuous elapsed-day ramp on cleaning; F2 uses site local time;
    F5 modifies only the chosen POA sensor; F6 replaces all target device metrics
    with NaN. DC/electrical dynamics of power-loss faults are not modeled.
    """
    validate_frame(frame, system_id)
    lower, upper = utc_bound(fault.start), utc_bound(fault.end)
    target = frame.device_id == fault.device_id
    if fault.type != "F6":
        target &= frame.metric == ("poa_wm2" if fault.type == "F5" else "ac_power_kw")
    target &= (frame.ts >= lower) & (frame.ts < upper)
    if not target.any():
        raise ValueError("no target AC/POA/device observations in fault window")
    modify = target & frame.value.notna()
    if fault.type == "F3":
        poa = frame[(frame.device_id == fault.poa_device_id) & (frame.metric == "poa_wm2")]
        poa = poa[(poa.ts >= lower) & (poa.ts < upper)]
        bins = poa.set_index("ts").value.resample(FREQUENCY).mean()
        irradiance = frame.loc[target, "ts"].dt.floor(FREQUENCY).map(bins)
        if irradiance.isna().any():
            raise ValueError("missing POA in target bins; cannot establish an outage")
        modify.loc[target] &= irradiance > DAYLIGHT_WM2
    if fault.type == "F2":
        local = frame.ts.dt.tz_convert(fault.timezone)
        minutes = local.dt.hour * 60 + local.dt.minute
        start_min, end_min = (int(t[:2]) * 60 + int(t[3:]) for t in fault.daily_window)
        daily = (
            (minutes >= start_min) & (minutes < end_min)
            if start_min < end_min
            else ((minutes >= start_min) | (minutes < end_min))
        )
        modify &= daily
    result = frame.copy(deep=True)
    result["value"] = result.value.astype(float)
    if fault.type == "F3":
        result.loc[modify, "value"] = 0.0
    elif fault.type == "F4":
        result.loc[modify, "value"] *= 1 - fault.loss_fraction
    elif fault.type == "F1":
        elapsed = (frame.loc[modify, "ts"] - lower).dt.total_seconds() / 86400
        for reset in fault.cleaning_times:
            since_reset = (frame.loc[modify, "ts"] - utc_bound(reset)).dt.total_seconds() / 86400
            elapsed = elapsed.where(since_reset < 0, since_reset)
        result.loc[modify, "value"] *= 1 - fault.soiling_rate_per_day * elapsed
    elif fault.type == "F2":
        result.loc[modify, "value"] *= 1 - fault.severity
    elif fault.type == "F5":
        days = (frame.loc[modify, "ts"] - lower).dt.total_seconds() / 86400
        result.loc[modify, "value"] *= 1 + fault.drift_per_day * days
    else:
        result.loc[modify, "value"] = float("nan")
    changed = result.value.ne(frame.value) & ~(result.value.isna() & frame.value.isna())
    if not changed.any():
        raise ValueError("fault would change no values; choose an observable window")
    return result, {
        "fault": fault.to_dict(),
        "changed_rows": int(changed.sum()),
        "first_changed_ts": result.loc[changed, "ts"].min().isoformat(),
        "last_changed_ts": result.loc[changed, "ts"].max().isoformat(),
        "input_rows": len(frame),
        "output_rows": len(result),
        "changed_metric": (
            str(result.loc[changed, "metric"].iloc[0])
            if result.loc[changed, "metric"].nunique() == 1
            else "multiple_device_metrics"
        ),
    }
