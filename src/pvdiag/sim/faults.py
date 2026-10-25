"""M1 F3 (inverter outage) and F4 (string loss) injection, without input mutation."""

from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self):
        lower, upper = window_bounds(self.start, self.end)
        object.__setattr__(self, "start", lower.isoformat())
        object.__setattr__(self, "end", upper.isoformat())
        if not self.device_id:
            raise ValueError("fault requires a target device")
        if self.type == "F3":
            if not self.poa_device_id or self.loss_fraction is not None:
                raise ValueError("F3 requires a POA sensor and no loss fraction")
        elif self.type == "F4":
            if self.loss_fraction not in (0.125, 0.25) or self.poa_device_id is not None:
                raise ValueError("F4 loss fraction must be 1/8 or 1/4, without a POA sensor")
        else:
            raise ValueError(f"unsupported fault type: {self.type}")

    def to_dict(self) -> dict:
        params = (
            {"loss_fraction": self.loss_fraction}
            if self.type == "F4"
            else {
                "poa_device_id": self.poa_device_id,
                "irradiance_threshold_wm2": DAYLIGHT_WM2,
                "alignment": "UTC 15min bin mean; no interpolation",
            }
        )
        return {
            "type": self.type,
            "device_id": self.device_id,
            "start": self.start,
            "end": self.end,
            "params": params,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FaultSpec:
        params = data["params"]
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
        raise ValueError("unsupported fault parameters")


def inject_fault(
    frame: pd.DataFrame, system_id: int, fault: FaultSpec
) -> tuple[pd.DataFrame, dict]:
    """Modify only the target's finite AC observations in [start, end).

    F3 zeroes AC when the chosen sensor's same-bin POA mean is strictly >200.
    Missing POA for a target bin fails instead of guessing. F4 multiplies AC by
    1-loss_fraction, including night-time values. NaNs stay NaN. Other metrics,
    devices and timestamps are untouched; DC/electrical dynamics are not modeled.
    """
    validate_frame(frame, system_id)
    lower, upper = utc_bound(fault.start), utc_bound(fault.end)
    target = (frame.device_id == fault.device_id) & (frame.metric == "ac_power_kw")
    target &= (frame.ts >= lower) & (frame.ts < upper)
    if not target.any():
        raise ValueError("no target AC observations in fault window")
    modify = target & frame.value.notna()
    if fault.type == "F3":
        poa = frame[(frame.device_id == fault.poa_device_id) & (frame.metric == "poa_wm2")]
        poa = poa[(poa.ts >= lower) & (poa.ts < upper)]
        bins = poa.set_index("ts").value.resample(FREQUENCY).mean()
        irradiance = frame.loc[target, "ts"].dt.floor(FREQUENCY).map(bins)
        if irradiance.isna().any():
            raise ValueError("missing POA in target bins; cannot establish an outage")
        modify.loc[target] &= irradiance > DAYLIGHT_WM2
    result = frame.copy(deep=True)
    result["value"] = result.value.astype(float)
    if fault.type == "F3":
        result.loc[modify, "value"] = 0.0
    else:
        result.loc[modify, "value"] *= 1 - fault.loss_fraction
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
        "changed_metric": "ac_power_kw",
    }
