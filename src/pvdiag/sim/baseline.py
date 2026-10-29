"""Conservative screening of real data before synthetic fault injection.

This is a data selection heuristic, not a diagnosis algorithm or a certificate of
healthy operation. Peer ratios use empirical 95th-percentile power scales, not
unavailable nameplate capacities. No missing values are interpolated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pvdiag.data.normalize import KEY, SCHEMA, utc_bound

FREQUENCY = "15min"
DAYLIGHT_WM2 = 200.0


def validate_frame(frame: pd.DataFrame, system_id: int) -> None:
    if not set(SCHEMA) <= set(frame.columns) or frame.empty:
        raise ValueError("expected nonempty normalized long table")
    if str(frame.ts.dt.tz) != "UTC" or frame.ts.isna().any():
        raise ValueError("timestamps must be nonmissing UTC values")
    if set(frame.system_id) != {system_id}:
        raise ValueError("input must contain only the requested system")
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate normalized keys")
    if frame.value.isin([float("inf"), -float("inf")]).any():
        raise ValueError("infinite observation")


def window_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    lower, upper = utc_bound(start), utc_bound(end)
    if lower >= upper or lower.floor(FREQUENCY) != lower or upper.floor(FREQUENCY) != upper:
        raise ValueError("window must increase and align to UTC 15-minute boundaries")
    return lower, upper


def aligned_signals(
    frame: pd.DataFrame,
    start: str,
    end: str,
    device_ids: tuple[str, ...],
    poa_device_id: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Use means in half-open UTC bins; absent bins remain NaN."""
    lower, upper = window_bounds(start, end)
    grid = pd.date_range(lower, upper, freq=FREQUENCY, inclusive="left")
    window = frame[(frame.ts >= lower) & (frame.ts < upper)]
    power_rows = window[(window.metric == "ac_power_kw") & window.device_id.isin(device_ids)]
    if power_rows.empty:
        power = pd.DataFrame(index=grid, columns=list(device_ids), dtype=float)
    else:
        power = power_rows.pivot(index="ts", columns="device_id", values="value")
        power = power.resample(FREQUENCY).mean().reindex(index=grid, columns=list(device_ids))
    poa_rows = window[(window.metric == "poa_wm2") & (window.device_id == poa_device_id)]
    poa = poa_rows.set_index("ts").value.resample(FREQUENCY).mean().reindex(grid)
    return power, poa


def screen_baseline(
    frame: pd.DataFrame,
    system_id: int,
    start: str,
    end: str,
    device_ids: tuple[str, ...],
    poa_device_id: str,
    max_peer_outlier_fraction: float = 0.0,
    daytime_site: tuple[float, float] | None = None,
) -> dict:
    """Select devices with full daylight data and no obvious peer outliers.

    A window needs >=99% overall coverage and >=16 high-irradiance bins.
    A device is rejected for any daylight missing/near-zero/negative value, or
    any daylight normalized leave-one-out peer ratio outside [0.85, 1.20].
    At least two valid peers are required for every daylight comparison.

    ``max_peer_outlier_fraction`` > 0 is an explicit, separately versioned rule for
    large sites, where passing cloud shadows desynchronize distant inverters: up to
    floor(fraction x daylight bins) peer-ratio outliers are tolerated. Every other
    criterion is unchanged; the default keeps the strict m1-v1 rule.

    ``daytime_site=(latitude, longitude)`` is a second explicit rule: POA coverage is
    measured only over bins with the sun above the horizon (pvlib solar position),
    so night-time logger gaps no longer reject a window. Daylight device checks
    are unchanged.
    """
    if not 0.0 <= max_peer_outlier_fraction <= 0.05:
        raise ValueError("peer outlier tolerance must be between 0 and 5% of daylight bins")
    validate_frame(frame, system_id)
    lower, upper = window_bounds(start, end)
    if len(set(device_ids)) != len(device_ids) or len(device_ids) < 3:
        raise ValueError("screening requires at least three distinct inverter IDs")
    power, poa = aligned_signals(frame, start, end, device_ids, poa_device_id)
    daylight = poa > DAYLIGHT_WM2
    coverage_bins = poa.index
    if daytime_site is not None:
        from pvlib.solarposition import get_solarposition

        latitude, longitude = daytime_site
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("invalid site coordinates")
        elevation = get_solarposition(poa.index, latitude, longitude)["apparent_elevation"]
        coverage_bins = poa.index[elevation.to_numpy() > 0]
    poa_coverage = float(poa.loc[coverage_bins].notna().mean()) if len(coverage_bins) else 0.0
    allowed_outliers = int(max_peer_outlier_fraction * int(daylight.sum()))
    scales = power[daylight].quantile(0.95)
    normalized = power.div(scales.where(scales > 0), axis="columns")
    devices = []
    for device in device_ids:
        peers = normalized.drop(columns=device)
        peer_median = peers.median(axis=1).where(peers.notna().sum(axis=1) >= 2)
        ratio = normalized[device].div(peer_median.where(peer_median > 0))
        daytime = power.loc[daylight, device]
        item = {
            "device_id": device,
            "coverage": float(power[device].notna().mean()),
            "daylight_missing_bins": int(daytime.isna().sum()),
            "daylight_near_zero_bins": int((daytime <= 0.1).sum()),
            "negative_power_bins": int((power[device] < 0).sum()),
            "daylight_uncomparable_bins": int(ratio[daylight].isna().sum()),
            "daylight_peer_outlier_bins": int(((ratio < 0.85) | (ratio > 1.20))[daylight].sum()),
            "empirical_p95_kw": float(scales[device]) if pd.notna(scales[device]) else None,
        }
        item["eligible"] = bool(
            poa_coverage >= 0.99
            and int(daylight.sum()) >= 16
            and item["coverage"] >= 0.99
            and all(
                item[key] == 0
                for key in (
                    "daylight_missing_bins",
                    "daylight_near_zero_bins",
                    "negative_power_bins",
                    "daylight_uncomparable_bins",
                )
            )
            and item["daylight_peer_outlier_bins"] <= allowed_outliers
        )
        devices.append(item)
    return {
        "schema_version": 1,
        "system_id": system_id,
        "window": [lower.isoformat(), upper.isoformat()],
        "poa_device_id": poa_device_id,
        "poa_coverage": poa_coverage,
        "daylight_bins": int(daylight.sum()),
        "devices": devices,
        "eligible_devices": [d["device_id"] for d in devices if d["eligible"]],
        "status": "screened_not_manually_verified",
        "screening_version": "m1-v1"
        + ("-large-site" if max_peer_outlier_fraction else "")
        + ("-daytime-poa" if daytime_site is not None else ""),
        "thresholds": {
            "daylight_wm2": DAYLIGHT_WM2,
            "coverage": 0.99,
            "minimum_daylight_bins": 16,
            "near_zero_kw": 0.1,
            "peer_ratio": [0.85, 1.20],
            "minimum_valid_peers": 2,
        }
        # Only the explicit large-site rule adds fields, so strict reports stay byte-identical.
        | (
            {
                "max_peer_outlier_fraction": max_peer_outlier_fraction,
                "allowed_peer_outlier_bins": allowed_outliers,
            }
            if max_peer_outlier_fraction
            else {}
        )
        | (
            {"poa_coverage_scope": "sun_above_horizon", "site": list(daytime_site)}
            if daytime_site is not None
            else {}
        ),
        "limitations": "Heuristic screening may miss persistent faults; no normal ground truth.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_id", type=int)
    parser.add_argument("--clean-root", type=Path, default=Path("data/clean"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--poa-device", required=True)
    args = parser.parse_args(argv)
    frame = pd.read_parquet(args.clean_root / f"{args.system_id}.parquet")
    report = screen_baseline(
        frame, args.system_id, args.start, args.end, tuple(args.devices), args.poa_device
    )
    path = args.clean_root / f"{args.system_id}_baseline_screen.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{path}: eligible_devices={report['eligible_devices']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
