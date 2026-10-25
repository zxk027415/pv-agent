"""Describe normalized data quality and suggest complete 6–12 month work windows.

Coverage means at least one finite observation per UTC 15-minute bin. It is not
a claim that the source data is fault-free. Missing bins are never interpolated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pvdiag.data.normalize import KEY, channels_from_metadata, utc_bound


def build_datacard(
    frame: pd.DataFrame,
    metadata: dict,
    *,
    normalization: dict | None = None,
    min_coverage: float = 0.95,
) -> dict:
    if frame.empty:
        raise ValueError("cannot describe an empty dataset")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if frame.ts.dt.tz is None or str(frame.ts.dt.tz) != "UTC":
        raise ValueError("timestamps must use UTC")
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate normalized keys")
    system_id = int(metadata["System"]["system_id"])
    if set(frame.system_id) != {system_id}:
        raise ValueError("system_id does not match metadata")
    step = pd.Timedelta(minutes=15)
    start, end = frame.ts.min().floor("15min"), frame.ts.max().floor("15min") + step
    if normalization is not None:
        if normalization["system_id"] != system_id:
            raise ValueError("normalization audit system_id does not match metadata")
        lower, upper = normalization["window_utc"]
        if lower is not None:
            lower = utc_bound(lower)
            if frame.ts.min() < lower:
                raise ValueError("observations precede normalization window")
            start = lower.floor("15min")
        if upper is not None:
            upper = utc_bound(upper)
            if frame.ts.max() >= upper:
                raise ValueError("observations exceed normalization window")
            end = upper.ceil("15min")
    expected = int((end - start) / step)
    channels = channels_from_metadata(metadata)
    groups = {
        (device, metric): series.sort_values("ts")
        for (device, metric), series in frame.groupby(["device_id", "metric"])
    }
    monthly = {}
    series_cards = []
    # Include metadata-described but absent channels so partial downloads cannot
    # masquerade as complete inverter coverage.
    identities = sorted({(c.device_id, c.metric) for c in channels} | set(groups))
    empty = frame.iloc[:0]
    for device, metric in identities:
        series = groups.get((device, metric), empty)
        valid = series[series.value.notna() & ~series.value.isin([float("inf"), -float("inf")])]
        bins = pd.DatetimeIndex(valid.ts.dt.floor("15min").unique()).sort_values()
        observed = len(bins)
        differences = series.ts.drop_duplicates().diff().dropna()
        cadence = None if differences.empty else float(differences.mode().iloc[0].total_seconds())
        item = {
            "device_id": device,
            "metric": metric,
            "rows": len(series),
            "missing_value_rows": int(series.value.isna().sum()),
            "nonfinite_value_rows": int(series.value.isin([float("inf"), -float("inf")]).sum()),
            "negative_value_rows": int((series.value < 0).sum()),
            "zero_value_rows": int((series.value == 0).sum()),
            "modal_interval_seconds": cadence,
            "observed_bins": observed,
            "expected_bins": expected,
            "missing_bins": expected - observed,
            "coverage": observed / expected,
        }
        series_cards.append(item)
        counts = pd.Series(1, index=bins, dtype="int64").resample("MS").sum()
        monthly[(device, metric)] = counts

    power_devices = {
        c.device_id
        for c in channels
        if c.metric == "ac_power_kw" and c.device_id.startswith(("inv", "inverter_"))
    }
    poa_sensors = {c.device_id for c in channels if c.metric == "poa_wm2"}
    required = {(d, "ac_power_kw") for d in power_devices}
    month_cards = []
    months = pd.date_range(start.normalize().replace(day=1), end, freq="MS", inclusive="left")
    for month in months:
        next_month = month + pd.offsets.MonthBegin(1)
        left, right = max(month, start), min(next_month, end)
        n = int((right - left) / step)
        coverage = {key: int(counts.get(month, 0)) / n for key, counts in monthly.items()}
        inverter_min = min((coverage[key] for key in required), default=0.0)
        poa_best = max((coverage[(d, "poa_wm2")] for d in poa_sensors), default=0.0)
        complete = month >= start and next_month <= end
        month_cards.append(
            {
                "month": month.strftime("%Y-%m"),
                "start": month.isoformat(),
                "end": next_month.isoformat(),
                "complete_month": complete,
                "min_inverter_coverage": inverter_min,
                "best_poa_coverage": poa_best,
                "eligible": complete and min(inverter_min, poa_best) >= min_coverage,
            }
        )
    windows = []
    for i, month in enumerate(month_cards):
        if not month["eligible"]:
            continue
        run = []
        for candidate in month_cards[i : i + 12]:
            if not candidate["eligible"]:
                break
            run.append(candidate)
        if len(run) >= 6:
            windows.append(
                {
                    "start": run[0]["start"],
                    "end": run[-1]["end"],
                    "months": len(run),
                    "min_coverage": min(
                        min(m["min_inverter_coverage"], m["best_poa_coverage"]) for m in run
                    ),
                }
            )
    windows.sort(key=lambda w: (-w["months"], -w["min_coverage"], w["start"]))
    warnings = []
    missing_inverters = sorted(d for d in power_devices if (d, "ac_power_kw") not in groups)
    if missing_inverters:
        warnings.append(f"missing inverter power channels: {', '.join(missing_inverters)}")
    if not windows:
        warnings.append("no complete 6–12 month window meets the coverage threshold")
    warnings.append(
        "coverage does not certify healthy data; inspect natural faults before injection"
    )
    return {
        "schema_version": 1,
        "system_id": system_id,
        "source_timezone": metadata["System"]["timezone_code"],
        "timezone": "UTC",
        "coverage_bin": "15min",
        "coverage_definition": "at least one finite value per bin",
        "min_coverage_threshold": min_coverage,
        "window": {"start": start.isoformat(), "end": end.isoformat(), "end_exclusive": True},
        "rows": len(frame),
        "series": series_cards,
        "monthly": month_cards,
        "recommended_window": windows[0] if windows else None,
        "site": metadata.get("Site", {}),
        "mounts": metadata.get("Mount", {}),
        "inverters": [
            {
                "name": inv["name"],
                "inverter_id": inv["inverter_id"],
                "model": inv.get("model"),
                "rated_ac_kw": None,
                "capacity_note": "not explicitly supplied in source metadata",
            }
            for inv in metadata.get("Inverters", {}).values()
        ],
        "normalization": normalization,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_id", type=int)
    parser.add_argument("--clean-root", type=Path, default=Path("data/clean"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--min-coverage", type=float, default=0.95)
    args = parser.parse_args(argv)
    metadata = json.loads(
        (
            args.raw_root
            / str(args.system_id)
            / "metadata"
            / f"{args.system_id}_system_metadata.json"
        ).read_text()
    )
    audit_path = args.clean_root / f"{args.system_id}_normalization.json"
    card = build_datacard(
        pd.read_parquet(args.clean_root / f"{args.system_id}.parquet"),
        metadata,
        normalization=json.loads(audit_path.read_text()) if audit_path.exists() else None,
        min_coverage=args.min_coverage,
    )
    path = args.clean_root / f"{args.system_id}_card.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"{path}: {card['rows']:,} rows; recommended_window={card['recommended_window']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
