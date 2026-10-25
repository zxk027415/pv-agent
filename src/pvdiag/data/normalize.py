"""Normalize metadata-described PVDAQ CSVs to a UTC long table.

Only original ``*data.csv`` files are selected by default. Yearly extensions and
alternate revisions must be selected explicitly to avoid silently mixing sources.
No interpolation, clipping or cumulative-energy-to-power guessing is performed.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

SCHEMA = ["ts", "system_id", "device_id", "metric", "value"]
KEY = SCHEMA[:-1]
METRICS = {
    "ac power": "ac_power_kw",
    "dc power": "dc_power_kw",
    "irradiance poa": "poa_wm2",
    "temperature ambient": "ambient_c",
    "temperature module": "module_c",
}


@dataclass(frozen=True)
class Channel:
    column: str
    device_id: str
    metric: str
    source_unit: str
    scale: float
    offset: float = 0.0


def channels_from_metadata(metadata: dict) -> list[Channel]:
    """Map measured channels, using explicit units and stable source device IDs.

    The 2107 metadata omits inverter ``source_id``. Its explicit ``inv_01``
    channel names are used as a fallback; unsupported device mappings fail.
    Metadata ``units`` describes exported CSV values, so raw calc_scale/offset
    must not be applied a second time.
    """
    channels = []
    for column, info in metadata["Metrics"].items():
        metric = METRICS.get(info["common_name"].strip().lower())
        if metric is None:
            continue
        unit = info["units"].strip().lower().replace("²", "^2")
        if metric.endswith("power_kw"):
            scales = {"kw": 1.0, "w": 0.001, "mw": 1000.0}
        elif metric == "poa_wm2":
            scales = {"w/m^2": 1.0, "w/m2": 1.0, "kw/m^2": 1000.0}
        else:
            scales = {"c": 1.0, "°c": 1.0, "f": 5 / 9, "°f": 5 / 9}
        if unit not in scales:
            raise ValueError(f"unsupported unit {unit!r} for {column}")
        offset = -32 * 5 / 9 if unit in {"f", "°f"} else 0.0
        inverter = re.match(r"inv_(\d+)_", column)
        source_type = info["source_type"].upper()
        if source_type == "INVERTER":
            if inverter:
                device_id = f"inv{int(inverter[1]):02d}"
            elif info.get("source_id") is not None:
                device_id = f"inverter_{info['source_id']}"
            else:
                raise ValueError(f"cannot identify inverter for {column}")
        elif source_type == "METER":
            device_id = f"meter_{info['metric_id']}"
        else:
            # Keep separate sensors distinct, rather than averaging them away.
            device_id = f"sensor_{info['metric_id']}"
        channels.append(Channel(column, device_id, metric, unit, scales[unit], offset))
    return channels


def utc_bound(value: str | pd.Timestamp) -> pd.Timestamp:
    """CLI/window bounds without an offset explicitly mean UTC."""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("time bound cannot be NaT")
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def normalize_files(
    files: list[Path],
    metadata: dict,
    *,
    source_timezone: str | None = None,
    start: str | None = None,
    end: str | None = None,
    chunksize: int = 50_000,
) -> tuple[pd.DataFrame, dict]:
    """Read selected columns in chunks, retaining NaNs and rejecting conflicts.

    ``utc_measured_on`` is UTC. Naive ``measured_on`` uses the metadata's zone
    (or explicit override). DST ambiguous/nonexistent local times are excluded
    and counted, never guessed or shifted. Windows use [start, end) in UTC.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    lower, upper = utc_bound(start) if start else None, utc_bound(end) if end else None
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("start must precede end")
    channels = channels_from_metadata(metadata)
    timezone = source_timezone or metadata["System"]["timezone_code"]
    audit = {
        "system_id": int(metadata["System"]["system_id"]),
        "source_timezone": timezone,
        "window_utc": [
            str(lower) if lower is not None else None,
            str(upper) if upper is not None else None,
        ],
        "channels": [asdict(c) for c in channels],
        "sources": [],
        "duplicate_rows_collapsed": 0,
    }
    parts = []
    for path in sorted(files):
        header = pd.read_csv(path, nrows=0).columns
        time_columns = [c for c in ("measured_on", "utc_measured_on") if c in header]
        if len(time_columns) != 1:
            raise ValueError(f"{path}: expected exactly one timestamp column")
        time_column = time_columns[0]
        selected = []
        aliases = {}
        for channel in channels:
            column = channel.column
            if column not in header:
                # Original 2107 CSV spells inverter 15's channel as "iinv";
                # the exported metric ID remains authoritative and unchanged.
                metric_id = metadata["Metrics"][column]["metric_id"]
                matches = [name for name in header if name.endswith(f"_{metric_id}")]
                if len(matches) > 1:
                    raise ValueError(f"{path}: ambiguous metric ID {metric_id}")
                if not matches:
                    continue
                aliases[matches[0]] = column
                column = matches[0]
            selected.append(replace(channel, column=column))
        source = {
            "path": str(path),
            "timestamp_column": time_column,
            "timezone": "UTC" if time_column == "utc_measured_on" else timezone,
            "input_rows": 0,
            "dst_rows_excluded": 0,
            "selected_rows": 0,
            "columns": [c.column for c in selected],
            "column_aliases": aliases,
        }
        audit["sources"].append(source)
        if not selected:
            continue
        for chunk in pd.read_csv(
            path, usecols=[time_column, *(c.column for c in selected)], chunksize=chunksize
        ):
            source["input_rows"] += len(chunk)
            ts = pd.to_datetime(chunk[time_column], errors="raise")
            if ts.isna().any():
                raise ValueError(f"{path}: missing timestamp")
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(source["timezone"], ambiguous="NaT", nonexistent="NaT")
            ts = ts.dt.tz_convert("UTC")
            source["dst_rows_excluded"] += int(ts.isna().sum())
            keep = ts.notna()
            if lower is not None:
                keep &= ts >= lower
            if upper is not None:
                keep &= ts < upper
            source["selected_rows"] += int(keep.sum())
            if not keep.any():
                continue
            for channel in selected:
                values = pd.to_numeric(chunk.loc[keep, channel.column], errors="raise")
                values = values.astype(float) * channel.scale + channel.offset
                if values.isin([float("inf"), -float("inf")]).any():
                    raise ValueError(f"{path}: infinite value in {channel.column}")
                parts.append(
                    pd.DataFrame(
                        {
                            "ts": ts[keep],
                            "system_id": audit["system_id"],
                            "device_id": channel.device_id,
                            "metric": channel.metric,
                            "value": values,
                        }
                    )
                )
    if not parts:
        raise ValueError("no supported observations in the selected files/window")
    result = pd.concat(parts, ignore_index=True)
    duplicates = result.duplicated(KEY, keep=False)
    if duplicates.any():
        repeated = result.loc[duplicates]
        # Missing and present observations from overlapping exports also conflict.
        conflicts = repeated.groupby(KEY)["value"].nunique(dropna=False) > 1
        if conflicts.any():
            raise ValueError(f"conflicting observations for {conflicts[conflicts].index[0]}")
        before = len(result)
        result = result.drop_duplicates(KEY)
        audit["duplicate_rows_collapsed"] = before - len(result)
    return result[SCHEMA].sort_values(KEY).reset_index(drop=True), audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_id", type=int)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dest", type=Path, default=Path("data/clean"))
    parser.add_argument("--pattern", action="append", help="CSV glob; repeatable")
    parser.add_argument("--source-timezone", help="override naive measured_on timezone")
    parser.add_argument("--start", help="inclusive UTC bound")
    parser.add_argument("--end", help="exclusive UTC bound")
    args = parser.parse_args(argv)
    root = args.raw_root / str(args.system_id)
    metadata = json.loads(
        (root / "metadata" / f"{args.system_id}_system_metadata.json").read_text()
    )
    if int(metadata["System"]["system_id"]) != args.system_id:
        parser.error("metadata system_id does not match request")
    files = sorted(
        {p for pattern in (args.pattern or ["*data.csv"]) for p in (root / "data").glob(pattern)}
    )
    frame, audit = normalize_files(
        files, metadata, source_timezone=args.source_timezone, start=args.start, end=args.end
    )
    args.dest.mkdir(parents=True, exist_ok=True)
    parquet = args.dest / f"{args.system_id}.parquet"
    frame.to_parquet(parquet, index=False)
    (args.dest / f"{args.system_id}_normalization.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(f"{parquet}: {len(frame):,} rows, {frame.device_id.nunique()} devices/sensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
