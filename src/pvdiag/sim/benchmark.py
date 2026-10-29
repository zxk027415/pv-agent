"""Build reproducible six-class development candidates, with unreviewed normal candidates.

This does not generate or evaluate held-out scenes. Unmodified baselines remain
explicitly ineligible for normal-scene scoring until independent review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from pvdiag.sim.baseline import aligned_signals, screen_baseline, validate_frame, window_bounds
from pvdiag.sim.faults import FaultSpec
from pvdiag.sim.scenarios import Scenario, materialize_scenario, validate_split_isolation

FAULT_TYPES = ("F1", "F2", "F3", "F4", "F5", "F6")


def screened_pool(
    frame, system_id, start, end, device_ids, poa_device_id, window_days, stride_days=7
):
    lower, upper = window_bounds(start, end)
    if not 7 <= window_days <= 90 or stride_days < 1:
        raise ValueError("candidate windows must be 7–90 days with positive stride")
    if len(set(device_ids)) != len(device_ids) or len(device_ids) < 3:
        raise ValueError("at least three distinct development devices required")
    selected_frame = frame[
        (frame.system_id == system_id) & (frame.ts >= lower) & (frame.ts < upper)
    ]
    validate_frame(selected_frame, system_id)
    first, extra = [], []
    for left in pd.date_range(lower, upper, freq=f"{stride_days}D", inclusive="left"):
        right = left + pd.Timedelta(days=window_days)
        if right > upper:
            break
        window = selected_frame[(selected_frame.ts >= left) & (selected_frame.ts < right)]
        if window.empty:
            continue
        report = screen_baseline(
            window,
            system_id,
            left.isoformat(),
            right.isoformat(),
            tuple(sorted(device_ids)),
            poa_device_id,
        )
        local_cohorts = []
        # Limit combinatorial expansion while retaining distinct normal candidates.
        for selected in list(combinations(report["eligible_devices"], 3))[:20]:
            final = screen_baseline(
                window, system_id, left.isoformat(), right.isoformat(), selected, poa_device_id
            )
            if set(final["eligible_devices"]) == set(selected):
                local_cohorts.append((left, right, selected))
        if local_cohorts:
            first.append(local_cohorts[0])
            extra.extend(local_cohorts[1:])
    return first + extra


def make_fault(kind, left, right, devices, poa_device_id, timezone, number, per_type, frame):
    onset = left + pd.Timedelta(days=3)
    device = devices[(number - 1) % len(devices)]
    fraction = (number - 1) / max(1, per_type - 1)
    common = (kind, device, onset.isoformat(), right.isoformat())
    if kind == "F1":
        cleaning = onset + (right - onset) * 0.75
        return FaultSpec(
            *common,
            soiling_rate_per_day=0.001 + 0.004 * fraction,
            cleaning_times=(cleaning.floor("15min").isoformat(),),
        )
    if kind == "F2":
        return FaultSpec(
            *common,
            severity=0.3 + 0.4 * fraction,
            daily_window=("15:00", "17:00"),
            timezone=timezone,
        )
    if kind == "F3":
        power, poa = aligned_signals(
            frame, left.isoformat(), right.isoformat(), devices, poa_device_id
        )
        eligible = power.index[
            (poa > 200)
            & power[device].gt(0.1)
            & (power.index >= onset)
            & (power.index < right - pd.Timedelta(hours=3))
        ]
        if eligible.empty:
            raise ValueError("no observable high-irradiance outage interval")
        event = eligible[len(eligible) // 2]
        return FaultSpec(
            kind,
            device,
            event.isoformat(),
            (event + pd.Timedelta(hours=3)).isoformat(),
            poa_device_id=poa_device_id,
        )
    if kind == "F4":
        return FaultSpec(*common, loss_fraction=0.125 if number % 2 else 0.25)
    if kind == "F5":
        return FaultSpec(
            kind,
            poa_device_id,
            onset.isoformat(),
            right.isoformat(),
            drift_per_day=0.002 + 0.004 * fraction,
        )
    if kind == "F6":
        return FaultSpec(
            kind,
            device,
            onset.isoformat(),
            (onset + pd.Timedelta(hours=6 + 6 * ((number - 1) % 4))).isoformat(),
        )
    raise ValueError("unsupported fault type")


def generate_candidates(
    frame,
    system_id,
    start,
    end,
    device_ids,
    poa_device_id,
    timezone,
    *,
    per_type=10,
    normal_count=20,
    window_days=35,
    stride_days=7,
    split="dev",
):
    if not 1 <= per_type <= 100 or not 0 <= normal_count <= 200:
        raise ValueError("invalid candidate counts")
    # Held-out ("test") scenes are not generated here; cross-site scenes never tune anything.
    prefixes = {"dev": "D", "cross_site": "X"}
    if split not in prefixes:
        raise ValueError("candidates are generated only for dev or cross_site")
    pool = screened_pool(
        frame, system_id, start, end, device_ids, poa_device_id, window_days, stride_days
    )
    if not pool:
        raise ValueError("no screened development cohorts; baseline criteria were not relaxed")
    scenarios = []
    for kind in FAULT_TYPES:
        generated = []
        for left, right, selected in pool:
            number = len(generated) + 1
            try:
                fault = make_fault(
                    kind, left, right, selected, poa_device_id, timezone, number, per_type, frame
                )
                # Short slow ramps are intentionally marked hard, not strengthened
                # beyond the design's physical rate limits to make them detectable.
                difficulty = "hard" if window_days < 21 and kind in {"F1", "F5"} else "medium"
                scenario = Scenario(
                    f"{prefixes[split]}{system_id}-{kind}-{number:04d}",
                    system_id,
                    (left.isoformat(), right.isoformat()),
                    selected,
                    poa_device_id,
                    (fault,),
                    split=split,
                    difficulty=difficulty,
                )
                materialize_scenario(frame, scenario)
            except ValueError:
                continue
            generated.append(scenario)
            if len(generated) == per_type:
                break
        if len(generated) != per_type:
            raise ValueError(f"only {len(generated)}/{per_type} observable {kind} scenarios")
        scenarios.extend(generated)
    if len(pool) < normal_count:
        raise ValueError(f"only {len(pool)}/{normal_count} distinct unmodified baseline candidates")
    for number, (left, right, selected) in enumerate(pool[:normal_count], start=1):
        scenarios.append(
            Scenario(
                f"{prefixes[split]}{system_id}-N-{number:04d}",
                system_id,
                (left.isoformat(), right.isoformat()),
                selected,
                poa_device_id,
                (),
                split=split,
            )
        )
    validate_split_isolation(scenarios)
    return scenarios


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_id", type=int)
    parser.add_argument("--source", type=Path, default=Path("data/clean/2107.parquet"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--poa-device", required=True)
    parser.add_argument("--timezone", required=True)
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--normal-count", type=int, default=20)
    parser.add_argument("--window-days", type=int, default=35)
    parser.add_argument("--stride-days", type=int, default=7)
    parser.add_argument("--split", choices=["dev", "cross_site"], default="dev")
    parser.add_argument("--manifest", type=Path, default=Path("bench/m2_dev_candidates.json"))
    parser.add_argument("--dest", type=Path, default=Path("data/m2_candidates"))
    args = parser.parse_args(argv)
    frame = pd.read_parquet(args.source)
    # Only development dates/devices contribute to screening and candidate choice.
    scenarios = generate_candidates(
        frame,
        args.system_id,
        args.start,
        args.end,
        tuple(args.devices),
        args.poa_device,
        args.timezone,
        per_type=args.per_type,
        normal_count=args.normal_count,
        window_days=args.window_days,
        stride_days=args.stride_days,
        split=args.split,
    )
    manifest = {
        "schema_version": 2,
        "generator_version": "m2-v1",
        "purpose": "development_candidates_not_final_benchmark"
        if args.split == "dev"
        else "cross_site_candidates_never_used_for_tuning",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "selection_window": list(map(str, window_bounds(args.start, args.end))),
        "development_devices": sorted(args.devices),
        "window_days": args.window_days,
        "site_timezone": args.timezone,
        "scenarios": [],
    }
    for scenario in scenarios:
        observation, audit = materialize_scenario(frame, scenario)
        destination = args.dest / scenario.scenario_id
        destination.mkdir(parents=True, exist_ok=True)
        observation.to_parquet(destination / "timeseries.parquet", index=False)
        (destination / "truth.json").write_text(json.dumps(scenario.to_dict(), indent=2) + "\n")
        (destination / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
        manifest["scenarios"].append(scenario.to_dict() | audit)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        f"{args.manifest}: {len(scenarios)} {args.split} candidates; normal candidates are not scoring eligible"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
