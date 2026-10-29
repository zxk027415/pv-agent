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
from pvdiag.sim.scenarios import (
    Scenario,
    materialize_scenario,
    screening_options,
    validate_split_isolation,
)

FAULT_TYPES = ("F1", "F2", "F3", "F4", "F5", "F6")


def screened_pool(
    frame,
    system_id,
    start,
    end,
    device_ids,
    poa_device_id,
    window_days,
    stride_days=7,
    max_peer_outlier_fraction=0.0,
    daytime_site=None,
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
            max_peer_outlier_fraction,
            daytime_site,
        )
        local_cohorts = []
        # Limit combinatorial expansion while retaining distinct normal candidates.
        for selected in list(combinations(report["eligible_devices"], 3))[:20]:
            final = screen_baseline(
                window,
                system_id,
                left.isoformat(),
                right.isoformat(),
                selected,
                poa_device_id,
                max_peer_outlier_fraction,
                daytime_site,
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
    max_peer_outlier_fraction=0.0,
    fault_types=FAULT_TYPES,
    daytime_site=None,
    compounds=(),
    per_compound=0,
):
    # Single-fault scenes may be omitted only in an explicit compound-only pool.
    if not (0 if compounds else 1) <= per_type <= 100 or not 0 <= normal_count <= 200:
        raise ValueError("invalid candidate counts")
    # Held-out and cross-site scenes are generated but never tune anything; the
    # benchmark runner still refuses non-dev manifests.
    prefixes = {"dev": "D", "test": "T", "cross_site": "X"}
    if split not in prefixes:
        raise ValueError("candidates are generated only for dev, test or cross_site")
    if (
        not fault_types
        or len(set(fault_types)) != len(fault_types)
        or not set(fault_types) <= set(FAULT_TYPES)
    ):
        raise ValueError("fault types must be distinct F1–F6 codes")
    if bool(compounds) != bool(per_compound) or not 0 <= per_compound <= 50:
        raise ValueError("compound scenes need both pairs and a per-pair count of 1–50")
    for pair in compounds:
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(FAULT_TYPES):
            raise ValueError("compound pairs are two distinct F1–F6 codes")
    pool = screened_pool(
        frame,
        system_id,
        start,
        end,
        device_ids,
        poa_device_id,
        window_days,
        stride_days,
        max_peer_outlier_fraction,
        daytime_site,
    )
    if not pool:
        raise ValueError("no screened development cohorts; baseline criteria were not relaxed")
    screening = {
        key: value
        for key, value in (
            ("max_peer_outlier_fraction", max_peer_outlier_fraction),
            ("daytime_site", daytime_site),
        )
        if value
    }
    scenarios = []
    for kind in fault_types if per_type else ():
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
                materialize_scenario(frame, scenario, **screening)
            except ValueError:
                continue
            generated.append(scenario)
            if len(generated) == per_type:
                break
        if len(generated) != per_type:
            raise ValueError(f"only {len(generated)}/{per_type} observable {kind} scenarios")
        scenarios.extend(generated)
    for first, second in compounds:
        generated = []
        for left, right, selected in pool:
            number = len(generated) + 1
            # The second fault targets another device of the cohort (F5 targets the POA).
            rotated = (*selected[1:], selected[0])
            try:
                faults = (
                    make_fault(
                        first,
                        left,
                        right,
                        selected,
                        poa_device_id,
                        timezone,
                        number,
                        per_compound,
                        frame,
                    ),
                    make_fault(
                        second,
                        left,
                        right,
                        rotated,
                        poa_device_id,
                        timezone,
                        number,
                        per_compound,
                        frame,
                    ),
                )
                scenario = Scenario(
                    f"{prefixes[split]}{system_id}-{first}{second}-{number:04d}",
                    system_id,
                    (left.isoformat(), right.isoformat()),
                    selected,
                    poa_device_id,
                    faults,
                    split=split,
                    difficulty="hard",
                )
                # Rejects masked or unobservable components as well as screening failures.
                materialize_scenario(frame, scenario, **screening)
            except ValueError:
                continue
            generated.append(scenario)
            if len(generated) == per_compound:
                break
        if len(generated) != per_compound:
            raise ValueError(
                f"only {len(generated)}/{per_compound} observable {first}+{second} scenarios"
            )
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
    parser.add_argument("--split", choices=["dev", "test", "cross_site"], default="dev")
    parser.add_argument(
        "--daytime-poa-coverage",
        nargs=2,
        type=float,
        metavar=("LATITUDE", "LONGITUDE"),
        help="explicit rule: measure POA coverage only while the sun is above the horizon",
    )
    parser.add_argument(
        "--compound",
        action="append",
        default=[],
        metavar="FA,FB",
        help="explicit hard two-fault pair, repeatable (e.g. F3,F5)",
    )
    parser.add_argument("--per-compound", type=int, default=0)
    parser.add_argument("--fault-types", nargs="+", choices=FAULT_TYPES, default=list(FAULT_TYPES))
    parser.add_argument(
        "--max-peer-outlier-fraction",
        type=float,
        default=0.0,
        help="explicit large-site screening tolerance (default 0 = strict m1-v1)",
    )
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
        max_peer_outlier_fraction=args.max_peer_outlier_fraction,
        fault_types=tuple(args.fault_types),
        daytime_site=tuple(args.daytime_poa_coverage) if args.daytime_poa_coverage else None,
        compounds=tuple(tuple(pair.split(",")) for pair in args.compound),
        per_compound=args.per_compound,
    )
    manifest = {
        "schema_version": 2,
        "generator_version": "m2-v1",
        "purpose": {
            "dev": "development_candidates_not_final_benchmark",
            "test": "heldout_candidates_never_used_for_tuning",
            "cross_site": "cross_site_candidates_never_used_for_tuning",
        }[args.split],
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "selection_window": list(map(str, window_bounds(args.start, args.end))),
        "development_devices": sorted(args.devices),
        "window_days": args.window_days,
        "site_timezone": args.timezone,
        "scenarios": [],
    }
    if args.max_peer_outlier_fraction:
        manifest["max_peer_outlier_fraction"] = args.max_peer_outlier_fraction
    if args.daytime_poa_coverage:
        manifest["poa_coverage_scope"] = {
            "scope": "sun_above_horizon",
            "site": list(args.daytime_poa_coverage),
        }
    if args.compound:
        manifest["compounds"] = {"pairs": args.compound, "per_pair": args.per_compound}
    if tuple(args.fault_types) != FAULT_TYPES:
        manifest["fault_types"] = list(args.fault_types)
    for scenario in scenarios:
        observation, audit = materialize_scenario(frame, scenario, **screening_options(manifest))
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
