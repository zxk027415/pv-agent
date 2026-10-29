"""Failure-driven development scenes (§3.4): weak fault types get more, harder variants.

Generated scenes reuse the parent manifest's development dates, devices and
baseline screening, are always ``dev`` and never become normal-scene truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pvdiag.eval.outcome import case_passed
from pvdiag.observability.store import digest
from pvdiag.sim.baseline import aligned_signals
from pvdiag.sim.benchmark import FAULT_TYPES, make_fault, screened_pool
from pvdiag.sim.scenarios import Scenario, materialize_scenario, validate_split_isolation

GENERATOR_VERSION = "gen-v1"
CLOUDY_DAY_RATIO = 0.6
HARD_CLOUDY_FRACTION = 0.3


def manifest_sha256(manifest):
    # Same binding as run_bench's summary["manifest_sha256"].
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def failure_profile(manifest, summary):
    scenes = {s["scenario_id"]: s for s in manifest["scenarios"]}
    if summary.get("manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("evaluation summary was not produced from this manifest")
    cases = summary.get("cases", [])
    if not cases or not {c["scenario_id"] for c in cases} <= set(scenes):
        raise ValueError("evaluation cases must belong to the parent manifest")
    profile = {kind: {"evaluated": 0, "failed": 0} for kind in FAULT_TYPES}
    normal = {"evaluated": 0, "failed": 0}
    for case in cases:
        kinds = {f["type"] for f in scenes[case["scenario_id"]]["faults"]} or {"N"}
        for kind in kinds:
            target = normal if kind == "N" else profile[kind]
            target["evaluated"] += 1
            target["failed"] += not case_passed(case)
    return profile, normal


def allocate(profile, budget):
    """Largest-remainder split of the budget in proportion to failures per fault type."""
    failed = {kind: v["failed"] for kind, v in profile.items() if v["failed"]}
    total = sum(failed.values())
    if not total:
        raise ValueError("no development fault failures to target")
    shares = {kind: budget * n / total for kind, n in failed.items()}
    counts = {kind: int(share) for kind, share in shares.items()}
    order = sorted(failed, key=lambda k: (-(shares[k] - counts[k]), FAULT_TYPES.index(k)))
    for kind in order[: budget - sum(counts.values())]:
        counts[kind] += 1
    return {kind: counts[kind] for kind in FAULT_TYPES if counts.get(kind)}


def cloudy_fraction(frame, left, right, devices, poa_device_id):
    _, poa = aligned_signals(frame, left.isoformat(), right.isoformat(), devices, poa_device_id)
    daily = poa.resample("1D").sum(min_count=1).dropna()
    if daily.empty or daily.max() <= 0:
        return 1.0
    return float((daily < CLOUDY_DAY_RATIO * daily.max()).mean())


def interleaved(pool, cloudiness):
    """Alternate the clearest and the cloudiest remaining windows (ties keep pool order)."""
    ranked = sorted(range(len(pool)), key=lambda i: (cloudiness[i], i))
    order, low, high = [], 0, len(ranked) - 1
    while low <= high:
        order.append(ranked[low])
        if low != high:
            order.append(ranked[high])
        low, high = low + 1, high - 1
    return [pool[i] for i in order]


def generate_from_failures(
    frame, manifest, summary, budget, *, window_days=None, stride_days=7, timezone=None, label=None
):
    scenes = manifest.get("scenarios", [])
    if not scenes or any(s["split"] != "dev" for s in scenes):
        raise ValueError("scene generation only reads development manifests")
    if not 1 <= budget <= 200:
        raise ValueError("generation budget must be 1–200 scenes")
    systems = {s["system_id"] for s in scenes}
    sensors = {s["poa_device_id"] for s in scenes}
    if len(systems) != 1 or len(sensors) != 1:
        raise ValueError("parent manifest must use one system and one POA sensor")
    system_id, poa_device_id = systems.pop(), sensors.pop()
    profile, normal = failure_profile(manifest, summary)
    counts = allocate(profile, budget)
    window_days = window_days or manifest.get("window_days")
    timezone = manifest.get("site_timezone") or timezone
    if not window_days or not timezone:
        raise ValueError("set the window length and site timezone the parent manifest omits")
    config = {
        "generator_version": GENERATOR_VERSION,
        "parent_manifest_sha256": manifest_sha256(manifest),
        "summary_cases_sha256": digest(summary["cases"]),
        "budget": budget,
        "window_days": window_days,
        "stride_days": stride_days,
        "site_timezone": timezone,
    } | ({"label": label} if label is not None else {})
    tag = digest(config)[:8]
    start, end = manifest["selection_window"]
    pool = screened_pool(
        frame,
        system_id,
        start,
        end,
        tuple(manifest["development_devices"]),
        poa_device_id,
        window_days,
        stride_days,
    )
    if not pool:
        raise ValueError("no screened development cohorts; baseline criteria were not relaxed")
    cloudiness = [cloudy_fraction(frame, *entry, poa_device_id) for entry in pool]
    by_entry = {entry: value for entry, value in zip(pool, cloudiness, strict=True)}
    existing = [Scenario.from_dict(s) for s in scenes]
    seen = {
        (s.query_window, s.device_ids, tuple(f.to_dict()["type"] for f in s.faults))
        for s in existing
    }
    generated, details, shortfall = [], {}, {}
    for kind, count in counts.items():
        made = 0
        # Drop windows already used for this fault type first, so weather alternation
        # applies to the windows actually available.
        fresh = [
            entry
            for entry in pool
            if ((entry[0].isoformat(), entry[1].isoformat()), tuple(entry[2]), (kind,)) not in seen
        ]
        for left, right, selected in interleaved(fresh, [by_entry[e] for e in fresh]):
            window = (left.isoformat(), right.isoformat())
            number = made + 1
            cloudy = by_entry[(left, right, selected)]
            try:
                fault = make_fault(
                    kind,
                    left,
                    right,
                    selected,
                    poa_device_id,
                    timezone,
                    number,
                    count,
                    frame,
                )
                hard = cloudy >= HARD_CLOUDY_FRACTION or window_days < 21 and kind in {"F1", "F5"}
                scenario = Scenario(
                    f"G{system_id}-{tag}-{kind}-{number:04d}",
                    system_id,
                    window,
                    selected,
                    poa_device_id,
                    (fault,),
                    difficulty="hard" if hard else "medium",
                )
                materialize_scenario(frame, scenario)
            except ValueError:
                continue
            generated.append(scenario)
            details[scenario.scenario_id] = {"cloudy_day_fraction": round(cloudy, 4)}
            made += 1
            if made == count:
                break
        if made < count:
            shortfall[kind] = count - made
    if not generated:
        raise ValueError("no observable scenes could be generated for the failing fault types")
    validate_split_isolation(existing + generated)
    report = {
        "config": config,
        "tag": tag,
        "failure_profile": profile,
        "normal_profile": normal | {"generated": 0, "reason": "unreviewed baselines are not truth"},
        "allocation": counts,
        "shortfall": shortfall,
        "pool_windows": len(pool),
    }
    return generated, details, report


def write_pool(frame, scenarios, details, report, parent, source_sha, dest):
    """Write observations/truth/audit per scene and return the generated manifest."""
    output = {
        "schema_version": 2,
        "generator_version": GENERATOR_VERSION,
        "purpose": "development_pool_generated_from_failures_not_heldout",
        "source_sha256": source_sha,
        "selection_window": parent["selection_window"],
        "development_devices": parent["development_devices"],
        "window_days": report["config"]["window_days"],
        "site_timezone": report["config"]["site_timezone"],
        "generation": report,
        "scenarios": [],
    }
    for scenario in scenarios:
        observation, audit = materialize_scenario(frame, scenario)
        destination = Path(dest) / scenario.scenario_id
        destination.mkdir(parents=True, exist_ok=False)
        observation.to_parquet(destination / "timeseries.parquet", index=False)
        (destination / "truth.json").write_text(json.dumps(scenario.to_dict(), indent=2) + "\n")
        (destination / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
        output["scenarios"].append(
            scenario.to_dict() | audit | {"generation": details[scenario.scenario_id]}
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="parent development manifest")
    parser.add_argument("--summary", type=Path, required=True, help="run_bench summary.json")
    parser.add_argument("--source", type=Path, default=Path("data/clean/2107.parquet"))
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--window-days", type=int)
    parser.add_argument("--stride-days", type=int, default=7)
    parser.add_argument("--site-timezone", help="only when the parent manifest omits it")
    parser.add_argument("--output", type=Path, required=True, help="new generated manifest")
    parser.add_argument("--dest", type=Path, default=Path("data/generated"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output manifest already exists; generated pools are never overwritten")
    manifest = json.loads(args.manifest.read_text())
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_sha != manifest.get("source_sha256"):
        parser.error("source parquet differs from the parent manifest snapshot")
    frame = pd.read_parquet(args.source)
    try:
        scenarios, details, report = generate_from_failures(
            frame,
            manifest,
            json.loads(args.summary.read_text()),
            args.budget,
            window_days=args.window_days,
            stride_days=args.stride_days,
            timezone=args.site_timezone,
        )
    except ValueError as error:
        parser.exit(2, f"{error}\n")
    output = write_pool(frame, scenarios, details, report, manifest, source_sha, args.dest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        f"{args.output}: {len(scenarios)} generated development scenes "
        f"{report['allocation']}; shortfall {report['shortfall'] or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
