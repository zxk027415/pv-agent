"""Validated scenes and the legacy deterministic F3/F4 development smoke set.

No normal-scene labels or held-out evaluation results are produced. Scenario
truth is stored separately from the observation parquet consumed by a diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from pvdiag.sim.baseline import (
    DAYLIGHT_WM2,
    aligned_signals,
    screen_baseline,
    validate_frame,
    window_bounds,
)
from pvdiag.sim.faults import FaultSpec, inject_fault


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    system_id: int
    query_window: tuple[str, str]
    device_ids: tuple[str, ...]
    poa_device_id: str
    faults: tuple[FaultSpec, ...]
    split: str = "dev"
    difficulty: str = "easy"

    def __post_init__(self):
        object.__setattr__(self, "device_ids", tuple(self.device_ids))
        object.__setattr__(self, "faults", tuple(self.faults))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", self.scenario_id):
            raise ValueError("invalid scenario_id")
        if self.split not in {"dev", "test", "cross_site"}:
            raise ValueError("unsupported split")
        lower, upper = window_bounds(*self.query_window)
        if upper - lower > pd.Timedelta(days=90):
            raise ValueError("query window exceeds 90 days")
        object.__setattr__(self, "query_window", (lower.isoformat(), upper.isoformat()))
        if len(self.device_ids) < 3 or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("scenario requires at least three distinct inverters")
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("unsupported difficulty")
        if len(self.faults) > 2 or (len(self.faults) == 2 and self.difficulty != "hard"):
            raise ValueError("two-fault scenes must use hard difficulty; at most two faults")
        signatures = [json.dumps(f.to_dict(), sort_keys=True) for f in self.faults]
        if len(signatures) != len(set(signatures)):
            raise ValueError("duplicate injected faults")
        for fault in self.faults:
            start, end = window_bounds(fault.start, fault.end)
            allowed = {self.poa_device_id} if fault.type == "F5" else set(self.device_ids)
            if fault.device_id not in allowed or start < lower or end > upper:
                raise ValueError("fault device/window must belong to query scope")
            if fault.type == "F3" and fault.poa_device_id != self.poa_device_id:
                raise ValueError("fault POA sensor must match scenario sensor")
        for left, right in combinations(self.faults, 2):
            if left.type == right.type and left.device_id == right.device_id:
                a, b = window_bounds(left.start, left.end), window_bounds(right.start, right.end)
                if max(a[0], b[0]) < min(a[1], b[1]):
                    raise ValueError(
                        "overlapping same-type faults on one device are not independent"
                    )

    def to_dict(self) -> dict:
        result = {
            "scenario_id": self.scenario_id,
            "system_id": self.system_id,
            "query_window": list(self.query_window),
            "device_ids": list(self.device_ids),
            "poa_device_id": self.poa_device_id,
            "faults": [f.to_dict() for f in self.faults],
            "difficulty": self.difficulty,
            "split": self.split,
            "truth_scope": "injected_faults_only"
            if self.faults
            else "unmodified_baseline_candidate",
        }
        if not self.faults:
            result["evaluation_eligible"] = False
        return result

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        return cls(
            data["scenario_id"],
            data["system_id"],
            tuple(data["query_window"]),
            tuple(data["device_ids"]),
            data["poa_device_id"],
            tuple(FaultSpec.from_dict(f) for f in data["faults"]),
            data["split"],
            data.get("difficulty", "easy"),
        )


def validate_split_isolation(scenarios: list[Scenario]) -> None:
    """Dev/test query dates AND all visible inverter IDs must be disjoint.

    Peer devices count as visible, not just fault targets. Cross-site systems
    must not appear in either development or same-site held-out sets.
    """
    if len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario IDs")
    for left, right in combinations(scenarios, 2):
        if left.system_id != right.system_id or left.split == right.split:
            continue
        if "cross_site" in {left.split, right.split}:
            raise ValueError("cross-site system is also used in dev/test")
        if set(left.device_ids) & set(right.device_ids):
            raise ValueError("dev/test visible inverter IDs overlap")
        a, b = window_bounds(*left.query_window), window_bounds(*right.query_window)
        if max(a[0], b[0]) < min(a[1], b[1]):
            raise ValueError("dev/test query windows overlap")


def scenario_window(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """Expose selected peers, selected POA and temperatures; exclude site meters.

    An unchanged site aggregate would contradict injected inverter power. The
    smoke set therefore exposes a device cohort without a site meter or DC model.
    """
    lower, upper = window_bounds(*scenario.query_window)
    visible = (frame.metric == "ac_power_kw") & frame.device_id.isin(scenario.device_ids)
    visible |= (frame.metric == "poa_wm2") & (frame.device_id == scenario.poa_device_id)
    visible |= frame.metric == "ambient_c"
    visible |= (frame.metric == "module_c") & frame.device_id.isin(scenario.device_ids)
    return frame[
        (frame.system_id == scenario.system_id) & visible & (frame.ts >= lower) & (frame.ts < upper)
    ].copy()


def materialize_scenario(frame: pd.DataFrame, scenario: Scenario) -> tuple[pd.DataFrame, dict]:
    window = scenario_window(frame, scenario)
    report = screen_baseline(
        window,
        scenario.system_id,
        *scenario.query_window,
        scenario.device_ids,
        scenario.poa_device_id,
    )
    if set(report["eligible_devices"]) != set(scenario.device_ids):
        raise ValueError(f"{scenario.scenario_id}: baseline failed screening")
    if not scenario.faults:
        return window.copy(deep=True), {
            "baseline_screen": report,
            "injection": None,
            "evaluation_eligible": False,
        }

    def apply(faults):
        current, audits = window, []
        # Outages use physical (pre-drift) irradiance. Communication loss is an
        # observation failure and is applied after all physical/sensor changes.
        ordered = sorted(faults, key=lambda fault: {"F5": 1, "F6": 2}.get(fault.type, 0))
        for fault in ordered:
            current, audit = inject_fault(current, scenario.system_id, fault)
            audits.append(audit)
        return current, audits

    injected, audits = apply(scenario.faults)
    if len(scenario.faults) == 1:
        return injected, {"baseline_screen": report, "injection": audits[0]}
    for index, fault in enumerate(scenario.faults):
        without, _ = apply(scenario.faults[:index] + scenario.faults[index + 1 :])
        effect = injected.value.ne(without.value) & ~(injected.value.isna() & without.value.isna())
        if not effect.any():
            raise ValueError(f"unobservable compound fault: {fault.type}")
    return injected, {
        "baseline_screen": report,
        "injections": audits,
        "composition_order": [a["fault"]["type"] for a in audits],
    }


def generate_smoke_scenarios(
    frame: pd.DataFrame,
    system_id: int,
    start: str,
    end: str,
    device_ids: tuple[str, ...],
    poa_device_id: str,
    count: int = 10,
) -> list[Scenario]:
    """Scan disjoint 7-day query windows, making one F3 and one F4 per window.

    Caller must supply only development dates/devices. Choose the first three
    eligible peers deterministically and screen that final cohort again.
    """
    if count < 2 or count % 2:
        raise ValueError("count must be a positive even number >=2")
    lower, upper = window_bounds(start, end)
    if len(set(device_ids)) != len(device_ids) or len(device_ids) < 3:
        raise ValueError("requires at least three distinct development inverters")
    frame = frame[(frame.system_id == system_id) & (frame.ts >= lower) & (frame.ts < upper)]
    validate_frame(frame, system_id)
    scenarios = []
    for left in pd.date_range(lower, upper, freq="7D", inclusive="left"):
        right = left + pd.Timedelta(days=7)
        if right > upper:
            break
        window = frame[(frame.ts >= left) & (frame.ts < right)]
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
        selected = tuple(report["eligible_devices"][:3])
        if len(selected) < 3:
            continue
        report = screen_baseline(
            window, system_id, left.isoformat(), right.isoformat(), selected, poa_device_id
        )
        if set(selected) != set(report["eligible_devices"]):
            continue
        power, poa = aligned_signals(
            window, left.isoformat(), right.isoformat(), selected, poa_device_id
        )
        event_bins = power.index[
            (poa > DAYLIGHT_WM2)
            & (power[selected[0]] > 0.1)
            & (power.index >= left + pd.Timedelta(days=2))
            & (power.index < right - pd.Timedelta(days=1))
        ]
        if event_bins.empty:
            continue
        onset = event_bins[len(event_bins) // 2]
        number = len(scenarios) // 2 + 1
        faults = (
            FaultSpec(
                "F3",
                selected[0],
                onset.isoformat(),
                (onset + pd.Timedelta(hours=3)).isoformat(),
                poa_device_id=poa_device_id,
            ),
            FaultSpec(
                "F4",
                selected[1],
                (left + pd.Timedelta(days=3)).isoformat(),
                right.isoformat(),
                loss_fraction=0.125 if number % 2 else 0.25,
            ),
        )
        candidates = []
        for fault in faults:
            candidates.append(
                Scenario(
                    f"S{system_id}-{fault.type}-{number:04d}",
                    system_id,
                    (left.isoformat(), right.isoformat()),
                    selected,
                    poa_device_id,
                    (fault,),
                )
            )
        try:
            for candidate in candidates:
                materialize_scenario(window, candidate)
        except ValueError:
            # A screened week may still have no observable loss after the F4
            # onset, or a POA gap within the complete F3 interval. Skip the pair.
            continue
        scenarios.extend(candidates)
        if len(scenarios) == count:
            validate_split_isolation(scenarios)
            return scenarios
    raise ValueError(f"only found {len(scenarios)} of {count} screened scenarios")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_id", type=int)
    parser.add_argument("--clean-root", type=Path, default=Path("data/clean"))
    parser.add_argument("--start", required=True, help="development dates only, UTC inclusive")
    parser.add_argument("--end", required=True, help="development dates only, UTC exclusive")
    parser.add_argument("--devices", nargs="+", required=True, help="development inverter IDs only")
    parser.add_argument("--poa-device", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--manifest", type=Path, default=Path("bench/m1_smoke.json"))
    parser.add_argument("--dest", type=Path, default=Path("data/scenarios"))
    args = parser.parse_args(argv)
    source = args.clean_root / f"{args.system_id}.parquet"
    frame = pd.read_parquet(source)
    scenarios = generate_smoke_scenarios(
        frame,
        args.system_id,
        args.start,
        args.end,
        tuple(args.devices),
        args.poa_device,
        args.count,
    )
    manifest = {
        "schema_version": 1,
        "purpose": "development_smoke",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "truth_scope": "injected_faults_only",
        "generator_version": "m1-v1",
        "selection_window": [x.isoformat() for x in window_bounds(args.start, args.end)],
        "development_devices": sorted(args.devices),
        "scenarios": [],
    }
    for scenario in scenarios:
        observations, audit = materialize_scenario(frame, scenario)
        destination = args.dest / scenario.scenario_id
        destination.mkdir(parents=True, exist_ok=True)
        observations.to_parquet(destination / "timeseries.parquet", index=False)
        (destination / "truth.json").write_text(json.dumps(scenario.to_dict(), indent=2) + "\n")
        (destination / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
        manifest["scenarios"].append(scenario.to_dict() | audit)
        print(f"{scenario.scenario_id}: {audit['injection']['changed_rows']} changed AC rows")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"{args.manifest}: {len(scenarios)} development scenarios, no held-out evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
