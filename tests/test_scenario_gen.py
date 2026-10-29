"""Failure-driven scene generation stays in the development pool and is reproducible."""

import hashlib
import json

import pandas as pd
import pytest
from test_fault_extensions import baseline

from pvdiag.sim.benchmark import generate_candidates
from pvdiag.sim.scenario_gen import (
    allocate,
    generate_from_failures,
    manifest_sha256,
)
from pvdiag.sim.scenario_gen import main as generator_main
from pvdiag.sim.scenarios import Scenario, materialize_scenario, validate_split_isolation

DEVICES = ("inv01", "inv02", "inv03", "inv04")
WINDOW = ["2019-01-01 00:00:00+00:00", "2019-02-19 00:00:00+00:00"]


def cloudy_frame():
    frame = baseline(days=49)
    # Days 8–12 are overcast for the whole station: power and POA drop together.
    overcast = (frame.ts >= "2019-01-08") & (frame.ts < "2019-01-13")
    scaled = overcast & frame.metric.isin(["ac_power_kw", "poa_wm2"])
    frame.loc[scaled, "value"] *= 0.3
    return frame


def parent(frame, source_sha="0" * 64):
    scenes = generate_candidates(
        frame, 2107, *WINDOW, DEVICES, "poa", "PST8PDT", per_type=1, normal_count=1, window_days=7
    )
    return {
        "schema_version": 2,
        "generator_version": "m2-v1",
        "source_sha256": source_sha,
        "selection_window": WINDOW,
        "development_devices": list(DEVICES),
        "window_days": 7,
        "site_timezone": "PST8PDT",
        "scenarios": [s.to_dict() for s in scenes],
    }


def case(sid, passed, fault=True):
    return {
        "scenario_id": sid,
        "actual_fault": fault,
        "joint_root_device_correct": passed if fault else None,
        "temporal_hits": 1 if fault else 0,
        "temporal_denominator": 1 if fault else 0,
        "normal_correct": None if fault else passed,
        "valid_numeric_claims": 0,
        "numeric_claims": 0,
        "unapproved_write_executions": 0,
    }


def summary(manifest, failing=("F1", "F2")):
    cases = []
    for scene in manifest["scenarios"]:
        kinds = {f["type"] for f in scene["faults"]}
        cases.append(case(scene["scenario_id"], not kinds & set(failing), bool(kinds)))
    return {"manifest_sha256": manifest_sha256(manifest), "cases": cases}


def test_budget_follows_failures_with_deterministic_remainders():
    profile = {k: {"evaluated": 10, "failed": 0} for k in ("F1", "F2", "F3", "F4", "F5", "F6")}
    profile["F1"]["failed"], profile["F5"]["failed"] = 3, 1
    assert allocate(profile, 6) == {"F1": 5, "F5": 1}
    assert allocate(profile, 4) == {"F1": 3, "F5": 1}
    with pytest.raises(ValueError, match="no development fault failures"):
        allocate({k: {"evaluated": 1, "failed": 0} for k in profile}, 4)


def test_generated_scenes_target_failures_alternate_weather_and_stay_isolated():
    frame = cloudy_frame()
    manifest = parent(frame)
    report_input = summary(manifest)
    scenes, details, report = generate_from_failures(frame, manifest, report_input, 4)
    again = generate_from_failures(frame, manifest, report_input, 4)
    assert (scenes, details, report) == again
    assert report["allocation"] == {"F1": 2, "F2": 2}
    assert report["failure_profile"]["F1"] == {"evaluated": 1, "failed": 1}
    assert report["normal_profile"]["generated"] == 0
    assert [s.faults[0].type for s in scenes] == ["F1", "F1", "F2", "F2"]
    assert all(s.split == "dev" and s.scenario_id.startswith(f"G2107-{report['tag']}-") for s in scenes)
    assert all(s.faults[0].device_id in DEVICES for s in scenes)
    fractions = [details[s.scenario_id]["cloudy_day_fraction"] for s in scenes]
    assert fractions[0] < fractions[1] and fractions[2] < fractions[3]
    assert all(
        s.difficulty == ("hard" if details[s.scenario_id]["cloudy_day_fraction"] >= 0.3 else "medium")
        for s in scenes
        if s.faults[0].type == "F2"
    )
    base = [Scenario.from_dict(s) for s in manifest["scenarios"]]
    used = {(s.query_window, s.device_ids, s.faults[0].type) for s in base if s.faults}
    assert not {(s.query_window, s.device_ids, s.faults[0].type) for s in scenes} & used
    validate_split_isolation(base + scenes)
    for scene in scenes:
        assert Scenario.from_dict(scene.to_dict()) == scene
        assert materialize_scenario(frame, scene)[1]["injection"]["changed_rows"] > 0
        lower, upper = map(pd.Timestamp, WINDOW)
        assert lower <= pd.Timestamp(scene.query_window[0]) < pd.Timestamp(scene.query_window[1]) <= upper


def test_unbound_heldout_or_normal_only_inputs_are_refused():
    frame = cloudy_frame()
    manifest = parent(frame)
    stale = summary(manifest) | {"manifest_sha256": "f" * 64}
    with pytest.raises(ValueError, match="not produced from this manifest"):
        generate_from_failures(frame, manifest, stale, 4)
    foreign = summary(manifest)
    foreign["cases"].append(case("S-other", False))
    with pytest.raises(ValueError, match="belong to the parent"):
        generate_from_failures(frame, manifest, foreign, 4)
    held = json.loads(json.dumps(manifest))
    held["scenarios"][0]["split"] = "test"
    with pytest.raises(ValueError, match="only reads development"):
        generate_from_failures(frame, held, summary(held), 4)
    normal_only = summary(manifest, failing=())
    for item in normal_only["cases"]:
        if not item["actual_fault"]:
            item["normal_correct"] = False
    with pytest.raises(ValueError, match="no development fault failures"):
        generate_from_failures(frame, manifest, normal_only, 4)
    with pytest.raises(ValueError, match="1–200"):
        generate_from_failures(frame, manifest, summary(manifest), 0)


def test_cli_binds_the_source_snapshot_and_never_overwrites(tmp_path):
    frame = cloudy_frame()
    source = tmp_path / "source.parquet"
    frame.to_parquet(source, index=False)
    manifest = parent(frame, hashlib.sha256(source.read_bytes()).hexdigest())
    (tmp_path / "parent.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps(summary(manifest)))
    common = [
        "--manifest", str(tmp_path / "parent.json"),
        "--summary", str(tmp_path / "summary.json"),
        "--source", str(source),
        "--budget", "2",
        "--dest", str(tmp_path / "scenes"),
    ]
    assert generator_main([*common, "--output", str(tmp_path / "generated.json")]) == 0
    spec = json.loads((tmp_path / "generated.json").read_text())
    assert spec["generator_version"] == "gen-v1"
    assert spec["purpose"] == "development_pool_generated_from_failures_not_heldout"
    assert [s["split"] for s in spec["scenarios"]] == ["dev", "dev"]
    for item in spec["scenarios"]:
        folder = tmp_path / "scenes" / item["scenario_id"]
        assert (folder / "timeseries.parquet").exists() and item["truth_scope"] == "injected_faults_only"
        assert "cloudy_day_fraction" in item["generation"]
    with pytest.raises(SystemExit):
        generator_main([*common, "--output", str(tmp_path / "generated.json")])
    frame.assign(value=frame.value + 1).to_parquet(source, index=False)
    with pytest.raises(SystemExit):
        generator_main([*common, "--output", str(tmp_path / "other.json")])
    assert not (tmp_path / "other.json").exists()
