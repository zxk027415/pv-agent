import asyncio
import json

import pandas as pd
import pytest

from pvdiag.eval.run_bench import run_benchmark
from pvdiag.sim.benchmark import generate_candidates
from pvdiag.sim.benchmark import main as benchmark_main
from pvdiag.sim.faults import FaultSpec, inject_fault
from pvdiag.sim.scenarios import Scenario, materialize_scenario


def baseline(days=42):
    ts = pd.date_range("2019-01-01", periods=96 * days, freq="15min", tz="UTC")
    sun = ts.hour >= 16
    rows = []
    for device, metric, values in (
        ("inv01", "ac_power_kw", sun.astype(float) * 10),
        ("inv02", "ac_power_kw", sun.astype(float) * 20),
        ("inv03", "ac_power_kw", sun.astype(float) * 30),
        ("inv04", "ac_power_kw", sun.astype(float) * 40),
        ("poa", "poa_wm2", sun.astype(float) * 600),
        ("ambient", "ambient_c", 20.0),
        ("inv01", "module_c", 30.0),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "ts": ts,
                    "system_id": 2107,
                    "device_id": device,
                    "metric": metric,
                    "value": values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def at(frame, device, ts, metric="ac_power_kw"):
    return (
        (frame.device_id == device)
        & (frame.metric == metric)
        & (frame.ts == pd.Timestamp(ts, tz="UTC"))
    )


def test_soiling_elapsed_days_cleaning_reset_and_source_immutability():
    frame = baseline()
    original = frame.copy(deep=True)
    fault = FaultSpec(
        "F1",
        "inv01",
        "2019-01-01 16:00",
        "2019-02-01",
        soiling_rate_per_day=0.005,
        cleaning_times=("2019-01-11 16:00",),
    )
    result, audit = inject_fault(frame, 2107, fault)
    assert result.loc[at(frame, "inv01", "2019-01-01 16:00"), "value"].iloc[0] == 10
    assert result.loc[at(frame, "inv01", "2019-01-10 16:00"), "value"].iloc[0] == pytest.approx(
        9.55
    )
    assert result.loc[at(frame, "inv01", "2019-01-11 16:00"), "value"].iloc[0] == 10
    assert result.loc[at(frame, "inv01", "2019-01-21 16:00"), "value"].iloc[0] == pytest.approx(9.5)
    other = (frame.device_id != "inv01") | (frame.metric != "ac_power_kw")
    pd.testing.assert_frame_equal(result.loc[other], frame.loc[other])
    pd.testing.assert_frame_equal(frame, original)
    assert FaultSpec.from_dict(fault.to_dict()) == fault
    assert audit["changed_metric"] == "ac_power_kw"


def test_shading_uses_local_clock_across_dst_and_half_open_daily_window():
    ts = pd.to_datetime(
        [
            "2019-03-09 22:45Z",
            "2019-03-09 23:00Z",
            "2019-03-10 01:00Z",
            "2019-03-10 21:45Z",
            "2019-03-10 22:00Z",
            "2019-03-11 00:00Z",
        ]
    )
    frame = pd.DataFrame(
        {"ts": ts, "system_id": 2107, "device_id": "inv01", "metric": "ac_power_kw", "value": 10.0}
    )
    fault = FaultSpec(
        "F2",
        "inv01",
        "2019-03-09",
        "2019-03-12",
        severity=0.5,
        daily_window=("15:00", "17:00"),
        timezone="America/Los_Angeles",
    )
    result, _ = inject_fault(frame, 2107, fault)
    assert result.value.tolist() == [10, 5, 10, 10, 5, 10]
    assert FaultSpec.from_dict(fault.to_dict()) == fault
    assert result.ts.tolist() == frame.ts.tolist()


def test_shading_crosses_local_midnight():
    frame = baseline(days=2)
    fault = FaultSpec(
        "F2",
        "inv01",
        "2019-01-01",
        "2019-01-03",
        severity=0.3,
        daily_window=("23:00", "01:00"),
        timezone="Asia/Shanghai",
    )
    result, _ = inject_fault(frame, 2107, fault)
    assert result.loc[at(frame, "inv01", "2019-01-01 16:00"), "value"].iloc[0] == 7
    assert result.loc[at(frame, "inv01", "2019-01-01 17:00"), "value"].iloc[0] == 10


def test_irradiance_drift_preserves_all_actual_power_and_other_sensors():
    frame = baseline(days=14)
    fault = FaultSpec("F5", "poa", "2019-01-01 16:00", "2019-01-15", drift_per_day=0.005)
    result, audit = inject_fault(frame, 2107, fault)
    assert result.loc[at(frame, "poa", "2019-01-11 16:00", "poa_wm2"), "value"].iloc[
        0
    ] == pytest.approx(630)
    unchanged = frame.device_id != "poa"
    pd.testing.assert_frame_equal(result.loc[unchanged], frame.loc[unchanged])
    assert audit["changed_metric"] == "poa_wm2"
    assert FaultSpec.from_dict(fault.to_dict()) == fault


def test_communication_loss_preserves_rows_and_replaces_all_target_metrics_with_nan():
    frame = baseline(days=7)
    fault = FaultSpec("F6", "inv01", "2019-01-03", "2019-01-04")
    result, audit = inject_fault(frame, 2107, fault)
    target = (frame.device_id == "inv01") & (frame.ts >= "2019-01-03") & (frame.ts < "2019-01-04")
    assert result.loc[target, "value"].isna().all()
    assert len(result) == len(frame)
    assert audit["changed_rows"] == 192  # AC and module-temperature records, including night
    assert audit["changed_metric"] == "multiple_device_metrics"
    pd.testing.assert_frame_equal(result.loc[~target], frame.loc[~target])
    assert FaultSpec.from_dict(fault.to_dict()) == fault
    malformed = fault.to_dict() | {"params": None}
    with pytest.raises(TypeError, match="must be an object"):
        FaultSpec.from_dict(malformed)


@pytest.mark.parametrize(
    "fault",
    [
        {"type": "F1", "soiling_rate_per_day": 0.006},
        {"type": "F1", "soiling_rate_per_day": float("nan")},
        {"type": "F1", "soiling_rate_per_day": 0.005, "cleaning_times": ("2019-02-01",)},
        {"type": "F2", "severity": 0.2, "daily_window": ("15:00", "17:00"), "timezone": "UTC"},
        {"type": "F2", "severity": 0.5, "daily_window": ("15:01", "17:00"), "timezone": "UTC"},
        {
            "type": "F2",
            "severity": 0.5,
            "daily_window": ("15:00", "17:00"),
            "timezone": "Invalid/Zone",
        },
        {"type": "F5", "drift_per_day": -0.001},
        {"type": "F6", "severity": 0.5},
    ],
)
def test_extension_parameters_reject_unsupported_or_nonfinite_values(fault):
    with pytest.raises(ValueError):
        FaultSpec(device_id="inv01", start="2019-01-01", end="2019-01-08", **fault)


def scene(faults, **kwargs):
    return Scenario(
        "candidate",
        2107,
        ("2019-01-01", "2019-01-08"),
        ("inv01", "inv02", "inv03"),
        "poa",
        tuple(faults),
        **kwargs,
    )


def test_normal_candidates_are_unmodified_and_refused_by_evaluation(tmp_path):
    scenario = scene([])
    result, audit = materialize_scenario(baseline(), scenario)
    assert audit["injection"] is None and audit["evaluation_eligible"] is False
    assert scenario.to_dict()["truth_scope"] == "unmodified_baseline_candidate"
    assert Scenario.from_dict(scenario.to_dict()) == scenario
    assert result[result.metric == "ac_power_kw"].value.isna().sum() == 0
    with pytest.raises(ValueError, match="not normal ground truth"):
        asyncio.run(
            run_benchmark(
                {"scenarios": [scenario.to_dict()]},
                tmp_path,
                tmp_path / "no_metadata",
                tmp_path / "out",
                lambda: None,
            )
        )


def test_compound_sensor_drift_is_applied_after_physical_outage_independent_of_input_order():
    frame = baseline(days=7)
    # Baseline POA below the outage threshold must remain non-triggering even
    # though observed POA crosses the threshold after drift.
    # Exercise the primitive order through an eligible frame with one
    # non-triggering 190 bin; observed drift subsequently crosses the threshold.
    frame.loc[at(frame, "poa", "2019-01-03 16:00", "poa_wm2"), "value"] = 190
    outage = FaultSpec("F3", "inv01", "2019-01-03 16:00", "2019-01-03 18:00", poa_device_id="poa")
    drift = FaultSpec("F5", "poa", "2019-01-01", "2019-01-08", drift_per_day=0.02)
    a, audit = materialize_scenario(frame, scene([drift, outage], difficulty="hard"))
    b, _ = materialize_scenario(frame, scene([outage, drift], difficulty="hard"))
    pd.testing.assert_frame_equal(a, b)
    assert audit["composition_order"] == ["F3", "F5"]
    assert a.loc[at(frame, "inv01", "2019-01-03 16:00"), "value"].iloc[0] == 10
    assert a.loc[at(frame, "poa", "2019-01-03 16:00", "poa_wm2"), "value"].iloc[0] > 200
    assert a.loc[at(frame, "inv01", "2019-01-03 16:15"), "value"].iloc[0] == 0


def test_compound_rules_reject_nonhard_and_observation_masked_faults():
    outage = FaultSpec("F3", "inv01", "2019-01-03 16:00", "2019-01-03 18:00", poa_device_id="poa")
    comm = FaultSpec("F6", "inv01", "2019-01-03", "2019-01-04")
    with pytest.raises(ValueError, match="hard difficulty"):
        scene([outage, comm])
    with pytest.raises(ValueError, match="unobservable compound"):
        materialize_scenario(baseline(), scene([outage, comm], difficulty="hard"))
    drift = FaultSpec("F5", "inv01", "2019-01-01", "2019-01-08", drift_per_day=0.002)
    with pytest.raises(ValueError, match="query scope"):
        scene([drift])


def test_six_class_generator_is_deterministic_and_preserves_physical_rate_limits():
    frame = baseline(days=49)
    args = (
        frame,
        2107,
        "2019-01-01",
        "2019-02-19",
        ("inv01", "inv02", "inv03", "inv04"),
        "poa",
        "PST8PDT",
    )
    generated = generate_candidates(*args, per_type=2, normal_count=4)
    assert generated == generate_candidates(*args, per_type=2, normal_count=4)
    assert len(generated) == 16
    assert {s.faults[0].type for s in generated if s.faults} == {"F1", "F2", "F3", "F4", "F5", "F6"}
    assert all(s.split == "dev" for s in generated)
    assert all(
        0.001 <= s.faults[0].soiling_rate_per_day <= 0.005
        for s in generated
        if s.faults and s.faults[0].type == "F1"
    )
    for scenario in generated:
        assert Scenario.from_dict(scenario.to_dict()) == scenario
        result, audit = materialize_scenario(frame, scenario)
        assert len(result) > 0
        if scenario.faults:
            assert audit["injection"]["changed_rows"] > 0


def test_candidate_cli_outputs_observations_truth_and_versioned_manifest(tmp_path):
    source = tmp_path / "source.parquet"
    baseline(days=42).to_parquet(source, index=False)
    manifest = tmp_path / "candidate.json"
    dest = tmp_path / "data"
    assert (
        benchmark_main(
            [
                "2107",
                "--source",
                str(source),
                "--start",
                "2019-01-01",
                "--end",
                "2019-02-12",
                "--devices",
                "inv01",
                "inv02",
                "inv03",
                "inv04",
                "--poa-device",
                "poa",
                "--timezone",
                "PST8PDT",
                "--per-type",
                "1",
                "--normal-count",
                "2",
                "--manifest",
                str(manifest),
                "--dest",
                str(dest),
            ]
        )
        == 0
    )
    spec = json.loads(manifest.read_text())
    assert spec["generator_version"] == "m2-v1" and len(spec["scenarios"]) == 8
    for item in spec["scenarios"]:
        assert (dest / item["scenario_id"] / "timeseries.parquet").exists()


def test_cross_site_candidates_are_tagged_and_never_mix_with_dev_scenes():
    from pvdiag.sim.scenarios import validate_split_isolation

    frame = baseline(days=49)
    args = (frame, 2107, "2019-01-01", "2019-02-19", ("inv01", "inv02", "inv03", "inv04"), "poa", "PST8PDT")
    cross = generate_candidates(*args, per_type=1, normal_count=1, split="cross_site")
    assert {s.split for s in cross} == {"cross_site"}
    assert all(s.scenario_id.startswith("X2107-") for s in cross)
    dev = generate_candidates(*args, per_type=1, normal_count=1)
    with pytest.raises(ValueError, match="cross-site"):
        validate_split_isolation(dev + cross)
    with pytest.raises(ValueError, match="only for dev, test or cross_site"):
        generate_candidates(*args, per_type=1, normal_count=1, split="train")
    held = generate_candidates(*args, per_type=1, normal_count=0, split="test", fault_types=("F3", "F4"))
    assert [s.scenario_id[:6] for s in held] == ["T2107-", "T2107-"]
    assert {s.split for s in held} == {"test"} and {s.faults[0].type for s in held} == {"F3", "F4"}
    with pytest.raises(ValueError, match="distinct F1–F6"):
        generate_candidates(*args, per_type=1, normal_count=0, fault_types=("F3", "F9"))


def test_large_site_tolerance_is_explicit_bounded_and_versioned():
    from pvdiag.sim.baseline import screen_baseline

    frame = baseline(days=7)
    # One cloud-shadow-like daylight bin where a single inverter dips alone.
    dip = (frame.device_id == "inv01") & (frame.metric == "ac_power_kw") & (frame.ts == pd.Timestamp("2019-01-03 18:00", tz="UTC"))
    frame.loc[dip, "value"] *= 0.5
    args = (frame, 2107, "2019-01-01", "2019-01-08", ("inv01", "inv02", "inv03", "inv04"), "poa")
    strict = screen_baseline(*args)
    assert strict["screening_version"] == "m1-v1" and "inv01" not in strict["eligible_devices"]
    tolerant = screen_baseline(*args, max_peer_outlier_fraction=0.01)
    assert tolerant["screening_version"] == "m1-v1-large-site"
    assert tolerant["thresholds"]["allowed_peer_outlier_bins"] >= 1
    assert "inv01" in tolerant["eligible_devices"]
    with pytest.raises(ValueError, match="between 0 and 5%"):
        screen_baseline(*args, max_peer_outlier_fraction=0.2)


def test_daytime_poa_coverage_ignores_night_gaps_only_when_requested():
    from pvdiag.sim.baseline import screen_baseline

    frame = baseline(days=7)
    night = (frame.metric == "poa_wm2") & frame.ts.dt.hour.isin([7, 8, 9])
    frame = frame[~night]
    args = (frame, 2107, "2019-01-01", "2019-01-08", ("inv01", "inv02", "inv03", "inv04"), "poa")
    strict = screen_baseline(*args)
    assert strict["poa_coverage"] < 0.99 and strict["eligible_devices"] == []
    daytime = screen_baseline(*args, daytime_site=(38.996306, -122.134111))
    assert daytime["poa_coverage"] == 1.0
    assert daytime["screening_version"] == "m1-v1-daytime-poa"
    assert daytime["thresholds"]["poa_coverage_scope"] == "sun_above_horizon"
    assert set(daytime["eligible_devices"]) == {"inv01", "inv02", "inv03", "inv04"}
    with pytest.raises(ValueError, match="coordinates"):
        screen_baseline(*args, daytime_site=(120.0, 0.0))


def test_explicit_compound_candidates_are_hard_observable_and_on_distinct_targets():
    frame = baseline(days=49)
    args = (frame, 2107, "2019-01-01", "2019-02-19", ("inv01", "inv02", "inv03", "inv04"), "poa", "PST8PDT")
    options = {"per_type": 1, "normal_count": 0, "compounds": (("F3", "F5"), ("F2", "F4")), "per_compound": 1}
    scenes = generate_candidates(*args, **options)
    assert scenes == generate_candidates(*args, **options)
    compound = [s for s in scenes if len(s.faults) == 2]
    assert [s.scenario_id for s in compound] == ["D2107-F3F5-0001", "D2107-F2F4-0001"]
    for scene in compound:
        assert scene.difficulty == "hard"
        assert scene.faults[0].device_id != scene.faults[1].device_id
        _, audit = materialize_scenario(frame, scene)
        assert len(audit["injections"]) == 2 and all(a["changed_rows"] > 0 for a in audit["injections"])
    assert len(generate_candidates(*args, per_type=1, normal_count=0)) == 6
    for bad in ((("F3", "F3"),), (("F3", "F9"),)):
        with pytest.raises(ValueError, match="distinct F1–F6"):
            generate_candidates(*args, per_type=1, normal_count=0, compounds=bad, per_compound=1)
    with pytest.raises(ValueError, match="per-pair count"):
        generate_candidates(*args, per_type=1, normal_count=0, compounds=(("F3", "F5"),))
