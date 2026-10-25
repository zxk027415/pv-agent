import json
from dataclasses import replace

import pandas as pd
import pytest

from pvdiag.sim.baseline import screen_baseline
from pvdiag.sim.faults import FaultSpec, inject_fault
from pvdiag.sim.scenarios import (
    Scenario,
    generate_smoke_scenarios,
    main,
    materialize_scenario,
    validate_split_isolation,
)

DEVICES = ("inv01", "inv02", "inv03", "inv04")


def baseline(start="2019-01-01", days=14):
    ts = pd.date_range(start, periods=96 * days, freq="15min", tz="UTC")
    daylight = (ts.hour >= 16) & (ts.hour < 23)
    parts = [
        pd.DataFrame(
            {
                "ts": ts,
                "system_id": 2107,
                "device_id": device,
                "metric": "ac_power_kw",
                "value": daylight.astype(float) * scale,
            }
        )
        for device, scale in zip(DEVICES, (10, 20, 30, 40), strict=True)
    ]
    parts += [
        pd.DataFrame(
            {"ts": ts, "system_id": 2107, "device_id": device, "metric": metric, "value": values}
        )
        for device, metric, values in (
            ("poa", "poa_wm2", daylight.astype(float) * 600),
            ("temperature", "ambient_c", 20.0),
            ("meter", "ac_power_kw", daylight.astype(float) * 100),
        )
    ]
    return pd.concat(parts, ignore_index=True)


def target_at(frame, device, time):
    return (frame.device_id == device) & (frame.ts == pd.Timestamp(time, tz="UTC"))


def make_scenario():
    fault = FaultSpec("F4", "inv01", "2019-01-04", "2019-01-08", loss_fraction=0.125)
    return Scenario(
        "S2107-F4-0001", 2107, ("2019-01-01", "2019-01-08"), DEVICES[:3], "poa", (fault,)
    )


def test_f3_threshold_nan_boundary_and_input_immutability():
    frame = baseline(days=1)
    frame.loc[target_at(frame, "poa", "2019-01-01 16:00"), "value"] = 200
    frame.loc[target_at(frame, "inv01", "2019-01-01 16:30"), "value"] = float("nan")
    original = frame.copy(deep=True)
    fault = FaultSpec("F3", "inv01", "2019-01-01 16:00", "2019-01-01 17:00", poa_device_id="poa")
    result, audit = inject_fault(frame, 2107, fault)
    pd.testing.assert_frame_equal(frame, original)
    changed = result.value.ne(frame.value) & ~(result.value.isna() & frame.value.isna())
    assert result.loc[changed, "ts"].tolist() == [
        pd.Timestamp("2019-01-01 16:15", tz="UTC"),
        pd.Timestamp("2019-01-01 16:45", tz="UTC"),
    ]
    assert result.loc[changed, "value"].eq(0).all()
    assert result.loc[target_at(frame, "inv01", "2019-01-01 17:00"), "value"].iloc[0] == 10
    assert result.loc[target_at(frame, "inv01", "2019-01-01 16:30"), "value"].isna().all()
    pd.testing.assert_frame_equal(result.loc[~changed], frame.loc[~changed])
    assert audit["changed_rows"] == 2


@pytest.mark.parametrize("fraction", [0.125, 0.25])
def test_f4_fraction_half_open_window_and_other_channels_unchanged(fraction):
    frame = baseline(days=1)
    original = frame.copy(deep=True)
    fault = FaultSpec("F4", "inv02", "2019-01-01 16:00", "2019-01-01 17:00", loss_fraction=fraction)
    result, audit = inject_fault(frame, 2107, fault)
    expected = frame.copy()
    target = (frame.device_id == "inv02") & frame.ts.between(
        pd.Timestamp("2019-01-01 16:00", tz="UTC"),
        pd.Timestamp("2019-01-01 17:00", tz="UTC"),
        inclusive="left",
    )
    expected.loc[target, "value"] = 20 * (1 - fraction)
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(frame, original)
    assert audit["changed_rows"] == 4


def test_f3_poa_alignment_uses_mean_without_forward_fill():
    ts = pd.date_range("2019-01-01 16:00", periods=3, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts": list(ts) * 2,
            "system_id": 2107,
            "device_id": ["inv01"] * 3 + ["poa"] * 3,
            "metric": ["ac_power_kw"] * 3 + ["poa_wm2"] * 3,
            "value": [10.0] * 3 + [100.0, 200.0, 400.0],
        }
    )
    fault = FaultSpec("F3", "inv01", "2019-01-01 16:00", "2019-01-01 16:15", poa_device_id="poa")
    result, audit = inject_fault(frame, 2107, fault)
    assert audit["changed_rows"] == 3
    assert result[result.device_id == "inv01"].value.eq(0).all()
    missing = frame[frame.device_id != "poa"]
    with pytest.raises(ValueError, match="missing POA"):
        inject_fault(missing, 2107, fault)


def test_injection_fails_for_missing_poa_noop_unknown_device_or_duplicate_keys():
    frame = baseline(days=1)
    fault = FaultSpec("F3", "inv01", "2019-01-01 16:00", "2019-01-01 17:00", poa_device_id="poa")
    missing = frame[~target_at(frame, "poa", "2019-01-01 16:15")]
    with pytest.raises(ValueError, match="missing POA"):
        inject_fault(missing, 2107, fault)
    with pytest.raises(ValueError, match="no target AC"):
        inject_fault(frame, 2107, replace(fault, device_id="unknown"))
    with pytest.raises(ValueError, match="change no values"):
        inject_fault(frame, 2107, replace(fault, start="2019-01-01 02:00", end="2019-01-01 03:00"))
    with pytest.raises(ValueError, match="duplicate normalized keys"):
        inject_fault(pd.concat([frame, frame.iloc[:1]], ignore_index=True), 2107, fault)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "F1"},
        {"type": "F4", "loss_fraction": 0.5},
        {"type": "F3", "poa_device_id": None},
        {"start": "2019-01-01 00:01"},
        {"end": "2018-01-01"},
    ],
)
def test_fault_parameter_validation(kwargs):
    values = {
        "type": "F3",
        "device_id": "inv01",
        "start": "2019-01-01",
        "end": "2019-01-02",
        "poa_device_id": "poa",
    } | kwargs
    with pytest.raises(ValueError):
        FaultSpec(**values)


def test_screen_uses_capacity_normalization_and_rejects_natural_zero():
    frame = baseline(days=7)
    report = screen_baseline(frame, 2107, "2019-01-01", "2019-01-08", DEVICES, "poa")
    assert report["eligible_devices"] == list(DEVICES)
    assert report["status"] == "screened_not_manually_verified"
    frame.loc[target_at(frame, "inv01", "2019-01-03 18:00"), "value"] = 0
    report = screen_baseline(frame, 2107, "2019-01-01", "2019-01-08", DEVICES, "poa")
    assert "inv01" not in report["eligible_devices"]
    assert report["devices"][0]["daylight_near_zero_bins"] == 1
    assert report["eligible_devices"] == list(DEVICES[1:])
    json.dumps(report, allow_nan=False)


def test_screen_rejects_missing_irradiance_and_is_not_a_normal_label():
    frame = baseline(days=7)
    frame = frame[frame.device_id != "poa"]
    report = screen_baseline(frame, 2107, "2019-01-01", "2019-01-08", DEVICES, "poa")
    assert report["poa_coverage"] == 0
    assert report["eligible_devices"] == []
    assert report["daylight_bins"] == 0


def test_scenario_serialization_scope_and_meter_exclusion():
    frame = baseline()
    # Per-device temperatures must not expose devices outside the selected cohort.
    hidden = frame[frame.device_id == "inv04"].copy()
    hidden["metric"] = "module_c"
    hidden["value"] = 25.0
    frame = pd.concat([frame, hidden], ignore_index=True)
    scenario = make_scenario()
    assert Scenario.from_dict(json.loads(json.dumps(scenario.to_dict()))) == scenario
    f3 = FaultSpec("F3", "inv01", "2019-01-01", "2019-01-02", poa_device_id="poa")
    assert FaultSpec.from_dict(f3.to_dict()) == f3
    observations, audit = materialize_scenario(frame, scenario)
    assert observations.ts.max() < pd.Timestamp("2019-01-08", tz="UTC")
    assert "meter" not in set(observations.device_id)
    assert "inv04" not in set(observations.device_id)
    assert audit["injection"]["changed_rows"] > 0
    assert audit["baseline_screen"]["eligible_devices"] == list(DEVICES[:3])
    invalid = replace(scenario.faults[0], start="2018-12-31")
    with pytest.raises(ValueError, match="fault device/window"):
        replace(scenario, faults=(invalid,))
    with pytest.raises(ValueError, match="invalid scenario_id"):
        replace(scenario, scenario_id="../../escape")


def test_split_isolation_checks_peer_devices_and_dates_independently():
    dev = make_scenario()
    # A different target still leaks if any visible peer is shared.
    test = replace(dev, scenario_id="heldout", split="test")
    with pytest.raises(ValueError, match="visible inverter IDs overlap"):
        validate_split_isolation([dev, test])
    fault = replace(test.faults[0], device_id="inv13")
    test = replace(test, device_ids=("inv13", "inv14", "inv15"), faults=(fault,))
    with pytest.raises(ValueError, match="query windows overlap"):
        validate_split_isolation([dev, test])
    fault = replace(fault, start="2020-01-04", end="2020-01-08")
    test = replace(test, query_window=("2020-01-01", "2020-01-08"), faults=(fault,))
    validate_split_isolation([dev, test])
    with pytest.raises(ValueError, match="cross-site system"):
        validate_split_isolation([dev, replace(test, split="cross_site")])
    with pytest.raises(ValueError, match="duplicate scenario IDs"):
        validate_split_isolation([dev, dev])


def test_generator_is_deterministic_and_refuses_dirty_baselines():
    frame = baseline()
    args = (frame, 2107, "2019-01-01", "2019-01-15", DEVICES, "poa")
    scenarios = generate_smoke_scenarios(*args, count=4)
    assert scenarios == generate_smoke_scenarios(*args, count=4)
    assert [s.faults[0].type for s in scenarios] == ["F3", "F4", "F3", "F4"]
    assert {s.split for s in scenarios} == {"dev"}
    # An obviously faulty device is never selected as target or visible peer.
    frame.loc[frame.device_id == "inv01", "value"] = 0.0
    screened = generate_smoke_scenarios(frame, *args[1:], count=4)
    assert all("inv01" not in s.device_ids for s in screened)
    with pytest.raises(ValueError, match="baseline failed screening"):
        materialize_scenario(frame, make_scenario())
    with pytest.raises(ValueError, match="only found"):
        generate_smoke_scenarios(*args, count=10)


def test_smoke_cli_writes_separate_observations_and_truth(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    baseline().to_parquet(clean / "2107.parquet", index=False)
    manifest_path = tmp_path / "manifest.json"
    destination = tmp_path / "scenarios"
    assert (
        main(
            [
                "2107",
                "--clean-root",
                str(clean),
                "--start",
                "2019-01-01",
                "--end",
                "2019-01-15",
                "--devices",
                *DEVICES,
                "--poa-device",
                "poa",
                "--count",
                "4",
                "--manifest",
                str(manifest_path),
                "--dest",
                str(destination),
            ]
        )
        == 0
    )
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["scenarios"]) == 4
    assert len(manifest["source_sha256"]) == 64
    for spec in manifest["scenarios"]:
        directory = destination / spec["scenario_id"]
        observations = pd.read_parquet(directory / "timeseries.parquet")
        assert set(observations.columns) == {"ts", "system_id", "device_id", "metric", "value"}
        assert (
            json.loads((directory / "truth.json").read_text())["truth_scope"]
            == "injected_faults_only"
        )


def test_generator_skips_pair_if_f4_onset_has_no_observable_power():
    frame = baseline(days=7)
    # Enough daylight for screening/F3, but no generation after the F4 onset.
    no_generation = frame.ts >= pd.Timestamp("2019-01-04", tz="UTC")
    frame.loc[no_generation & frame.metric.isin(["ac_power_kw", "poa_wm2"]), "value"] = 0.0
    with pytest.raises(ValueError, match="only found 0 of 2"):
        generate_smoke_scenarios(frame, 2107, "2019-01-01", "2019-01-08", DEVICES, "poa", count=2)
