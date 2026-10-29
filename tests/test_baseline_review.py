import asyncio
import copy
import json
from dataclasses import replace

import pandas as pd
import pytest
from test_fault_extensions import baseline

from pvdiag.agent.rules import RuleReasoner
from pvdiag.eval.baseline_review import (
    CHECKS,
    ReviewLedger,
    canonical,
    digest,
    load_bundle,
    prepare,
    released_manifest,
    verify_release,
)
from pvdiag.eval.run_bench import run_benchmark
from pvdiag.sim.faults import FaultSpec
from pvdiag.sim.scenarios import Scenario, materialize_scenario, validate_split_isolation


@pytest.fixture
def review_fixture(tmp_path):
    source, metadata = tmp_path / "source.parquet", tmp_path / "metadata.json"
    frame = baseline(days=7)
    frame.to_parquet(source, index=False)
    metadata.write_text(
        json.dumps(
            {
                "System": {"system_id": 2107, "timezone_code": "UTC"},
                "Metrics": {
                    f"inv_{i:02d}_ac_power_{i}": {
                        "metric_id": i,
                        "common_name": "AC power",
                        "units": "kw",
                        "source_type": "INVERTER",
                        "source_id": None,
                    }
                    for i in range(1, 4)
                },
            }
        )
    )
    common = {
        "system_id": 2107,
        "query_window": ("2019-01-01", "2019-01-08"),
        "device_ids": ("inv01", "inv02", "inv03"),
        "poa_device_id": "poa",
    }
    scenes = [
        Scenario(
            "fixture-F4",
            **common,
            faults=(FaultSpec("F4", "inv01", "2019-01-04", "2019-01-08", loss_fraction=0.25),),
        ),
        Scenario("fixture-N", **common, faults=()),
    ]
    root = tmp_path / "scenarios"
    for scene in scenes:
        path = root / scene.scenario_id
        path.mkdir(parents=True)
        materialize_scenario(frame, scene)[0].to_parquet(path / "timeseries.parquet", index=False)
    manifest = {
        "schema_version": 2,
        "source_sha256": digest(source.read_bytes()),
        "scenarios": [s.to_dict() for s in scenes],
    }
    return manifest, source, metadata, root, tmp_path / "review"


def prepare_fixture(fixture):
    manifest, source, metadata, root, bundle = fixture
    return prepare(manifest, source, metadata, root, bundle)


def accept_fixture(bundle):
    _, packets = load_bundle(bundle)
    ledger = ReviewLedger(bundle / "reviews.sqlite3")
    for review_id in packets:
        ledger.record(
            review_id,
            accepted=True,
            reviewer="controlled-fixture-reviewer",
            reason="Controlled synthetic data; test fixture only.",
            checks=CHECKS,
        )


def test_review_preparation_binds_reproducible_source_and_keeps_packets_blind(review_fixture):
    catalog = prepare_fixture(review_fixture)
    manifest, source, _, root, bundle = review_fixture
    assert len(catalog["packet_ids"]) == 1 and len(catalog["bindings"]) == 2
    _, packets = load_bundle(bundle)
    packet = next(iter(packets.values()))
    text = canonical(packet).decode()
    assert all(
        key not in text
        for key in ['"faults"', '"injection"', '"scenario_id"', '"diagnosis"', "fixture-F4"]
    )
    assert packet["source_sha256"] == digest(source.read_bytes())
    assert packet["signals"]["ac_power_kw"]["inv01"][64] == 10
    assert all(t.endswith("+00:00") for t in packet["signals"]["utc"])
    assert ReviewLedger(bundle / "reviews.sqlite3").decisions() == {}
    assert manifest["scenarios"][1]["evaluation_eligible"] is False
    assert pd.read_parquet(root / "fixture-F4/timeseries.parquet").value.min() >= 0
    with pytest.raises(ValueError, match="new bundle"):
        prepare_fixture(review_fixture)


def test_review_rejects_source_and_any_exact_observation_change_before_creating_bundle(
    review_fixture,
):
    manifest, source, metadata, root, bundle = review_fixture
    forged = copy.deepcopy(manifest)
    forged["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source fingerprint"):
        prepare(forged, source, metadata, root, bundle)
    assert not bundle.exists()
    path = root / "fixture-N/timeseries.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "value"] += 1e-10
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="reproducible injection"):
        prepare_fixture(review_fixture)
    assert not bundle.exists()


@pytest.mark.parametrize("artifact", ["packet", "baseline", "catalog"])
def test_review_rejects_changed_packets_raw_snapshots_and_catalogs(review_fixture, artifact):
    catalog = prepare_fixture(review_fixture)
    *_, bundle = review_fixture
    review_id = catalog["packet_ids"][0]
    if artifact == "packet":
        path = bundle / "packets" / f"{review_id}.json"
        value = json.loads(path.read_text())
        value["scope"]["system_id"] = 999
        path.write_text(json.dumps(value))
    elif artifact == "baseline":
        path = bundle / "packets" / f"{review_id}.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        path = bundle / "catalog.json"
        value = json.loads(path.read_text())
        value["candidate_manifest"]["scenarios"][1]["device_ids"] = ["forged"]
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="modified|ledger binding"):
        load_bundle(bundle)


def test_local_review_decisions_are_named_complete_immutable_and_idempotent(review_fixture):
    catalog = prepare_fixture(review_fixture)
    *_, bundle = review_fixture
    review_id = catalog["packet_ids"][0]
    ledger = ReviewLedger(bundle / "reviews.sqlite3")
    for changes in [{"checks": []}, {"reviewer": " "}, {"reason": ""}, {"accepted": "yes"}]:
        with pytest.raises(ValueError):
            ledger.record(
                review_id,
                **(
                    {
                        "accepted": True,
                        "reviewer": "fixture",
                        "reason": "controlled data",
                        "checks": CHECKS,
                    }
                    | changes
                ),
            )
    with pytest.raises(ValueError, match="not registered"):
        ledger.record(
            "0" * 64, accepted=True, reviewer="fixture", reason="controlled data", checks=CHECKS
        )
    assert ledger.decisions() == {}
    decision = ledger.record(
        review_id, accepted=False, reviewer="fixture", reason="evidence insufficient", checks=[]
    )
    assert (
        ledger.record(
            review_id, accepted=False, reviewer="fixture", reason="evidence insufficient", checks=[]
        )
        == decision
    )
    with pytest.raises(ValueError, match="different baseline decision"):
        ledger.record(review_id, accepted=True, reviewer="fixture", reason="changed", checks=CHECKS)
    with pytest.raises(ValueError, match="accepted human review"):
        released_manifest(bundle)


def test_unreviewed_or_tampered_normal_labels_cannot_be_scored(review_fixture):
    prepare_fixture(review_fixture)
    manifest, _, metadata, root, bundle = review_fixture
    called = []

    def factory():
        called.append(True)
        return RuleReasoner()

    forged = copy.deepcopy(manifest)
    forged["scenarios"][1].update(evaluation_eligible=True, truth_scope="healthy")
    with pytest.raises(ValueError, match="not normal ground truth"):
        asyncio.run(run_benchmark(forged, root, metadata, bundle / "eval", factory))
    with pytest.raises(ValueError, match="accepted human review"):
        released_manifest(bundle)
    assert called == []
    accept_fixture(bundle)
    released = released_manifest(bundle)
    verify_release(released, bundle)
    released["scenarios"][1]["query_window"][0] = "2019-01-02"
    with pytest.raises(ValueError, match="does not match"):
        verify_release(released, bundle)


def test_reviewed_normal_and_fault_runs_use_exact_frozen_inputs_without_review_truth_in_state(
    review_fixture,
):
    prepare_fixture(review_fixture)
    _, _, metadata, root, bundle = review_fixture
    accept_fixture(bundle)
    manifest = released_manifest(bundle)
    seen = []

    class SpyRule(RuleReasoner):
        async def plan(self, state, tools, prompt):
            assert not {"faults", "review_id", "baseline_review", "scenario_id"} & state.keys()
            assert "controlled-fixture-reviewer" not in str(state)
            seen.append(state["request"].model_dump(mode="json"))
            return await super().plan(state, tools, prompt)

    summary = asyncio.run(
        run_benchmark(manifest, root, metadata, bundle / "eval", SpyRule, review_bundle=bundle)
    )
    assert summary["metrics"]["scenarios"] == 2
    assert summary["metrics"]["normal_scenarios"] == 1
    assert set(summary["by_fault_type"]) == {"F4", "N"}
    assert summary["metrics"]["input_tokens"] == 0
    assert len(summary["observation_snapshots"]) == 2
    assert all(s["device_ids"] == ["inv01", "inv02", "inv03"] for s in seen)


@pytest.mark.parametrize("changed", ["late_observation", "metadata"])
def test_all_reviewed_inputs_are_preflighted_before_first_reasoner_call(review_fixture, changed):
    prepare_fixture(review_fixture)
    _, _, metadata, root, bundle = review_fixture
    accept_fixture(bundle)
    manifest = released_manifest(bundle)
    if changed == "metadata":
        metadata.write_text(metadata.read_text() + " ")
    else:
        path = root / "fixture-N/timeseries.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "value"] += 1
        frame.to_parquet(path, index=False)
    called = []
    with pytest.raises(ValueError, match="differs|differ"):
        asyncio.run(
            run_benchmark(
                manifest,
                root,
                metadata,
                bundle / "eval",
                lambda: called.append(True),
                review_bundle=bundle,
            )
        )
    assert called == [] and not (bundle / "eval").exists()


def test_explicit_fault_only_mode_excludes_normals_and_preserves_fixed_scorer(review_fixture):
    manifest, _, metadata, root, bundle = review_fixture
    summary = asyncio.run(
        run_benchmark(manifest, root, metadata, bundle / "eval", RuleReasoner, faults_only=True)
    )
    assert summary["excluded_scenario_ids"] == ["fixture-N"]
    assert summary["metrics"]["normal_scenarios"] == 0
    assert summary["metrics"]["normal_false_alarm_rate"] is None
    assert summary["metrics"]["scorer_version"] == "v1.2"
    assert summary["by_fault_type"]["F4"]["scenarios"] == 1


@pytest.mark.parametrize("level", ["A0", "A1", "A2", "A3"])
def test_offline_ablation_runner_records_policy_and_never_passes_truth(
    review_fixture, monkeypatch, level
):
    from test_ablation import inconclusive, offline_model

    from pvdiag.agent.llm import LLMReasoner
    from pvdiag.knowledge.store import KnowledgeStore

    attempts = offline_model(monkeypatch, inconclusive)
    manifest, _, metadata, root, bundle = review_fixture
    summary = asyncio.run(
        run_benchmark(
            manifest,
            root,
            metadata,
            bundle / "eval",
            lambda: LLMReasoner("offline"),
            faults_only=True,
            ablation=level,
            knowledge_store=KnowledgeStore() if level == "A3" else None,
        )
    )
    saved = json.loads((bundle / "eval/fixture-F4.json").read_text())
    assert saved["ablation"]["level"] == summary["ablation"]["level"] == level
    assert saved["ablation"]["summary_sha256"] == summary["summary_snapshots"]["fixture-F4"]
    assert summary["verification_mode"] == ("domain_llm" if level == "A3" else "structural")
    assert saved["model"]["reported_models"] == ["offline-ablation-fixture"]
    assert summary["metrics"]["scorer_version"] == "v1.2"
    assert summary["metrics"]["normal_false_alarm_rate"] is None
    assert summary["excluded_scenario_ids"] == ["fixture-N"]
    for forbidden in ('"faults"', '"scenario_id"', '"truth_scope"', "fixture-F4", "fixture-N"):
        assert forbidden not in json.dumps(attempts)


def test_f5_target_isolation_includes_sensor_ids_even_when_all_inverters_and_dates_are_disjoint():
    fault = FaultSpec("F5", "poa", "2019-01-04", "2019-01-08", drift_per_day=0.005)
    dev = Scenario(
        "dev", 2107, ("2019-01-01", "2019-01-08"), ("inv01", "inv02", "inv03"), "poa", (fault,)
    )
    heldout = Scenario(
        "heldout",
        2107,
        ("2019-02-01", "2019-02-08"),
        ("inv04", "inv05", "inv06"),
        "poa",
        (replace(fault, start="2019-02-04", end="2019-02-08"),),
        split="test",
    )
    with pytest.raises(ValueError, match="fault target"):
        validate_split_isolation([dev, heldout])
    other = replace(
        heldout,
        poa_device_id="independent_poa",
        faults=(replace(heldout.faults[0], device_id="independent_poa"),),
    )
    validate_split_isolation([dev, other])


def test_heldout_baselines_can_be_reviewed_but_never_mixed_or_benchmarked(review_fixture, tmp_path):
    import asyncio
    from copy import deepcopy

    from pvdiag.eval.run_bench import run_benchmark

    manifest, source, metadata, root, bundle = review_fixture
    held = deepcopy(manifest)
    for scene in held["scenarios"]:
        scene["split"] = "test"
    prepare(held, source, metadata, root, bundle)
    mixed = deepcopy(manifest)
    mixed["scenarios"][0]["split"] = "test"
    with pytest.raises(ValueError, match="exactly one dev or test split|overlap"):
        prepare(mixed, source, metadata, root, tmp_path / "mixed")
    with pytest.raises(ValueError, match="only permits development"):
        asyncio.run(run_benchmark(held, root, metadata, tmp_path / "out", lambda: None))
