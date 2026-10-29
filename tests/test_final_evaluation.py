"""Held-out/cross-site runs: free preflight, explicit opt-in and one scored run per stage."""

import asyncio
import json
from copy import deepcopy

import pytest
from test_agent import observation as observation  # noqa: PLC0414 -- re-export pytest fixture
from test_evolve import bench_layout

from pvdiag.agent.rules import RuleReasoner
from pvdiag.eval.run_bench import run_benchmark


def heldout(bench):
    held = deepcopy(bench)
    for scene in held["scenarios"]:
        scene["split"] = "test"
    return held


def run(manifest, tmp_path, observation, factory, out="out", **options):
    return asyncio.run(
        run_benchmark(
            manifest, tmp_path / "scenes", observation[1], tmp_path / out, factory, **options
        )
    )


def test_heldout_preflight_is_free_and_scored_runs_are_opt_in_and_one_shot(observation, tmp_path):
    held = heldout(bench_layout(observation, tmp_path))
    created = []

    def factory():
        created.append(1)
        return RuleReasoner()

    with pytest.raises(ValueError, match="only permits development"):
        run(held, tmp_path, observation, factory)
    with pytest.raises(ValueError, match="allow-final-evaluation"):
        run(held, tmp_path, observation, factory, scope="heldout")
    ledger = tmp_path / "ledger.json"
    check = run(held, tmp_path, observation, factory, scope="heldout", preflight_only=True)
    assert check["evaluation_consumed"] is False and check["scenarios"] == 2
    assert len(check["observation_snapshots"]) == 2
    assert created == [] and not (tmp_path / "out").exists() and not ledger.exists()

    final = {"scope": "heldout", "allow_final_evaluation": True, "final_ledger": ledger}
    summary = run(held, tmp_path, observation, factory, stage="round0", **final)
    assert summary["evaluation_scope"] == "heldout" and summary["final_stage"] == "round0"
    assert summary["scope"] == "heldout_injected_faults_only" and len(created) == 2
    assert [e["stage"] for e in json.loads(ledger.read_text())] == ["round0"]
    with pytest.raises(ValueError, match="already evaluated"):
        run(held, tmp_path, observation, factory, out="again", stage="round0", **final)
    assert len(created) == 2 and not (tmp_path / "again").exists()
    run(held, tmp_path, observation, factory, out="final", stage="final", **final)
    assert [e["stage"] for e in json.loads(ledger.read_text())] == ["round0", "final"]
    with pytest.raises(ValueError, match="stage label"):
        run(held, tmp_path, observation, factory, out="x", stage="bad stage!", **final)


def test_scope_must_match_every_split_and_final_runs_refuse_memory_or_candidate_harness(
    observation, tmp_path
):
    bench = bench_layout(observation, tmp_path)
    with pytest.raises(ValueError, match="split=test"):
        run(bench, tmp_path, observation, RuleReasoner, scope="heldout", preflight_only=True)
    with pytest.raises(ValueError, match="split=cross_site"):
        run(heldout(bench), tmp_path, observation, RuleReasoner, scope="cross_site", preflight_only=True)
    with pytest.raises(ValueError, match="frozen harness"):
        run(
            heldout(bench),
            tmp_path,
            observation,
            RuleReasoner,
            scope="heldout",
            preflight_only=True,
            harness_root=tmp_path,
        )
    dev = run(bench, tmp_path, observation, RuleReasoner, out="dev")
    assert dev["evaluation_scope"] == "dev" and dev["scope"] == "development_smoke_injected_faults_only"
