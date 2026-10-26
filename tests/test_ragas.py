"""Core preparation/ledger checks run without installing Ragas in the main runtime."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_ablation import offline_model

from pvdiag.agent.llm import LLMReasoner
from pvdiag.eval.ragas import CachedCaller, GroundedAnswer, main, preflight_budget, run_ragas
from pvdiag.eval.ragas_inputs import RagasPlan, prepare_plan, validate_sources
from pvdiag.eval.ragas_store import RagasStore
from pvdiag.eval.retrieval import QASet
from pvdiag.knowledge.store import KnowledgeStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    dataset = QASet.model_validate_json((ROOT / "bench/kb_qa_dev.json").read_bytes())
    corpus = KnowledgeStore()
    return dataset, corpus, prepare_plan(dataset, corpus, question_ids=["KB001"])


def test_prepare_all_thirty_questions_with_provenance_and_no_model(inputs):
    dataset, corpus, _ = inputs
    plan = prepare_plan(dataset, corpus)
    validate_sources(plan, dataset, corpus)
    assert (
        len(plan.cases) == 30
        and plan.annotation_status == "project_authored_not_independently_reviewed"
    )
    assert plan.required_calls()["answer"] == 30
    assert plan.required_calls()["judge"] <= 180
    assert RagasPlan.model_validate_json(plan.model_dump_json()).sha256 == plan.sha256
    assert all(
        h.document.source_url and h.document.content_sha256
        for c in plan.cases
        for h in c.evidence.result.hits
    )


@pytest.mark.parametrize("ids", [["KB999"], ["KB001", "KB001"]])
def test_selection_rejects_unknown_or_duplicate_ids(inputs, ids):
    dataset, corpus, _ = inputs
    with pytest.raises(ValueError, match="selected question"):
        prepare_plan(dataset, corpus, question_ids=ids)


@pytest.mark.parametrize("change", ["text", "request", "source", "config"])
def test_prepared_evidence_must_keep_its_content_identity(inputs, change):
    _, _, plan = inputs
    raw = plan.model_dump(mode="json")
    evidence = raw["cases"][0]["evidence"]
    if change == "text":
        evidence["result"]["hits"][0]["text"] += "forged"
    elif change == "request":
        evidence["request"]["query"] = "different question"
    elif change == "source":
        evidence["source_sha256"] = "0" * 64
    else:
        raw["retrieval_config"] = {"mode": "different"}
    with pytest.raises(ValueError, match="identity or configuration"):
        RagasPlan.model_validate(raw)


def test_rehashed_forged_passage_is_rejected_against_licensed_source(inputs):
    dataset, corpus, plan = inputs
    raw = plan.model_dump(mode="json")
    evidence = raw["cases"][0]["evidence"]
    evidence["result"]["hits"][0]["text"] = "forged but self-consistently rehashed text"
    payload = {k: v for k, v in evidence.items() if k != "evidence_id"}
    evidence["evidence_id"] = (
        "ev_"
        + hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[
            :24
        ]
    )
    tampered = RagasPlan.model_validate(raw)
    with pytest.raises(ValueError, match="licensed corpus"):
        validate_sources(tampered, dataset, corpus)


def test_rehashed_retrieval_configuration_must_keep_its_aggregate_source_hash(inputs):
    dataset, corpus, plan = inputs
    raw = plan.model_dump(mode="json")
    raw["retrieval_config"]["bm25"]["k1"] = 9.0
    for case in raw["cases"]:
        evidence = case["evidence"]
        evidence["result"]["retrieval_config"] = raw["retrieval_config"]
        payload = {k: v for k, v in evidence.items() if k != "evidence_id"}
        evidence["evidence_id"] = (
            "ev_"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()[:24]
        )
    changed = RagasPlan.model_validate(raw)
    with pytest.raises(ValueError, match="source fingerprint"):
        validate_sources(changed, dataset, corpus)


@pytest.mark.parametrize("change", ["qa", "question", "manifest"])
def test_execute_requires_original_qa_and_corpus_snapshots(inputs, change):
    dataset, corpus, plan = inputs
    if change == "qa":
        dataset = dataset.model_copy(deep=True)
        dataset.questions[0].reference_answer = "changed reference"
    elif change == "question":
        plan = plan.model_copy(deep=True)
        plan.cases[0].question.reference_answer = "changed reference"
    else:
        plan = plan.model_copy(update={"corpus_manifest_sha256": "0" * 64})
    with pytest.raises(ValueError):
        validate_sources(plan, dataset, corpus)


@pytest.mark.parametrize(
    "allowed,budget",
    [
        ((False, True), {"answer": 1, "judge": 6}),
        ((True, False), {"answer": 1, "judge": 6}),
        ((True, True), {"answer": 0, "judge": 6}),
        ((True, True), {"answer": 1, "judge": 0}),
    ],
)
def test_roles_and_budgets_are_preflighted_before_factory_or_optional_import(
    inputs, tmp_path, allowed, budget
):
    dataset, corpus, plan = inputs

    def forbidden():
        pytest.fail("no SDK may be constructed after a failed preflight")

    with pytest.raises(ValueError):
        asyncio.run(
            run_ragas(
                plan,
                dataset,
                corpus,
                RagasStore(tmp_path / "jobs.sqlite3"),
                "preflight",
                forbidden,
                forbidden,
                budgets=budget,
                allow_generation=allowed[0],
                allow_judge=allowed[1],
            )
        )


def test_full_plan_cannot_silently_use_one_question_budgets(inputs):
    dataset, corpus, _ = inputs
    with pytest.raises(ValueError, match="cannot cover"):
        preflight_budget(
            prepare_plan(dataset, corpus),
            {"answer": 1, "judge": 6},
            allow_generation=True,
            allow_judge=True,
        )


def test_jobs_are_bound_and_explicit_resume_cannot_change_budgets(tmp_path):
    store = RagasStore(tmp_path / "jobs.sqlite3")
    store.bind("one", {"source": "fixed"}, {"answer": 1, "judge": 6})
    with pytest.raises(ValueError, match="explicit resume"):
        store.bind("one", {"source": "fixed"}, {"answer": 1, "judge": 6})
    with pytest.raises(ValueError, match="context or budgets"):
        store.bind("one", {"source": "fixed"}, {"answer": 2, "judge": 6}, resume=True)
    with pytest.raises(ValueError, match="context or budgets"):
        store.bind("one", {"source": "changed"}, {"answer": 1, "judge": 6}, resume=True)
    store.bind("one", {"source": "fixed"}, {"answer": 1, "judge": 6}, resume=True)
    with (
        store.exclusive_job("one"),
        pytest.raises(ValueError, match="already active"),
        RagasStore(store.path).exclusive_job("one"),
    ):
        pytest.fail("concurrent same-job controllers must not run")


def test_unknown_call_survives_process_exit_and_cannot_reset_its_budget(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    store = RagasStore(path)
    store.bind("crash", {}, {"answer": 1, "judge": 1})
    code = """
import asyncio, os, sys
from pvdiag.eval.ragas_store import RagasStore
async def main():
    store=RagasStore(sys.argv[1])
    with store.exclusive_job('crash'):
        await store.tracker('crash','KB001','judge','one').reserve()
        os._exit(17)
asyncio.run(main())
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, check=False, timeout=30
    )
    assert child.returncode == 17
    reopened = RagasStore(path)
    assert reopened.usage("crash", role="judge")[0]["usage_missing_calls"] == 1
    with reopened.exclusive_job("crash"), pytest.raises(ValueError, match="budget_exhausted"):
        asyncio.run(reopened.tracker("crash", "KB002", "judge", "retry").reserve())
    # The answer role has a separate reservation budget.
    assert asyncio.run(reopened.tracker("crash", "KB001", "answer", "answer").reserve())


def test_cached_valid_output_does_not_repeat_sdk_requests_or_mix_roles(tmp_path, monkeypatch):
    def handler(payload, schema):
        return GroundedAnswer(answer="受控回答。", cited_document_ids=["playbook-f1"])

    attempts = offline_model(monkeypatch, handler)
    store = RagasStore(tmp_path / "jobs.sqlite3")
    store.bind("cache", {}, {"answer": 1, "judge": 1})

    async def verify():
        model = LLMReasoner("offline", max_model_calls=30)
        first = CachedCaller(store, "cache", "KB001", "answer", model)
        result = await first.call(GroundedAnswer, {"task": "controlled"}, "fixture")
        assert await first.call(GroundedAnswer, {"task": "controlled"}, "fixture") == result
        second = CachedCaller(
            RagasStore(store.path),
            "cache",
            "KB001",
            "answer",
            LLMReasoner("offline", max_model_calls=30),
        )
        assert await second.call(GroundedAnswer, {"task": "controlled"}, "fixture") == result
        judge = CachedCaller(
            store, "cache", "KB001", "judge", LLMReasoner("offline", max_model_calls=30)
        )
        assert await judge.call(GroundedAnswer, {"task": "controlled"}, "fixture") == result

    asyncio.run(verify())
    assert len(attempts) == 2
    assert store.usage("cache", role="answer")[0]["model_calls"] == 1
    assert store.usage("cache", role="judge")[0]["model_calls"] == 1


def test_prepare_cli_never_initializes_any_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("prepare must be free of model/Judge initialization")

    monkeypatch.setattr("pvdiag.eval.ragas.load_metrics", forbidden)
    monkeypatch.setattr("pvdiag.agent.llm.init_chat_model", forbidden)
    output = tmp_path / "prepared.json"
    assert main(["prepare", "--question-ids", "KB001", "--output", str(output)]) == 0
    assert len(RagasPlan.model_validate_json(output.read_text()).cases) == 1
    assert not list(tmp_path.glob("*.sqlite3"))


def test_native_ragas_contracts_in_locked_optional_runtime():
    executable = ROOT / ".venv-eval/bin/python"
    if not executable.is_file():
        pytest.skip("install the locked optional .venv-eval runtime for native Ragas checks")
    environment = os.environ.copy()
    environment["RAGAS_DO_NOT_TRACK"] = "true"
    completed = subprocess.run(
        [str(executable), str(ROOT / "tests/ragas_contract.py")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"external_network_attempts": 0' in completed.stdout
    assert '"native_ragas_tests": 9' in completed.stdout
