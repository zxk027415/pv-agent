"""Long-term memory admission, A4 wiring and real pgvector recall boundaries."""

import asyncio
import hashlib
import json
import os
from argparse import Namespace
from uuid import uuid4

import pytest
from test_ablation import inconclusive, offline_model
from test_agent import inject, request
from test_agent import observation as observation  # noqa: PLC0414 -- re-export pytest fixture

from pvdiag.agent.__main__ import run_diagnosis
from pvdiag.agent.ablation import resolve_ablation, validate_ablation
from pvdiag.agent.llm import LLMReasoner
from pvdiag.agent.rules import RuleReasoner
from pvdiag.eval.run_bench import run_benchmark
from pvdiag.knowledge.store import KnowledgeStore
from pvdiag.mcp_servers.store import ObservationStore
from pvdiag.memory.__main__ import execute as memory_cli
from pvdiag.memory.cli import memory_from_args
from pvdiag.memory.models import MemoryLesson, scored_development_case
from pvdiag.memory.store import PgVectorStore, vector_text


def scenario(**updates):
    return {
        "scenario_id": "dev-f3",
        "split": "dev",
        "system_id": 2107,
        "device_ids": ["inv01", "inv02", "inv03"],
        "query_window": ["2019-01-01", "2019-01-08"],
        "faults": [
            {
                "type": "F3",
                "device_id": "inv01",
                "start": "2019-01-03T16:00:00Z",
                "end": "2019-01-03T18:00:00Z",
            }
        ],
    } | updates


def development_run(observation):
    inject(observation, "F3")
    return asyncio.run(run_diagnosis(ObservationStore(*observation), request(), RuleReasoner()))


def lesson(case_id, **updates):
    return {
        "title": "停机需同时看辐照",
        "text": "白天功率归零时先核对同期 POA，排除夜间与采集缺失。",
        "source_case_ids": [case_id],
        "outcome": "accepted",
        "reviewer": "tester",
        "review_reason": "受控测试",
        "review_method": "controlled_test",
    } | updates


def test_vectors_are_normalized_and_bad_embeddings_are_rejected():
    assert vector_text([3, 4], 2) == "[0.59999999999999998,0.80000000000000004]"
    for values in ([1, 0, 0], [0, 0], [float("nan"), 1]):
        with pytest.raises(ValueError, match="finite nonzero"):
            vector_text(values, 2)


def test_lessons_need_distinct_sources_and_nonblank_review():
    with pytest.raises(ValueError, match="duplicate"):
        MemoryLesson.model_validate(lesson("case_a", source_case_ids=["case_a", "case_a"]))
    with pytest.raises(ValueError, match="blank"):
        MemoryLesson.model_validate(lesson("case_a", review_reason="  "))


def test_development_case_admission_scores_but_never_exposes_truth(observation):
    run = development_run(observation)
    manifest = {"scenarios": [scenario()]}
    case = scored_development_case(manifest, "dev-f3", run)
    assert case.is_correct is True
    reference = json.dumps(case.reference(), ensure_ascii=False)
    for hidden in ("dev-f3", "2019-01-03T16:00:00Z", *run.evidence):
        assert hidden not in reference
    assert case.reference()["not_independent_field_confirmation"] is True

    for bad, message in (
        ({"scenarios": [scenario(split="test")]}, "only accept development"),
        ({"scenarios": [scenario(scenario_id="other")]}, "unknown"),
        ({"scenarios": [scenario(truth_scope="unmodified_baseline_candidate")]}, "unreviewed"),
        ({"scenarios": [scenario(faults=[])]}, "review bundle"),
    ):
        with pytest.raises(ValueError, match=message):
            scored_development_case(bad, "dev-f3", run)

    tampered = run.model_dump(mode="json")
    key = next(iter(tampered["evidence"]))
    tampered["evidence"][key]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        scored_development_case(manifest, "dev-f3", tampered)


class FakeSnapshot:
    sha256 = "1" * 64

    def model_dump(self, mode=None):
        return {"revision": 1}


class FakeView:
    snapshot = FakeSnapshot()

    def __init__(self):
        self.evidence_tools = []

    async def recall(self, request, evidence):
        self.evidence_tools.append(sorted({item["tool"] for item in evidence.values()}))
        return {"cases": [{"summary": "HISTORICAL_MARKER"}], "lessons": []}


def test_memory_view_is_required_by_a4_and_forbidden_elsewhere():
    reasoner = Namespace(metadata={"mode": "LLM"})
    for level in ("A0", "A1", "A2", "A3"):
        policy, _ = resolve_ablation(level)
        knowledge = "kb" if level == "A3" else None
        with pytest.raises(ValueError, match="only A4"):
            validate_ablation(policy, reasoner, knowledge, None, FakeView())
    with pytest.raises(ValueError, match="only A4"):
        validate_ablation(None, reasoner, None, None, FakeView())
    a4, mode = resolve_ablation("A4")
    assert mode == "domain_llm" and a4.feedback
    with pytest.raises(ValueError, match="only A4"):
        validate_ablation(a4, reasoner, "kb", None)
    with pytest.raises(ValueError, match="knowledge snapshot"):
        validate_ablation(a4, reasoner, None, None, FakeView())
    validate_ablation(a4, reasoner, "kb", None, FakeView())


def test_a4_recalls_after_fixed_summary_and_only_a4_model_inputs_see_memory(
    observation, monkeypatch
):
    attempts = offline_model(monkeypatch, inconclusive)
    view = FakeView()

    async def verify():
        a3 = await run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline"),
            ablation="A3",
            knowledge_store=KnowledgeStore(),
        )
        a3_inputs = len(attempts)
        a4 = await run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline"),
            ablation="A4",
            knowledge_store=KnowledgeStore(),
            memory_view=view,
        )
        return a3, a3_inputs, a4

    a3, a3_inputs, a4 = asyncio.run(verify())
    assert a3.memory is None
    assert "HISTORICAL_MARKER" not in json.dumps(attempts[:a3_inputs])
    assert view.evidence_tools == [
        ["data_quality", "get_site_info", "get_timeseries", "list_devices"]
    ]
    assert [a["task"] for a in attempts[a3_inputs:]] == ["plan", "hypothesize"]
    assert all(a["memory"]["cases"][0]["summary"] == "HISTORICAL_MARKER" for a in attempts[a3_inputs:])
    assert a4.memory == {"cases": [{"summary": "HISTORICAL_MARKER"}], "lessons": []}
    assert a4.ablation["level"] == "A4" and a4.ablation["summary_sha256"] == a3.ablation[
        "summary_sha256"
    ]


def test_benchmark_and_cli_reject_memory_configuration_before_any_work(tmp_path):
    def forbidden():
        pytest.fail("bad memory configuration must fail before creating a reasoner")

    for ablation, knowledge, view, message in (
        ("A4", KnowledgeStore(), None, "only A4"),
        ("A3", KnowledgeStore(), FakeView(), "only A4"),
        ("A4", None, FakeView(), "knowledge snapshot"),
    ):
        with pytest.raises(ValueError, match=message):
            asyncio.run(
                run_benchmark(
                    {"scenarios": [scenario()]},
                    tmp_path,
                    tmp_path / "missing",
                    tmp_path / "out",
                    forbidden,
                    knowledge_store=knowledge,
                    ablation=ablation,
                    memory_view=view,
                )
            )
    assert not (tmp_path / "out").exists()

    async def opened(**options):
        args = Namespace(
            ablation=None,
            memory_dsn_env="PVDIAG_MEMORY_DSN_MUST_NOT_BE_READ",
            memory_schema="pvdiag_memory",
            memory_revision=None,
            memory_embedding_path=None,
        )
        for key, value in options.items():
            setattr(args, key, value)
        async with memory_from_args(args) as view:
            return view

    assert asyncio.run(opened(ablation="A3")) is None
    with pytest.raises(ValueError, match="only valid with --ablation A4"):
        asyncio.run(opened(ablation="A3", memory_revision=1))
    with pytest.raises(ValueError, match="requires --memory-revision"):
        asyncio.run(opened(ablation="A4", memory_revision=1))


def test_lesson_cli_validates_drafts_and_abandoned_reviews_never_connect(tmp_path):
    draft = tmp_path / "lesson.json"
    args = Namespace(
        command="add-lesson", file=draft, dsn_env="PVDIAG_MEMORY_DSN_MUST_NOT_BE_READ"
    )
    draft.write_text(json.dumps(lesson("case_a")))
    with pytest.raises(ValueError, match="only title"):
        asyncio.run(memory_cli(args, review=lambda d: pytest.fail("invalid drafts are not shown")))
    draft.write_text(json.dumps({"title": "t", "text": "x", "source_case_ids": ["case_a"]}))
    assert asyncio.run(memory_cli(args, review=lambda d: None)) == {"status": "not_reviewed"}


class HashEmbedder:
    dimensions = 8

    def __init__(self, backend="test-hash-embedding"):
        self.identity = {"backend": backend, "dimensions": 8}

    def encode(self, text, *, query=False):
        raw = hashlib.sha256(text.encode()).digest()
        return [b - 127.5 for b in raw[: self.dimensions]]


@pytest.fixture
def pg():
    dsn = os.getenv("PVDIAG_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set an explicitly isolated PVDIAG_TEST_POSTGRES_DSN for real PostgreSQL tests")
    return dsn, "pvdiag_memory_test_" + uuid4().hex


def test_pgvector_initialization_and_writes_are_explicit(pg):
    dsn, schema = pg
    embedder = HashEmbedder()

    async def verify():
        with pytest.raises(ValueError, match="explicit write mode"):
            async with PgVectorStore.open(dsn, embedder, schema=schema, initialize=True):
                pytest.fail("initialization must not default to read-only")
        with pytest.raises(ValueError, match="initialize"):
            async with PgVectorStore.open(dsn, embedder, schema=schema):
                pytest.fail("opening an absent schema must not initialize it")
        async with PgVectorStore.open(
            dsn, embedder, schema=schema, initialize=True, read_only=False
        ):
            pass
        async with PgVectorStore.open(dsn, embedder, schema=schema) as reader:
            with pytest.raises(ValueError, match="read-only"):
                await reader.add_lesson(lesson("case_a"))
            assert (await reader.view()).snapshot.revision == 0

        with pytest.raises(ValueError, match="identity"):
            async with PgVectorStore.open(dsn, HashEmbedder("another-embedding"), schema=schema):
                pytest.fail("a different embedder cannot read existing vectors")

    asyncio.run(verify())


def test_pgvector_recall_is_frozen_and_excludes_overlapping_sources(pg, observation):
    dsn, schema = pg
    run = development_run(observation)
    manifest = {"scenarios": [scenario()]}
    embedder = HashEmbedder()
    elsewhere = request(start="2019-02-01", end="2019-02-08")

    async def verify():
        async with PgVectorStore.open(
            dsn, embedder, schema=schema, initialize=True, read_only=False
        ) as store:
            added = await store.add_case(manifest, "dev-f3", run)
            assert await store.add_case(manifest, "dev-f3", run) == added
            with pytest.raises(ValueError, match="existing development support"):
                await store.add_lesson(lesson("case_" + "0" * 64))
            await store.add_lesson(lesson(added["id"]))
            await store.add_lesson(lesson(added["id"], title="被拒经验", outcome="rejected"))
            frozen = await store.view(revision=added["revision"])
            latest = await store.view()
            assert (frozen.snapshot.cases, frozen.snapshot.lessons) == (1, 0)
            assert (latest.snapshot.cases, latest.snapshot.lessons) == (1, 2)

            remote = await frozen.recall(elsewhere, {})
            assert remote["snapshot"]["revision"] == added["revision"]
            assert [c["case_id"] for c in remote["cases"]] == [added["id"]]
            assert remote["lessons"] == []
            assert "dev-f3" not in json.dumps(remote, ensure_ascii=False)

            current = await latest.recall(elsewhere, {})
            assert [item["title"] for item in current["lessons"]] == ["停机需同时看辐照"]
            overlapping = await latest.recall(request(), {})
            assert overlapping["cases"] == [] and overlapping["lessons"] == []

        async with PgVectorStore.open(dsn, embedder, schema=schema) as reader:
            reopened = await reader.view(revision=added["revision"])
            assert reopened.snapshot == frozen.snapshot
            with pytest.raises(ValueError, match="unavailable"):
                await reader.view(revision=latest.snapshot.revision + 1)

    asyncio.run(verify())
