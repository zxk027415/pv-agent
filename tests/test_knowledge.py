import asyncio
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from pydantic import ValidationError
from test_verification import context

from pvdiag.agent.__main__ import run_diagnosis
from pvdiag.agent.gateway import MCPGateway
from pvdiag.agent.guards import check_report
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.state import DiagnosisReport, NumericClaim, ToolCall
from pvdiag.eval.retrieval import QASet, evaluate
from pvdiag.knowledge.build_corpus import EXCERPTS, LICENSE_SHA256, build
from pvdiag.knowledge.models import PlaybookRequest, SearchRequest
from pvdiag.knowledge.retrieval import Retriever
from pvdiag.knowledge.store import DEFAULT_KB, KnowledgeStore
from pvdiag.mcp_servers.pv_analytics import create_server as analytics_server
from pvdiag.mcp_servers.pv_data import create_server as data_server
from pvdiag.mcp_servers.pv_knowledge import create_server as knowledge_server


@pytest.fixture
def copied_kb(tmp_path):
    root = tmp_path / "kb"
    shutil.copytree(DEFAULT_KB, root)
    shutil.copytree(DEFAULT_KB.parent / "playbooks", tmp_path / "playbooks")
    return root


def alter_manifest(root, change):
    path = root / "manifest.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_reviewed_corpus_rebuilds_exactly_and_retains_upstream_license(copied_kb):
    before = {p.relative_to(copied_kb): p.read_bytes() for p in copied_kb.rglob("*") if p.is_file()}
    manifest = build(copied_kb)
    after = {p.relative_to(copied_kb): p.read_bytes() for p in copied_kb.rglob("*") if p.is_file()}
    assert before == after
    assert len(manifest.documents) == 11
    assert (
        hashlib.sha256((copied_kb / "licenses/pvlib-LICENSE.txt").read_bytes()).hexdigest()
        == LICENSE_SHA256
    )
    by_id = {d.document_id: d for d in manifest.documents}
    for identifier, _, function, _ in EXCERPTS:
        doc = by_id[identifier]
        assert (copied_kb / doc.path).read_text() == inspect.getdoc(function) + "\n"
        assert (
            doc.source_sha256
            == hashlib.sha256(Path(inspect.getfile(function)).read_bytes()).hexdigest()
        )
        assert doc.license == "BSD-3-Clause" and doc.license_sha256 == LICENSE_SHA256


@pytest.mark.parametrize(
    "query,fault",
    [
        ("积灰清洗恢复", "F1"),
        ("局部遮挡", "F2"),
        ("inverter outage", "F3"),
        ("string loss", "F4"),
        ("辐照传感器漂移", "F5"),
        ("communication gap", "F6"),
    ],
)
def test_bilingual_retrieval_and_playbooks_keep_provenance(query, fault):
    store = KnowledgeStore()
    response = store.search(SearchRequest(query=query, top_k=2)).result
    assert fault in response.hits[0].document.fault_types
    assert response.retrieval_mode == "bm25" and response.evidence_kind == "knowledge_reference"
    assert all(h.document.source_url and h.document.content_sha256 for h in response.hits)
    playbook = store.playbook(PlaybookRequest(fault_type=fault)).result.playbook
    assert playbook.document.kind == "project_playbook"
    assert playbook.document.license == "project-authored"
    assert "name:" in playbook.text and "description:" in playbook.text


def test_empty_search_and_fault_filter_do_not_invent_hits():
    store = KnowledgeStore()
    assert store.search(SearchRequest(query="zzqqxxunrelatedzzqq")).result.hits == []
    hits = store.search(SearchRequest(query="power loss", fault_type="F5", top_k=5)).result.hits
    assert hits and all("F5" in hit.document.fault_types for hit in hits)
    for arguments in (
        {"query": "  "},
        {"query": "power", "top_k": 6},
        {"query": "power", "fault_type": "N"},
    ):
        with pytest.raises(ValidationError):
            SearchRequest(**arguments)


def test_snapshot_stability_and_edited_files_require_a_new_manifest(copied_kb):
    store = KnowledgeStore(copied_kb)
    request = SearchRequest(query="积灰")
    initial = store.search(request)
    assert initial == KnowledgeStore(copied_kb).search(request)
    document = store.documents[0]
    path = copied_kb / document.path
    path.write_text(path.read_text() + "\nchanged reference\n")
    assert store.search(request) == initial
    with pytest.raises(ValueError, match="hash mismatch"):
        KnowledgeStore(copied_kb)
    altered_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    alter_manifest(copied_kb, lambda m: m["documents"][0].update(content_sha256=altered_hash))
    changed = KnowledgeStore(copied_kb)
    assert changed.source_sha256 != store.source_sha256
    assert changed.search(request).evidence_id != initial.evidence_id


@pytest.mark.parametrize(
    "path", ["../bench/truth.json", "/tmp/truth.json", "excerpts/../truth.json", "data/truth.json"]
)
def test_manifest_cannot_escape_approved_content_folders(copied_kb, path):
    alter_manifest(copied_kb, lambda m: m["documents"][0].update(path=path))
    with pytest.raises(ValueError, match="approved folder"):
        KnowledgeStore(copied_kb)


def test_symlink_escape_unreviewed_license_and_duplicates_are_rejected(copied_kb, tmp_path):
    manifest = json.loads((copied_kb / "manifest.json").read_text())
    doc = manifest["documents"][0]
    original = copied_kb / doc["path"]
    outside = tmp_path / "outside.txt"
    original.rename(outside)
    original.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        KnowledgeStore(copied_kb)
    original.unlink()
    outside.rename(original)
    alter_manifest(copied_kb, lambda m: m["documents"][0].update(license="unknown"))
    with pytest.raises(ValidationError):
        KnowledgeStore(copied_kb)
    (copied_kb / "manifest.json").write_text(json.dumps(manifest))
    alter_manifest(copied_kb, lambda m: m["documents"].append(m["documents"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        KnowledgeStore(copied_kb)


def test_knowledge_mcp_schemas_and_real_stdio_protocol():
    async def verify():
        for transport in (
            knowledge_server(KnowledgeStore()),
            StdioTransport(sys.executable, ["-m", "pvdiag.mcp_servers.pv_knowledge"]),
        ):
            async with Client(transport) as client:
                tools = await client.list_tools()
                assert {t.name for t in tools} == {"search_fault_kb", "get_playbook"}
                assert all(
                    t.annotations.read_only_hint and not t.annotations.destructive_hint
                    for t in tools
                )
                assert all(t.input_schema and t.output_schema for t in tools)
                response = await client.call_tool(
                    "search_fault_kb", {"request": {"query": "积灰清洗"}}
                )
                assert response.structured_content["result"]["hits"]
                assert response.structured_content == KnowledgeStore().search(
                    SearchRequest(query="积灰清洗")
                ).model_dump(mode="json")
                response = await client.call_tool("get_playbook", {"request": {"fault_type": "F6"}})
                assert (
                    response.structured_content["result"]["playbook"]["document"]["document_id"]
                    == "playbook-f6"
                )

    asyncio.run(verify())


def test_optional_gateway_pins_knowledge_separately_and_denies_unmounted_tools(tmp_path):
    observation, request, scope, _, _ = context(tmp_path, "F3")
    kb = KnowledgeStore()
    call = ToolCall(name="search_fault_kb", arguments={"query": "停机"})

    async def verify():
        async with MCPGateway.connect(
            data_server(observation), analytics_server(observation)
        ) as gateway:
            assert "search_fault_kb" not in gateway.tools
            result, trace = await gateway.call(call, request, None, "investigate")
            assert result is None and not trace.success
        async with MCPGateway.connect(
            data_server(observation),
            analytics_server(observation),
            knowledge_server(kb),
            knowledge_source_sha256=kb.source_sha256,
        ) as gateway:
            assert scope.source_sha256 != kb.source_sha256
            result, trace = await gateway.call(call, request, scope, "investigate")
            assert trace.success and result["source_sha256"] == kb.source_sha256
            result, trace = await gateway.call(call, request, scope, "guard_in")
            assert result is None and not trace.success
        async with MCPGateway.connect(
            data_server(observation),
            analytics_server(observation),
            knowledge_server(kb),
            knowledge_source_sha256="0" * 64,
        ) as gateway:
            result, trace = await gateway.call(call, request, None, "investigate")
            assert result is None and trace.error == "ValueError"
        with pytest.raises(ValueError, match="fingerprint"):
            async with MCPGateway.connect(
                data_server(observation), analytics_server(observation), knowledge_server(kb)
            ):
                pytest.fail("an unpinned knowledge server must not be mounted")

    asyncio.run(verify())


def test_knowledge_references_cannot_replace_measurements_or_supply_plant_numbers(tmp_path):
    _, request, scope, evidence, report = context(tmp_path, "F3")
    kb = KnowledgeStore()
    references = [
        kb.search(SearchRequest(query="停机")),
        kb.playbook(PlaybookRequest(fault_type="F3")),
    ]
    evidence.update({item.evidence_id: item.model_dump(mode="json") for item in references})
    ids = [item.evidence_id for item in references]
    report.findings[0].evidence_ids = ids
    report.evidence_ids = ids
    assert "insufficient_distinct_tools" in check_report(report, evidence, request, scope)
    normal = DiagnosisReport(status="normal", summary="未见异常。", evidence_ids=ids)
    assert "normal_verdict_requires_measurements" in check_report(normal, evidence, request, scope)
    report.findings[0].numbers = [
        NumericClaim(
            label="检索分数冒充测量",
            value=references[0].result.hits[0].score,
            evidence_id=ids[0],
            path="/result/hits/0/score",
        )
    ]
    assert "invalid_numeric_citation" in check_report(report, evidence, request, scope)


def test_graph_fetches_knowledge_on_request_without_changing_observation_scope(tmp_path):
    store, request, _, _, _ = context(tmp_path, "F3")
    kb = KnowledgeStore()

    class ReadReferences(RuleReasoner):
        async def plan(self, state, tools, prompt):
            assert {"search_fault_kb", "get_playbook"} <= tools.keys()
            plan = await super().plan(state, tools, prompt)
            if state["loops"] == 0:
                plan.calls.extend(
                    [
                        ToolCall(name="search_fault_kb", arguments={"query": "停机与通讯缺测"}),
                        ToolCall(name="get_playbook", arguments={"fault_type": "F3"}),
                    ]
                )
            return plan

    result = asyncio.run(
        run_diagnosis(
            store, request, ReadReferences(), knowledge_store=kb, verification_mode="domain"
        )
    )
    assert result.report.status == "fault"
    assert result.knowledge_source_sha256 == kb.source_sha256
    assert result.request == request
    references = [
        e for e in result.evidence.values() if e["tool"] in {"search_fault_kb", "get_playbook"}
    ]
    assert len(references) == 2 and all(e["source_sha256"] == kb.source_sha256 for e in references)
    assert all(t.success for t in result.trace)


def test_hybrid_fusion_reranks_dense_only_hits_and_does_not_silently_fall_back():
    # A neural synonym match absent from BM25 must reach the reranker.
    class Neural:
        def __init__(self):
            self.identity = {"mode": "offline-test-double-not-a-model-result"}

        def encode(self, texts, *, query):
            return [[1, 0]] if query else [[0, 1], [1, 0], [-1, 0]]

        def rerank(self, query, texts):
            self.seen = texts
            return [0.9 if text == "逆变器无输出" else 0.1 for text in texts]

    backend = Neural()
    texts = ["inverter outage", "逆变器无输出", "通讯缺测"]
    retriever = Retriever(texts, mode="hybrid_bge", neural=backend)
    assert retriever.search("inverter outage", [0, 1, 2], 1)[0][0] == 1
    assert "逆变器无输出" in backend.seen
    assert retriever.search("inverter outage", [0, 2], 1)[0][0] != 1
    with pytest.raises(ValueError, match="requires"):
        Retriever(texts, mode="hybrid_bge")
    backend.rerank = lambda query, texts: [float("nan")] * len(texts)
    with pytest.raises(ValueError, match="reranker"):
        retriever.search("inverter", [0, 1, 2], 1)


@pytest.mark.parametrize("vectors", [[[0, 0], [1, 0]], [[float("nan"), 1], [1, 0]], [[1]], [1]])
def test_invalid_embeddings_fail_closed(vectors):
    backend = SimpleNamespace(identity={}, encode=lambda *a, **kw: vectors)
    with pytest.raises(ValueError, match="embedding"):
        Retriever(["a", "b"], mode="hybrid_bge", neural=backend)


def test_retrieval_metrics_match_hand_calculation_and_labels_never_enter_search():
    calls = []

    class Store:
        def __init__(self):
            self.documents = [SimpleNamespace(document_id=name) for name in ("a", "b", "c")]
            self.retriever = SimpleNamespace(config={"mode": "fixture"})
            self.source_sha256 = "fixture"

        def search(self, request):
            calls.append(request.model_dump())
            ids = ["b", "c"] if request.query == "first query" else ["a", "b"]
            return SimpleNamespace(
                evidence_id="fixture",
                result=SimpleNamespace(
                    hits=[
                        SimpleNamespace(document=SimpleNamespace(document_id=name)) for name in ids
                    ]
                ),
            )

    dataset = QASet(
        version="test",
        purpose="development_retrieval",
        annotation_status="test",
        questions=[
            {
                "question_id": "KB001",
                "question": "first query",
                "reference_answer": "private answer",
                "relevant_document_ids": ["a", "b"],
            },
            {
                "question_id": "KB002",
                "question": "second query",
                "reference_answer": "other answer",
                "relevant_document_ids": ["c"],
            },
        ],
    )
    metrics = evaluate(dataset, Store(), top_k=2)["metrics"]
    assert metrics == {
        "document_precision_at_k": 0.25,
        "document_recall_at_k": 0.25,
        "reciprocal_rank": 0.5,
    }
    assert calls == [
        {"query": query, "top_k": 2, "fault_type": None}
        for query in ("first query", "second query")
    ]
    dataset.questions[0].relevant_document_ids = ["absent"]
    with pytest.raises(ValueError, match="absent"):
        evaluate(dataset, Store())
    assert len(calls) == 2


def test_development_qa_has_thirty_sourced_questions_and_rejects_duplicates():
    dataset = QASet.model_validate_json((DEFAULT_KB.parent / "bench/kb_qa_dev.json").read_bytes())
    assert len(dataset.questions) == 30
    result = evaluate(dataset, KnowledgeStore())
    assert result["questions"] == 30
    assert len(result["qa_sha256"]) == 64
    invalid = dataset.model_dump()
    invalid["questions"].append(invalid["questions"][0])
    with pytest.raises(ValueError, match="duplicate"):
        QASet.model_validate(invalid)


def test_reference_dates_cannot_supply_a_fault_boundary(tmp_path):
    _, request, scope, evidence, report = context(tmp_path, "F3")
    reference = KnowledgeStore().playbook(PlaybookRequest(fault_type="F3"))
    item = reference.model_dump(mode="json")
    # Even an exactly matching ISO date in a reference's version is not an observation.
    item["result"]["playbook"]["document"]["source_version"] = report.findings[
        0
    ].window.start.value.isoformat()
    evidence[reference.evidence_id] = item
    report.findings[0].evidence_ids.append(reference.evidence_id)
    report.evidence_ids.append(reference.evidence_id)
    bound = report.findings[0].window.start
    bound.evidence_id = reference.evidence_id
    bound.path = "/result/playbook/document/source_version"
    assert "invalid_time_citation" in check_report(report, evidence, request, scope)
