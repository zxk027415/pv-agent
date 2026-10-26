"""Real local tracing + MCP, offline model SDK and strict Langfuse HTTP contracts."""

import asyncio
import base64
import json
import subprocess
import sys

import httpx
import pytest
from test_ablation import inconclusive, offline_model
from test_agent import observation as observation  # noqa: PLC0414 -- pytest fixture
from test_agent import request
from test_baseline_review import review_fixture as review_fixture  # noqa: PLC0414 -- fixture

from pvdiag.agent.__main__ import run_diagnosis
from pvdiag.agent.llm import LLMReasoner
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.state import Plan, ToolCall
from pvdiag.eval.run_bench import run_benchmark
from pvdiag.mcp_servers.store import ObservationStore
from pvdiag.observability.__main__ import main
from pvdiag.observability.langfuse import LangfuseExporter, prepare
from pvdiag.observability.store import TraceStore, bounded, canonical
from pvdiag.observability.tracing import observe, trace_session


def otel_spans(prepared):
    return [
        s
        for p in prepared["otlp"]
        for r in p["resourceSpans"]
        for scope in r["scopeSpans"]
        for s in scope["spans"]
    ]


def attributes(span):
    return {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}


@pytest.fixture
def evaluated(review_fixture, monkeypatch):
    manifest, _, metadata, root, bundle = review_fixture
    store = TraceStore(bundle / "traces.sqlite3")
    attempts = offline_model(monkeypatch, inconclusive)
    summary = asyncio.run(
        run_benchmark(
            manifest,
            root,
            metadata,
            bundle / "eval",
            lambda: LLMReasoner("offline"),
            faults_only=True,
            ablation="A0",
            trace_store=store,
        )
    )
    return store, summary, attempts


def test_generation_tools_usage_timestamps_and_truth_stay_separate(evaluated):
    store, summary, attempts = evaluated
    snapshot = store.snapshot()
    local_spans = list(snapshot["span"].values())
    assert len(local_spans) == 10  # root + eight real MCP calls + one offline generation
    assert len({s["trace_id"] for s in local_spans}) == 1
    root = next(s for s in local_spans if s["parent_id"] is None)
    assert all(s["parent_id"] == root["span_id"] for s in local_spans if s is not root)
    assert all(
        root["start_ns"] <= s["start_ns"] <= s["end_ns"] <= root["end_ns"] for s in local_spans
    )
    assert all(s["success"] for s in local_spans)
    generation = next(s for s in local_spans if s["type"] == "generation")
    assert generation["metadata"]["usage"]["input_tokens"] == 7
    assert generation["metadata"]["reported_model"] == "offline-ablation-fixture"
    assert len([s for s in local_spans if s["metadata"].get("request_sent")]) == 9
    assert all(
        '"faults"' not in canonical(p) and "fixture-F4" not in canonical(p) for p in attempts
    )
    assert summary["observability"]["experiment_id"] in snapshot["experiment"]
    assert len(snapshot["dataset_item"]) == 1 and len(snapshot["evaluation"]) == 1
    prepared = prepare(snapshot)
    exported = otel_spans(prepared)
    roots = [s for s in exported if "parentSpanId" not in s]
    for span in exported:
        a = attributes(span)
        assert len(span["traceId"]) == 32 and len(span["spanId"]) == 16
        int(span["traceId"], 16)
        assert int(span["endTimeUnixNano"]) >= int(span["startTimeUnixNano"])
        assert a["langfuse.experiment.id"] == summary["observability"]["experiment_id"]
        assert a["langfuse.experiment.item.root_observation_id"] == roots[0]["spanId"]
        assert a["langfuse.trace.public"] is False
        assert ("langfuse.experiment.item.expected_output" in a) == (span in roots)
    assert len(prepared["scores"][0]) > 5
    assert all(s["observationId"] == roots[0]["spanId"] for s in prepared["scores"][0])
    a = attributes(next(s for s in exported if s["name"] == "llm.hypothesize"))
    assert json.loads(a["langfuse.observation.usage_details"]) == {"input": 7, "output": 3}
    assert "langfuse.observation.cost_details" not in a


def test_b0_all_tools_are_traced_with_no_generation(observation, tmp_path):
    store = TraceStore(tmp_path / "traces.sqlite3")
    result = asyncio.run(
        run_diagnosis(ObservationStore(*observation), request(), RuleReasoner(), trace_store=store)
    )
    spans = store.all("span")
    assert len(spans) == len(result.trace) + 1
    assert not any(s["type"] == "generation" for s in spans)
    assert result.observability["trace_id"] == spans[0]["trace_id"]
    assert result.model_usage["model_calls"] == 0


def test_rejected_tool_is_recorded_without_dispatch(observation, tmp_path, monkeypatch):
    def handler(payload, schema):
        if schema is Plan:
            return Plan(calls=[ToolCall(name="get_playbook", arguments={"fault_type": "F3"})])
        return inconclusive(payload, schema)

    offline_model(monkeypatch, handler)
    store = TraceStore(tmp_path / "traces.sqlite3")
    result = asyncio.run(
        run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline"),
            ablation="A1",
            trace_store=store,
        )
    )
    rejected = next(s for s in store.all("span") if s["name"] == "get_playbook")
    assert not rejected["success"] and not rejected["metadata"]["request_sent"]
    assert rejected["metadata"]["error_code"] == "ValueError"
    assert rejected["output"] is None and not result.trace[-1].success


def test_provider_failure_marks_unknown_usage_and_never_records_raw_exception(
    observation, tmp_path, monkeypatch
):
    def broken(payload, schema):
        raise ValueError("PRIVATE_PROVIDER_KEY_MUST_NOT_APPEAR")

    offline_model(monkeypatch, broken)
    store = TraceStore(tmp_path / "traces.sqlite3")
    result = asyncio.run(
        run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline"),
            ablation="A0",
            trace_store=store,
        )
    )
    assert result.model_usage["usage_missing_calls"] == 1
    prepared = prepare(store.snapshot())
    assert "PRIVATE_PROVIDER_KEY" not in canonical(store.snapshot()) + canonical(prepared)
    generation = next(s for s in otel_spans(prepared) if s["name"] == "llm.hypothesize")
    a = attributes(generation)
    assert generation["status"]["code"] == 2
    assert a["langfuse.observation.metadata.usage_missing"] == "true"
    assert "langfuse.observation.usage_details" not in a


def test_concurrent_diagnoses_have_independent_contexts(observation, tmp_path, monkeypatch):
    offline_model(monkeypatch, inconclusive)
    store = TraceStore(tmp_path / "traces.sqlite3")

    async def verify():
        return await asyncio.gather(
            *(
                run_diagnosis(
                    ObservationStore(*observation),
                    request(),
                    LLMReasoner("offline"),
                    ablation="A0",
                    trace_store=store,
                )
                for _ in range(2)
            )
        )

    results = asyncio.run(verify())
    assert len({r.observability["trace_id"] for r in results}) == 2
    spans = store.all("span")
    for result in results:
        subset = [s for s in spans if s["trace_id"] == result.observability["trace_id"]]
        assert len(subset) == 10
        assert all(s["parent_id"] in {None, result.observability["root_span_id"]} for s in subset)


def test_pending_observation_survives_process_exit_without_fabricated_usage(tmp_path):
    path = tmp_path / "crash.sqlite3"
    code = """
import os,sys
from pvdiag.observability.store import TraceStore
from pvdiag.observability.tracing import trace_session, observe, mark_request_sent
with trace_session(TraceStore(sys.argv[1]), 'crash', {'scope':'controlled'}, {}):
    with observe('generation','llm.controlled',{}, {'model':'offline','temperature':0,'max_output_tokens':4096}):
        mark_request_sent()
        os._exit(17)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, timeout=30, check=False
    )
    assert child.returncode == 17
    store = TraceStore(path)
    prepared = prepare(store.snapshot())
    generation = next(s for s in otel_spans(prepared) if s["name"] == "llm.controlled")
    a = attributes(generation)
    assert a["langfuse.observation.metadata.incomplete"] == "true"
    assert a["langfuse.observation.metadata.duration_unknown"] == "true"
    assert a["langfuse.observation.metadata.usage_missing"] == "true"
    assert generation["startTimeUnixNano"] == generation["endTimeUnixNano"]
    assert "langfuse.observation.usage_details" not in a


def test_sensitive_fields_large_data_and_invalid_json_are_bounded(tmp_path):
    safe = bounded(
        {
            "nested": {"DEEPSEEK_API_KEY": "PRIVATE", "headers": {"Authorization": "PRIVATE"}},
            "input_tokens": 7,
            "nan": float("nan"),
        }
    )
    assert "PRIVATE" not in canonical(safe) and safe["input_tokens"] == 7
    large = bounded("x" * 140_000)
    assert large["truncated"] and len(large["sha256"]) == 64
    store = TraceStore(tmp_path / "traces.sqlite3")
    with (
        trace_session(store, "one", {"request": True}, {}),
        observe("tool", "controlled", {}) as span,
    ):
        span.update(output=safe)
    with (
        pytest.raises(ValueError, match="identity_conflict"),
        trace_session(store, "one", {"changed": True}, {}),
    ):
        pytest.fail("same run cannot change its public request")


class LangfuseWire:
    """Contract fixture for documented OTLP/JSON, Dataset v2 and Scores endpoints."""

    def __init__(self):
        self.requests, self.datasets, self.items = [], {}, {}
        self.reject = None

    def __call__(self, request):
        assert (
            request.headers["authorization"]
            == "Basic " + base64.b64encode(b"public-fixture:secret-fixture").decode()
        )
        assert request.headers["x-langfuse-ingestion-version"] == "4"
        path = request.url.path
        payload = json.loads(request.content) if request.content else None
        self.requests.append((request.method, path, payload))
        if self.reject and self.reject[0] == path:
            return httpx.Response(self.reject[1], json=self.reject[2])
        if request.method == "GET":
            if path.startswith("/api/public/v2/datasets/"):
                value = self.datasets.get(path.rsplit("/", 1)[1])
            else:
                assert path.startswith("/api/public/dataset-items/")
                value = self.items.get(path.rsplit("/", 1)[1])
            return httpx.Response(200 if value else 404, json=value or {})
        assert request.headers["content-type"] == "application/json"
        if path == "/api/public/v2/datasets":
            assert set(payload) == {"name", "description", "metadata"}
            value = payload | {"id": "remote-dataset-" + str(len(self.datasets))}
            self.datasets[payload["name"]] = value
            return httpx.Response(200, json=value)
        if path == "/api/public/dataset-items":
            assert set(payload) == {"id", "datasetName", "input", "expectedOutput", "metadata"}
            value = payload | {"datasetId": self.datasets[payload["datasetName"]]["id"]}
            self.items[payload["id"]] = value
            return httpx.Response(200, json=value)
        if path == "/api/public/otel/v1/traces":
            assert len(payload["resourceSpans"]) == 1
            for span in payload["resourceSpans"][0]["scopeSpans"][0]["spans"]:
                a = attributes(span)
                assert a["langfuse.experiment.dataset.id"].startswith("remote-dataset-")
                assert a["langfuse.experiment.item.id"] in self.items
            return httpx.Response(200, json={})
        assert path == "/api/public/scores" and isinstance(payload, list)
        assert all(
            s["source"] == "API" and s["dataType"] in {"BOOLEAN", "NUMERIC"} for s in payload
        )
        return httpx.Response(202, json={"message": "Accepted"})


def exporter(store, wire):
    return LangfuseExporter(
        store,
        base_url="https://cloud.langfuse.com",
        public_key="public-fixture",
        secret_key="secret-fixture",
        transport=httpx.MockTransport(wire),
    )


def test_cloud_upload_is_opt_in_and_resumable_without_duplicate_posts(evaluated):
    store, _, _ = evaluated
    wire = LangfuseWire()

    async def verify():
        first = exporter(store, wire)
        try:
            with pytest.raises(ValueError, match="allow-upload"):
                await first.upload()
            assert wire.requests == []
            assert (await first.upload(allow_upload=True))["observations"] == 10
        finally:
            await first.close()
        posts = sum(method == "POST" for method, _, _ in wire.requests)
        second = exporter(TraceStore(store.path), wire)
        try:
            assert (await second.upload(allow_upload=True))["status"] == "accepted"
        finally:
            await second.close()
        assert sum(method == "POST" for method, _, _ in wire.requests) == posts

    asyncio.run(verify())
    assert "secret-fixture" not in canonical(store.snapshot())
    assert store.delivery_report() == {"complete": 4, "partial": []}
    with store.connection() as connection:
        assert "secret-fixture" not in str(
            connection.execute("SELECT * FROM deliveries").fetchall()
        )


@pytest.mark.parametrize(
    "failure", ["partial_spans", "partial_scores", "http_error", "invalid_json"]
)
def test_partial_or_failed_upload_is_not_reported_as_complete(evaluated, failure):
    store, _, _ = evaluated
    wire = LangfuseWire()
    if failure == "partial_spans":
        wire.reject = (
            "/api/public/otel/v1/traces",
            200,
            {"partialSuccess": {"rejectedSpans": "1", "errorMessage": "PRIVATE_REMOTE_ERROR"}},
        )
    elif failure == "partial_scores":
        wire.reject = (
            "/api/public/scores",
            207,
            {"accepted": 1, "rejected": 1, "errors": ["PRIVATE_REMOTE_ERROR"]},
        )
    elif failure == "http_error":
        wire.reject = ("/api/public/otel/v1/traces", 401, {"error": "PRIVATE_REMOTE_ERROR"})
    else:
        wire.reject = ("/api/public/otel/v1/traces", 200, ["PRIVATE_REMOTE_ERROR"])

    async def verify():
        client = exporter(store, wire)
        try:
            with pytest.raises(RuntimeError) as error:
                await client.upload(allow_upload=True)
            assert "PRIVATE_REMOTE_ERROR" not in str(error.value)
            if failure.startswith("partial"):
                assert len(store.delivery_report()["partial"]) == 1
                posts = sum(m == "POST" for m, _, _ in wire.requests)
                with pytest.raises(RuntimeError, match="reconciliation"):
                    await client.upload(allow_upload=True)
                assert sum(m == "POST" for m, _, _ in wire.requests) == posts
        finally:
            await client.close()

    asyncio.run(verify())


def test_dataset_conflict_stops_before_trace_upload(evaluated):
    store, _, _ = evaluated
    wire = LangfuseWire()
    dataset = store.all("dataset")[0]
    wire.datasets[dataset["name"]] = {
        "id": "wrong",
        "name": dataset["name"],
        "metadata": {"pvdiag_dataset_sha256": "wrong"},
    }

    async def verify():
        client = exporter(store, wire)
        try:
            with pytest.raises(RuntimeError, match="snapshot_conflict"):
                await client.upload(allow_upload=True)
        finally:
            await client.close()

    asyncio.run(verify())
    assert all(method == "GET" for method, _, _ in wire.requests)


@pytest.mark.parametrize(
    "url",
    [
        "http://cloud.langfuse.com",
        "https://user:pass@cloud.langfuse.com",
        "https://cloud.langfuse.com/?key=PRIVATE",
    ],
)
def test_credentials_cannot_be_placed_in_endpoint_urls(tmp_path, url):
    with pytest.raises(ValueError, match="HTTPS origin"):
        LangfuseExporter(
            TraceStore(tmp_path / "traces.sqlite3"),
            base_url=url,
            public_key="fixture",
            secret_key="fixture",
        )


def test_cli_prepare_is_network_free_and_preserves_reviewable_payload(
    evaluated, tmp_path, monkeypatch
):
    store, _, _ = evaluated

    def forbidden(*args, **kwargs):
        pytest.fail("preparing an artifact must not construct any HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    output = tmp_path / "export.json"
    assert main(["prepare", "--trace-db", str(store.path), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == prepare(store.snapshot())
    with pytest.raises(SystemExit) as error:
        main(["upload", "--trace-db", str(store.path)])
    assert error.value.code == 2


def test_changed_reviewed_snapshot_is_rejected_before_any_upload(evaluated):
    store, _, _ = evaluated
    reviewed = prepare(store.snapshot())["source_snapshot_sha256"]
    with trace_session(store, "additional-run", {"public": True}, {}):
        pass
    wire = LangfuseWire()

    async def verify():
        client = exporter(store, wire)
        try:
            with pytest.raises(ValueError, match="reviewed_snapshot_changed"):
                await client.upload(allow_upload=True, expected_snapshot_sha256=reviewed)
        finally:
            await client.close()

    asyncio.run(verify())
    assert wire.requests == []
