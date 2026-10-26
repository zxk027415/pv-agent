"""Langfuse v4 OTLP/JSON + Dataset v2 + Scores REST, with durable delivery receipts.

Contract checked 2026-10-26 against cloud.langfuse.com/generated/api/openapi.yml
and langfuse.com/integrations/native/opentelemetry/experiments.md.
No provider calls, credential discovery or remote writes happen during preparation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import quote, urlsplit

import httpx

from pvdiag.observability.store import canonical, digest


def attribute(key, value):
    if isinstance(value, bool):
        encoded = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    else:
        encoded = {"stringValue": value if isinstance(value, str) else canonical(value)}
    return {"key": key, "value": encoded}


def batches(values, *, max_bytes=4_000_000, max_items=100):
    batch, size = [], 0
    for value in values:
        length = len(canonical(value).encode())
        if length > max_bytes:
            raise ValueError("langfuse_object_exceeds_batch_budget")
        if batch and (size + length > max_bytes or len(batch) >= max_items):
            yield batch
            batch, size = [], 0
        batch.append(value)
        size += length
    if batch:
        yield batch


def prepare(snapshot, dataset_ids=None):
    """Pure export projection. Expected outputs are attached only to finished eval roots."""
    spans = snapshot.get("span", {})
    roots = {s["trace_id"]: s for s in spans.values() if s["parent_id"] is None}
    evaluations = snapshot.get("evaluation", {})
    items = snapshot.get("dataset_item", {})
    experiments = snapshot.get("experiment", {})
    dataset_ids = dataset_ids or {k: k for k in snapshot.get("dataset", {})}
    otel, scores = [], []
    for span in sorted(spans.values(), key=lambda s: (s["trace_id"], s["start_ns"], s["span_id"])):
        root = roots.get(span["trace_id"])
        if root is None:
            raise ValueError("orphan_trace_observation")
        attrs = {
            "langfuse.trace.name": "pvdiag.diagnosis",
            "langfuse.trace.public": False,
            "langfuse.observation.type": span["type"],
            "langfuse.observation.input": canonical(span["input"]),
            "langfuse.environment": "development",
        }
        attrs.update(
            {f"langfuse.trace.metadata.{k}": canonical(v) for k, v in root["metadata"].items()}
        )
        attrs.update(
            {
                f"langfuse.observation.metadata.{k}": canonical(v)
                for k, v in span["metadata"].items()
            }
        )
        if span["output"] is not None:
            attrs["langfuse.observation.output"] = canonical(span["output"])
        if span["type"] == "generation":
            meta = span["metadata"]
            attrs["langfuse.observation.model.name"] = meta.get("reported_model") or meta["model"]
            attrs["langfuse.observation.model.parameters"] = canonical(
                {
                    "temperature": meta["temperature"],
                    "max_tokens": meta["max_output_tokens"],
                }
            )
            if usage := meta.get("usage"):
                details = {"input": usage["input_tokens"], "output": usage["output_tokens"]}
                attrs["langfuse.observation.usage_details"] = canonical(details)
            else:
                attrs["langfuse.observation.metadata.usage_missing"] = canonical(
                    bool(meta.get("request_sent"))
                )
        evaluation = evaluations.get(span["trace_id"])
        if evaluation:
            item = items[evaluation["item_id"]]
            experiment = experiments[evaluation["experiment_id"]]
            attrs.update(
                {
                    "langfuse.environment": "experiment",
                    "langfuse.experiment.id": experiment["id"],
                    "langfuse.experiment.name": experiment["name"],
                    "langfuse.experiment.dataset.id": dataset_ids[experiment["dataset_local_id"]],
                    "langfuse.experiment.item.id": item["id"],
                    "langfuse.experiment.item.root_observation_id": root["span_id"],
                }
            )
            attrs.update(
                {
                    f"langfuse.experiment.metadata.{k}": canonical(v)
                    for k, v in experiment["metadata"].items()
                }
            )
            if span["parent_id"] is None:
                attrs["langfuse.experiment.item.expected_output"] = canonical(
                    item["expectedOutput"]
                )
        if span["end_ns"] is None:
            attrs["langfuse.observation.metadata.incomplete"] = "true"
            attrs["langfuse.observation.metadata.duration_unknown"] = "true"
        record = {
            "traceId": span["trace_id"],
            "spanId": span["span_id"],
            "name": span["name"],
            "kind": 1,
            "startTimeUnixNano": str(span["start_ns"]),
            "endTimeUnixNano": str(span["end_ns"] or span["start_ns"]),
            "attributes": [attribute(k, v) for k, v in sorted(attrs.items())],
            "status": {"code": 1 if span["success"] and span["end_ns"] else 2},
        }
        if span["parent_id"]:
            record["parentSpanId"] = span["parent_id"]
        otel.append(record)
    for trace_id, evaluation in sorted(evaluations.items()):
        root = roots[trace_id]
        timestamp = datetime.fromtimestamp(root["start_ns"] / 1e9, UTC).isoformat()
        for name, value in sorted(evaluation["case"].items()):
            if not isinstance(value, (bool, int, float)):
                continue
            scores.append(
                {
                    "id": digest([trace_id, name]),
                    "traceId": trace_id,
                    "observationId": root["span_id"],
                    "name": "pvdiag." + name,
                    "value": int(value) if isinstance(value, bool) else value,
                    "dataType": "BOOLEAN" if isinstance(value, bool) else "NUMERIC",
                    "source": "API",
                    "environment": "experiment",
                    "timestamp": timestamp,
                    "metadata": {"evaluator": "fixed-code-scorer", "not_llm_as_judge": True},
                }
            )
    return {
        "version": "langfuse-otlp-v4",
        "source_snapshot_sha256": digest(snapshot),
        "datasets": list(snapshot.get("dataset", {}).values()),
        "dataset_items": list(items.values()),
        "experiments": list(experiments.values()),
        "otlp": [
            {
                "resourceSpans": [
                    {
                        "resource": {"attributes": [attribute("service.name", "pvdiag")]},
                        "scopeSpans": [
                            {
                                "scope": {"name": "pvdiag", "version": "local-otlp-v1"},
                                "spans": batch,
                            }
                        ],
                    }
                ]
            }
            for batch in batches(otel)
        ],
        "scores": list(batches(scores)),
    }


class LangfuseExporter:
    def __init__(self, store, *, base_url, public_key, secret_key, transport=None):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("Langfuse requires an explicit HTTPS origin without URL credentials")
        if not public_key or not secret_key:
            raise ValueError(
                "configure Langfuse public and secret keys through local environment fields"
            )
        self.store = store
        self.base_url = base_url.rstrip("/")
        self.target = digest([self.base_url, public_key])
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=(public_key, secret_key),
            transport=transport,
            timeout=30,
            follow_redirects=False,
            headers={"x-langfuse-ingestion-version": "4"},
        )

    async def close(self):
        await self.client.aclose()

    async def request(self, method, path, payload=None, *, missing_ok=False):
        try:
            response = await self.client.request(method, path, json=payload)
        except httpx.HTTPError:
            raise RuntimeError("langfuse_transport_failed") from None
        if missing_ok and response.status_code == 404:
            return None
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"langfuse_http_status_{response.status_code}")
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise TypeError
        except (ValueError, TypeError):
            raise RuntimeError("langfuse_invalid_response") from None
        try:
            rejected = int(body.get("partialSuccess", {}).get("rejectedSpans", 0))
            if rejected < 0:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise RuntimeError("langfuse_invalid_response") from None
        if response.status_code == 207 or rejected:
            self.store.delivery(
                self.target,
                method + ":" + path,
                payload,
                state="partial",
                response={"status": response.status_code},
            )
            raise RuntimeError("langfuse_partial_acceptance_no_automatic_retry")
        return body

    async def deliver(self, path, payload, validate=None):
        operation = "POST:" + path
        previous = self.store.delivery(self.target, operation, payload)
        if previous:
            if previous[0] == "partial":
                raise RuntimeError("langfuse_partial_acceptance_requires_reconciliation")
            if previous[0] == "complete":
                return previous[1]
        response = await self.request("POST", path, payload)
        if validate:
            validate(response)
        if path == "/api/public/scores" and response.get("message") != "Accepted":
            raise RuntimeError("langfuse_score_acceptance_unknown")
        self.store.delivery(
            self.target,
            operation,
            payload,
            state="complete",
            response={key: response[key] for key in ("id", "name") if key in response},
        )
        return response

    async def upload(self, *, allow_upload=False, expected_snapshot_sha256=None):
        if not allow_upload:
            raise ValueError("remote writes require explicit --allow-upload")
        snapshot = self.store.snapshot()
        if expected_snapshot_sha256 is not None and digest(snapshot) != expected_snapshot_sha256:
            raise ValueError("langfuse_reviewed_snapshot_changed")
        prepared = prepare(snapshot)  # Validate the full outbox before the first remote write.
        dataset_ids = {}
        for dataset in prepared["datasets"]:
            path = "/api/public/v2/datasets/" + quote(dataset["name"], safe="")
            remote = await self.request("GET", path, missing_ok=True)
            payload = {k: dataset[k] for k in ("name", "description", "metadata")}

            def validate_dataset(remote, dataset=dataset):
                if (
                    not isinstance(remote.get("id"), str)
                    or not remote["id"]
                    or remote.get("name") != dataset["name"]
                ):
                    raise RuntimeError("langfuse_dataset_response_mismatch")
                if (
                    not isinstance(remote.get("metadata"), dict)
                    or remote["metadata"].get("pvdiag_dataset_sha256") != dataset["local_id"]
                ):
                    raise RuntimeError("langfuse_dataset_snapshot_conflict")

            if remote is None:
                if self.store.delivery(self.target, "POST:/api/public/v2/datasets", payload):
                    raise RuntimeError("langfuse_remote_dataset_missing_after_delivery")
                remote = await self.deliver("/api/public/v2/datasets", payload, validate_dataset)
            validate_dataset(remote)
            dataset_ids[dataset["local_id"]] = remote["id"]
        for item in prepared["dataset_items"]:
            remote = await self.request(
                "GET", "/api/public/dataset-items/" + item["id"], missing_ok=True
            )
            payload = {
                k: item[k] for k in ("id", "datasetName", "input", "expectedOutput", "metadata")
            }

            def validate_item(remote, payload=payload, item=item):
                if (
                    any(remote.get(k) != v for k, v in payload.items())
                    or remote.get("datasetId") != dataset_ids[item["dataset_local_id"]]
                ):
                    raise RuntimeError("langfuse_dataset_item_conflict")

            if remote is None:
                if self.store.delivery(self.target, "POST:/api/public/dataset-items", payload):
                    raise RuntimeError("langfuse_remote_item_missing_after_delivery")
                remote = await self.deliver("/api/public/dataset-items", payload, validate_item)
            validate_item(remote)
        prepared = prepare(snapshot, dataset_ids)
        for payload in prepared["otlp"]:
            await self.deliver("/api/public/otel/v1/traces", payload)
        for payload in prepared["scores"]:
            await self.deliver("/api/public/scores", payload)
        return {
            "status": "accepted",
            "datasets": len(dataset_ids),
            "traces": sum(s["parent_id"] is None for s in snapshot.get("span", {}).values()),
            "observations": len(snapshot.get("span", {})),
            "experiments": len(prepared["experiments"]),
        }
