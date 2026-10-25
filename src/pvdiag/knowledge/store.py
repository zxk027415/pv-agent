"""Manifest-only, hash-checked immutable knowledge snapshot; no data discovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pvdiag.knowledge.models import Manifest, Passage, PlaybookResult, SearchResult
from pvdiag.knowledge.retrieval import Retriever
from pvdiag.mcp_servers.models import Evidence

DEFAULT_KB = Path(__file__).resolve().parents[3] / "kb"


class KnowledgeStore:
    def __init__(self, root=DEFAULT_KB, *, mode="bm25", neural=None):
        self.root = Path(root).resolve(strict=True)
        raw = (self.root / "manifest.json").read_bytes()
        if len(raw) > 500_000:
            raise ValueError("knowledge manifest is too large")
        manifest = Manifest.model_validate_json(raw)
        self.documents, self.texts, self.playbooks = [], [], {}
        ids = set()
        for doc in manifest.documents:
            if doc.document_id in ids:
                raise ValueError("duplicate knowledge document ID")
            ids.add(doc.document_id)
            folder = "excerpts" if doc.kind == "upstream_excerpt" else "playbooks"
            text = self._read(doc.path, folder, doc.content_sha256).decode("utf-8")
            if not text.strip() or len(text) > 7000:
                raise ValueError("knowledge passage must have one to seven thousand characters")
            if doc.license_path:
                self._read(doc.license_path, "licenses", doc.license_sha256)
            if doc.kind == "project_playbook":
                for kind in doc.fault_types:
                    if kind in self.playbooks:
                        raise ValueError("duplicate fault playbook")
                    self.playbooks[kind] = len(self.documents)
            self.documents.append(doc)
            self.texts.append(text)
        indexed = [
            f"{doc.title}\n{' '.join(doc.fault_types)}\n{text}"
            for doc, text in zip(self.documents, self.texts)
        ]
        self.retriever = Retriever(indexed, mode=mode, neural=neural)
        # Document and license contents have already been checked against these hashes.
        identity = {"manifest": manifest.model_dump(), "retrieval": self.retriever.config}
        self.source_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def _read(self, relative, folder, expected):
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 2
            or path.parts[0] != folder
        ):
            raise ValueError("knowledge files must be directly inside their approved folder")
        base = self.root.parent if folder == "playbooks" else self.root
        resolved = (base / path).resolve(strict=True)
        if not resolved.is_relative_to(base / folder) or not resolved.is_file():
            raise ValueError("knowledge path escapes its approved folder")
        if resolved.stat().st_size > 50_000:
            raise ValueError("knowledge file is too large")
        content = resolved.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("knowledge file hash mismatch")
        return content

    def _passage(self, index, score=None):
        return Passage(document=self.documents[index], text=self.texts[index], score=score)

    def _evidence(self, tool, request, result):
        payload = {
            "tool": tool,
            "version": "kb-v1",
            "source_sha256": self.source_sha256,
            "request": request.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "warnings": [
                "Knowledge references describe mechanisms, not measurements of this plant."
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        return Evidence[type(result)](evidence_id=f"ev_{digest[:24]}", **payload)

    def search(self, request):
        eligible = [
            i
            for i, doc in enumerate(self.documents)
            if request.fault_type is None or request.fault_type in doc.fault_types
        ]
        ranked = self.retriever.search(request.query, eligible, request.top_k)
        result = SearchResult(
            retrieval_mode=self.retriever.mode,
            retrieval_config=self.retriever.config,
            hits=[self._passage(i, score) for i, score in ranked],
            candidate_count=len(eligible),
        )
        return self._evidence("search_fault_kb", request, result)

    def playbook(self, request):
        if request.fault_type not in self.playbooks:
            raise ValueError("no reviewed playbook for this fault type")
        result = PlaybookResult(playbook=self._passage(self.playbooks[request.fault_type]))
        return self._evidence("get_playbook", request, result)
