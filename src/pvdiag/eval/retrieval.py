"""Source-document retrieval metrics on an authored development QA set, not Ragas."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import Field, model_validator

from pvdiag.knowledge.models import SearchRequest
from pvdiag.knowledge.store import DEFAULT_KB
from pvdiag.mcp_servers.models import Model
from pvdiag.mcp_servers.pv_knowledge import add_knowledge_arguments, knowledge_from_args


class Question(Model):
    question_id: str = Field(pattern=r"^KB[0-9]{3}$")
    question: str = Field(min_length=2, max_length=500)
    reference_answer: str = Field(min_length=1, max_length=2000)
    relevant_document_ids: list[str] = Field(min_length=1, max_length=11)


class QASet(Model):
    version: str
    purpose: str
    annotation_status: str
    questions: list[Question] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def development_only(self):
        if self.purpose != "development_retrieval":
            raise ValueError("only development retrieval questions are enabled")
        if len({q.question_id for q in self.questions}) != len(self.questions):
            raise ValueError("duplicate question IDs")
        if any(
            len(set(q.relevant_document_ids)) != len(q.relevant_document_ids)
            for q in self.questions
        ):
            raise ValueError("duplicate relevant document IDs")
        return self


def evaluate(dataset, store, *, top_k=3):
    known = {d.document_id for d in store.documents}
    if any(set(q.relevant_document_ids) - known for q in dataset.questions):
        raise ValueError("QA labels reference documents absent from this corpus")
    cases = []
    for question in dataset.questions:
        # The answer and relevance labels never enter the retriever or model.
        evidence = store.search(SearchRequest(query=question.question, top_k=top_k))
        retrieved = [h.document.document_id for h in evidence.result.hits]
        relevant = set(question.relevant_document_ids)
        positions = [i for i, doc in enumerate(retrieved, 1) if doc in relevant]
        cases.append(
            {
                "question_id": question.question_id,
                "retrieved_document_ids": retrieved,
                "relevant_document_ids": sorted(relevant),
                "document_precision_at_k": len(positions) / top_k,
                "document_recall_at_k": len(positions) / len(relevant),
                "reciprocal_rank": 1 / positions[0] if positions else 0.0,
                "evidence_id": evidence.evidence_id,
            }
        )
    return {
        "purpose": dataset.purpose,
        "annotation_status": dataset.annotation_status,
        "qa_version": dataset.version,
        "qa_sha256": hashlib.sha256(dataset.model_dump_json().encode()).hexdigest(),
        "knowledge_source_sha256": store.source_sha256,
        "retrieval_config": store.retriever.config,
        "top_k": top_k,
        "questions": len(cases),
        "metrics": {
            key: sum(c[key] for c in cases) / len(cases)
            for key in ("document_precision_at_k", "document_recall_at_k", "reciprocal_rank")
        },
        "cases": cases,
        "limitations": [
            "Project-authored questions and document labels are not independent expert annotations.",
            "Document-ID overlap is not Ragas context precision/recall or answer faithfulness.",
            "No generated answers, LLM judge or fault-diagnosis accuracy are measured here.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_knowledge_arguments(parser)
    parser.set_defaults(knowledge_root=DEFAULT_KB)
    parser.add_argument("--questions", type=Path, default=Path("bench/kb_qa_dev.json"))
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--output", type=Path, default=Path("data/eval/kb_bm25_dev.json"))
    args = parser.parse_args(argv)
    dataset = QASet.model_validate_json(args.questions.read_bytes())
    result = evaluate(dataset, knowledge_from_args(args), top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"questions": result["questions"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
