"""Model-backed patch proposer over the shared structured LLM adapter and its call budget."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from pydantic import Field

from pvdiag.evolve.harness import REPO, Patch
from pvdiag.mcp_servers.models import Model

PROMPT = REPO / "harness" / "evolve_prompt.md"
MAX_FAILURES = 20


class PatchProposal(Model):
    patches: list[Patch] = Field(default_factory=list, max_length=4)


def failure_summary(index, failure, category):
    """Score flags and the model's own report; no device IDs, windows or evidence payloads."""
    case, run = failure["case"], failure["run"]
    report = run.get("report", {})
    return {
        "failure": f"failure_{index}",
        "category": category,
        "score": {
            key: case.get(key)
            for key in (
                "status",
                "predicted_fault",
                "root_top1",
                "device_correct",
                "joint_root_device_correct",
                "temporal_hits",
                "temporal_denominator",
                "normal_correct",
                "abstained",
                "numeric_claims",
                "valid_numeric_claims",
            )
        },
        "report": {
            "status": report.get("status"),
            "summary": report.get("summary"),
            "findings": [
                {k: f.get(k) for k in ("fault_type", "confidence", "rationale")}
                for f in report.get("findings", [])
            ],
        },
        # Run records carry issues as structured dicts; the proposer needs
        # only the deduplicated codes. (Real-run regression, 2026-10-31: this
        # set comprehension crashed on unhashable dicts because the LLM
        # proposer path had only ever seen string fixtures.)
        "verification_issues": sorted(
            {
                code
                for record in run.get("verification", [])
                for issue in record.get("issues", [])
                if (
                    code := issue.get("code")
                    if isinstance(issue, dict)
                    else str(issue)
                )
            }
        )[:20],
        "tool_calls": dict(
            sorted(Counter(t["name"] for t in run.get("trace", []) if t.get("success")).items())
        ),
        "failed_tool_calls": sum(not t.get("success") for t in run.get("trace", [])),
        "loops": run.get("loops"),
    }


class LLMPatchProposer:
    def __init__(self, reasoner, categories, prompt_path=PROMPT):
        if reasoner.metadata.get("mode") != "LLM":
            raise ValueError("the model proposer needs an explicit LLM reasoner")
        self.reasoner, self.categories = reasoner, categories
        self.prompt = Path(prompt_path).read_text()
        self.metadata = {
            "kind": "llm",
            "model": {k: v for k, v in reasoner.metadata.items() if k != "reported_models"},
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "usage": dict(reasoner.usage),
            "reported_models": reasoner.metadata["reported_models"],
            "dropped_over_limit": 0,
        }

    async def propose(self, harness, failures, limit, history):
        ordered = sorted(failures, key=lambda f: f["scenario_id"])[:MAX_FAILURES]
        payload = {
            "task": "propose_patches",
            "limit": limit,
            "editable_files": harness.files,
            "failures": [
                failure_summary(i, f, self.categories[f["scenario_id"]])
                for i, f in enumerate(ordered, 1)
            ],
            "omitted_failures": len(failures) - len(ordered),
            "history": history,
        }
        try:
            proposal = await self.reasoner._call(PatchProposal, payload, self.prompt)
        finally:
            self.metadata["usage"] = dict(self.reasoner.usage)
        proposal = PatchProposal.model_validate(
            proposal.model_dump(mode="json") if isinstance(proposal, PatchProposal) else proposal
        )
        self.metadata["dropped_over_limit"] += max(0, len(proposal.patches) - limit)
        return proposal.patches[:limit]
