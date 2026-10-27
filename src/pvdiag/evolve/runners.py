"""Benchmark-backed evaluator and an offline, file-scripted patch proposer."""

import json

from pvdiag.eval.run_bench import run_benchmark
from pvdiag.evolve.harness import Patch


class BenchEvaluator:
    """Runs the fixed development benchmark against a materialized candidate harness."""

    def __init__(
        self,
        manifest,
        scenario_root,
        metadata,
        output,
        reasoner_factory,
        *,
        knowledge_factory=None,
        review_bundle=None,
        faults_only=False,
        ablation=None,
        verification_mode=None,
        memory_view=None,
        kb_root=None,
        harness_sensitive=True,
    ):
        self.kb_root = kb_root
        self.manifest, self.scenario_root, self.metadata_path = manifest, scenario_root, metadata
        self.output, self.reasoner_factory = output, reasoner_factory
        self.knowledge_factory, self.review_bundle = knowledge_factory, review_bundle
        self.faults_only, self.ablation = faults_only, ablation
        self.verification_mode, self.memory_view = verification_mode, memory_view
        self.metadata = {
            "kind": "run_bench",
            "ablation": ablation,
            "harness_sensitive": harness_sensitive,
        }

    async def evaluate(self, harness, scenario_ids, label):
        root = harness.materialize(self.output / "harness", **(
            {"kb_root": self.kb_root} if self.kb_root else {}
        ))
        runs = self.output / "runs" / label
        summary = await run_benchmark(
            self.manifest,
            self.scenario_root,
            self.metadata_path,
            runs,
            self.reasoner_factory,
            verification_mode=self.verification_mode,
            knowledge_store=self.knowledge_factory(root / "kb") if self.knowledge_factory else None,
            review_bundle=self.review_bundle,
            faults_only=self.faults_only,
            ablation=self.ablation,
            memory_view=self.memory_view,
            harness_root=root,
            scenario_ids=scenario_ids,
        )
        return {
            case["scenario_id"]: {
                "case": case,
                "run": json.loads((runs / f"{case['scenario_id']}.json").read_text()),
            }
            for case in summary["cases"]
        }


class FilePatchProposer:
    """Replays reviewed patch files per round; it never calls a model."""

    def __init__(self, path):
        raw = json.loads(path.read_text())
        self.rounds = [[Patch.model_validate(p) for p in batch] for batch in raw["rounds"]]
        self.calls = 0
        self.metadata = {"kind": "file", "rounds": len(self.rounds)}

    async def propose(self, harness, failures, limit, history):
        batch = self.rounds[self.calls] if self.calls < len(self.rounds) else []
        self.calls += 1
        return batch
