"""Benchmark-backed evaluator, failure-driven probe scenes and a file-scripted proposer."""

import json

from pvdiag.eval.run_bench import run_benchmark
from pvdiag.evolve.harness import Patch
from pvdiag.sim.scenario_gen import generate_from_failures, write_pool


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
        # Core pool first; generated probe pools are registered per round.
        self.pools = [("core", manifest, scenario_root, review_bundle)]
        self.metadata = {
            "kind": "run_bench",
            "ablation": ablation,
            "harness_sensitive": harness_sensitive,
        }

    async def evaluate(self, harness, scenario_ids, label):
        root = harness.materialize(
            self.output / "harness", **({"kb_root": self.kb_root} if self.kb_root else {})
        )
        result, remaining = {}, set(scenario_ids)
        for name, manifest, scenario_root, review_bundle in self.pools:
            selected = sorted(remaining & {s["scenario_id"] for s in manifest["scenarios"]})
            if not selected:
                continue
            remaining -= set(selected)
            runs = self.output / "runs" / label / name
            summary = await run_benchmark(
                manifest,
                scenario_root,
                self.metadata_path,
                runs,
                self.reasoner_factory,
                verification_mode=self.verification_mode,
                knowledge_store=self.knowledge_factory(root / "kb")
                if self.knowledge_factory
                else None,
                review_bundle=review_bundle,
                faults_only=self.faults_only,
                ablation=self.ablation,
                memory_view=self.memory_view,
                harness_root=root,
                scenario_ids=selected,
            )
            for case in summary["cases"]:
                result[case["scenario_id"]] = {
                    "case": case,
                    "run": json.loads((runs / f"{case['scenario_id']}.json").read_text()),
                }
        if remaining:
            raise ValueError("requested scenarios belong to no registered pool")
        return result

    def add_pool(self, name, manifest, scenario_root):
        if any(name == pool[0] for pool in self.pools):
            raise ValueError("duplicate scenario pool")
        self.pools.append((name, manifest, scenario_root, None))


class GeneratedScenes:
    """Per-round probe scenes generated from the current core failures (dev pool only)."""

    def __init__(
        self,
        frame,
        manifest,
        evaluator,
        root,
        per_round,
        *,
        source_sha256,
        window_days=None,
        timezone=None,
    ):
        if source_sha256 != manifest.get("source_sha256"):
            raise ValueError("generation source differs from the parent manifest snapshot")
        if not 1 <= per_round <= 200:
            raise ValueError("generated probe scenes per round must be 1–200")
        self.frame, self.manifest, self.evaluator = frame, manifest, evaluator
        self.root, self.per_round, self.source_sha256 = root, per_round, source_sha256
        self.window_days, self.timezone = window_days, timezone

    async def extend(self, number, summary):
        scenes, details, report = generate_from_failures(
            self.frame,
            self.manifest,
            summary,
            self.per_round,
            window_days=self.window_days,
            timezone=self.timezone,
            label=f"round{number}",
        )
        folder = self.root / f"round{number}"
        pool = write_pool(
            self.frame,
            scenes,
            details,
            report,
            self.manifest,
            self.source_sha256,
            folder / "scenes",
        )
        (folder / "manifest.json").write_text(json.dumps(pool, indent=2, allow_nan=False) + "\n")
        self.evaluator.add_pool(f"generated-round{number}", pool, folder / "scenes")
        return pool["scenarios"], report


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
