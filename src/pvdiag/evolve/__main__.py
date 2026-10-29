"""Offline-first self-evolution CLI; model-backed evaluation needs an explicit total call cap."""

import argparse
import asyncio
import hashlib
import json
from argparse import Namespace
from pathlib import Path

from pvdiag.agent.__main__ import add_model_arguments, reasoner_from_args
from pvdiag.evolve.loop import MAX_PATCHES, MAX_ROUNDS, category, max_diagnoses, run_evolution
from pvdiag.evolve.proposer import LLMPatchProposer
from pvdiag.evolve.runners import BenchEvaluator, FilePatchProposer, GeneratedScenes
from pvdiag.knowledge.store import KnowledgeStore
from pvdiag.mcp_servers.pv_knowledge import add_knowledge_arguments, knowledge_from_args
from pvdiag.memory.cli import memory_from_args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("bench/m1_smoke.json"))
    parser.add_argument("--scenario-root", type=Path, default=Path("data/scenarios"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/raw/2107/metadata/2107_system_metadata.json")
    )
    parser.add_argument("--output", type=Path, required=True, help="new evolution directory")
    proposer = parser.add_mutually_exclusive_group(required=True)
    proposer.add_argument(
        "--proposer-file", type=Path, help="reviewed patches per round (offline, no model)"
    )
    proposer.add_argument("--proposer-model", help="model that proposes harness patches")
    parser.add_argument("--proposer-base-url", default="https://api.deepseek.com")
    parser.add_argument("--proposer-credential-env")
    parser.add_argument(
        "--max-proposer-calls", type=int, help="proposer call budget; default and maximum: rounds"
    )
    parser.add_argument(
        "--allow-proposer-calls",
        action="store_true",
        help="explicitly permit potentially billable patch-proposal requests",
    )
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--patches", type=int, default=MAX_PATCHES)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--probe-sample", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--review-bundle", type=Path)
    parser.add_argument("--faults-only", action="store_true")
    parser.add_argument(
        "--generate-scenes",
        type=int,
        default=0,
        help="per-round failure-driven probe scenes (development pool only; 0 disables)",
    )
    parser.add_argument("--source", type=Path, default=Path("data/clean/2107.parquet"))
    parser.add_argument("--scene-window-days", type=int)
    parser.add_argument("--site-timezone", help="only when the manifest omits it")
    parser.add_argument(
        "--max-total-model-calls",
        type=int,
        help="LLM mode: refuse to start unless the worst-case call bound fits this cap",
    )
    add_model_arguments(parser)
    add_knowledge_arguments(parser)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    scenarios = [s for s in manifest["scenarios"] if s["faults"] or not args.faults_only]
    if args.proposer_model:
        calls = args.max_proposer_calls or args.rounds
        if not args.allow_proposer_calls:
            parser.error("the model proposer requires --allow-proposer-calls; nothing was sent")
        if not 1 <= calls <= args.rounds:
            parser.error("--max-proposer-calls must be between 1 and --rounds")
        try:
            patch_proposer = LLMPatchProposer(
                reasoner_from_args(
                    Namespace(
                        mode="llm",
                        allow_model_calls=True,
                        model=args.proposer_model,
                        base_url=args.proposer_base_url,
                        credential_env=args.proposer_credential_env,
                        max_model_calls=calls,
                    )
                ),
                {s["scenario_id"]: category(s) for s in scenarios},
            )
        except ValueError as error:
            parser.error(f"{error}; no model request was sent")
    else:
        if args.max_proposer_calls or args.allow_proposer_calls:
            parser.error("proposer call options only apply to --proposer-model")
        patch_proposer = FilePatchProposer(args.proposer_file)
    if args.generate_scenes < 0:
        parser.error("--generate-scenes cannot be negative")
    frame = source_sha = None
    if args.generate_scenes:
        import pandas as pd

        source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
        if source_sha != manifest.get("source_sha256"):
            parser.error("--source differs from the manifest's source snapshot")
        frame = pd.read_parquet(args.source)
    bound = max_diagnoses(len(scenarios), args.rounds, args.patches, args.generate_scenes)
    if args.mode == "llm":
        calls = bound * args.max_model_calls
        if args.max_total_model_calls is None or calls > args.max_total_model_calls:
            parser.error(
                f"worst case is {bound} diagnoses × {args.max_model_calls} = {calls} model calls; "
                "set --max-total-model-calls to at least that; no model request was sent"
            )
    else:
        print("B0 ignores prompts and playbooks: patches cannot change scores (plumbing run only)")
    # Validates the base kb once and reuses its loaded local models for every variant kb.
    base_knowledge = knowledge_from_args(args)
    neural = base_knowledge.retriever.neural if base_knowledge else None

    async def evolve():
        async with memory_from_args(args) as memory_view:
            evaluator = BenchEvaluator(
                manifest,
                args.scenario_root,
                args.metadata,
                args.output,
                lambda: reasoner_from_args(args),
                knowledge_factory=(
                    lambda root: KnowledgeStore(root, mode=args.retrieval_mode, neural=neural)
                )
                if base_knowledge
                else None,
                kb_root=args.knowledge_root,
                review_bundle=args.review_bundle,
                faults_only=args.faults_only,
                ablation=args.ablation,
                verification_mode=args.verification,
                memory_view=memory_view,
                harness_sensitive=args.mode == "llm",
            )
            evaluator.metadata["max_diagnoses"] = bound
            scenes = (
                GeneratedScenes(
                    frame,
                    manifest,
                    evaluator,
                    args.output / "generated",
                    args.generate_scenes,
                    source_sha256=source_sha,
                    window_days=args.scene_window_days,
                    timezone=args.site_timezone,
                )
                if args.generate_scenes
                else None
            )
            return await run_evolution(
                manifest,
                evaluator,
                patch_proposer,
                args.output,
                rounds=args.rounds,
                patches_per_round=args.patches,
                threshold=args.threshold,
                probe_sample=args.probe_sample,
                seed=args.seed,
                faults_only=args.faults_only,
                scene_source=scenes,
            )

    try:
        ledger = asyncio.run(evolve())
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps([point["development"]["pass_rate"] for point in ledger["curve"]]))
    print(f"{args.output / 'ledger.json'}: final harness {ledger['final_harness_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
