"""Benchmark runner; passes public query scope, never truth, to the graph.

Development scenes are the default. Held-out and cross-site manifests can always be
preflighted (snapshots checked, nothing diagnosed); a scored run of them needs an
explicit flag and a stage that a one-shot ledger has not consumed before.
"""

import argparse
import asyncio
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from pvdiag.agent.__main__ import add_model_arguments, reasoner_from_args, run_diagnosis
from pvdiag.agent.ablation import resolve_ablation
from pvdiag.agent.state import DiagRequest
from pvdiag.eval.scorer import aggregate, score_case
from pvdiag.mcp_servers.pv_knowledge import add_knowledge_arguments, knowledge_from_args
from pvdiag.mcp_servers.store import ObservationStore
from pvdiag.memory.cli import memory_from_args
from pvdiag.observability.evaluation import EvaluationRun
from pvdiag.observability.store import TraceStore

SCOPES = {"dev": "dev", "heldout": "test", "cross_site": "cross_site"}


def consume_final_stage(ledger, manifest_sha256, scope, stage):
    """Reserve one scored evaluation of a held-out/cross-site manifest stage, before any call."""
    if not stage or not stage.replace("-", "").replace("_", "").isalnum():
        raise ValueError("final evaluations need a stage label such as round0 or final")
    entries = json.loads(ledger.read_text()) if ledger.exists() else []
    if any(e["manifest_sha256"] == manifest_sha256 and e["stage"] == stage for e in entries):
        raise ValueError(f"{scope} stage {stage!r} was already evaluated; it is not repeatable")
    entries.append(
        {
            "manifest_sha256": manifest_sha256,
            "scope": scope,
            "stage": stage,
            "reserved_at": datetime.now(UTC).isoformat(),
        }
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    staging = ledger.with_name(ledger.name + ".partial")
    staging.write_text(json.dumps(entries, indent=2) + "\n")
    staging.replace(ledger)


async def run_benchmark(
    manifest,
    scenario_root,
    metadata,
    output,
    reasoner_factory,
    *,
    verification_mode=None,
    knowledge_store=None,
    review_bundle=None,
    faults_only=False,
    ablation=None,
    trace_store=None,
    memory_view=None,
    harness_root=None,
    scenario_ids=None,
    scope="dev",
    preflight_only=False,
    allow_final_evaluation=False,
    final_ledger=None,
    stage=None,
):
    policy, verification_mode = resolve_ablation(ablation, verification_mode)
    if policy and (knowledge_store is not None) != (policy.level in {"A3", "A4"}):
        raise ValueError("A3/A4 require a knowledge snapshot")
    if (memory_view is not None) != (policy is not None and policy.level == "A4"):
        raise ValueError("only A4 requires and permits a frozen memory view")
    all_scenarios = manifest["scenarios"]
    if scope not in SCOPES:
        raise ValueError("scope must be dev, heldout or cross_site")
    if not all_scenarios or any(s["split"] != SCOPES[scope] for s in all_scenarios):
        if scope == "dev":
            raise ValueError(
                "M1 runner only permits development scenarios; held-out evaluations are not enabled"
            )
        raise ValueError(f"{scope} scope requires every scenario to have split={SCOPES[scope]}")
    final = scope != "dev" and not preflight_only
    if final and (not allow_final_evaluation or final_ledger is None):
        raise ValueError(
            f"scored {scope} evaluation needs --allow-final-evaluation and a one-shot ledger"
        )
    if scope != "dev" and (memory_view is not None or harness_root is not None):
        raise ValueError("held-out/cross-site runs use the frozen harness and no memory writes")
    if len({s["scenario_id"] for s in all_scenarios}) != len(all_scenarios):
        raise ValueError("duplicate scenario IDs")
    reviewed = manifest.get("review_catalog_sha256") is not None
    if reviewed:
        if review_bundle is None:
            raise ValueError("reviewed benchmarks require the original local review bundle")
        from pvdiag.eval.baseline_review import verify_release

        verify_release(manifest, review_bundle)
    scenarios = [s for s in all_scenarios if s["faults"]] if faults_only else all_scenarios
    if scenario_ids is not None:
        selected = set(scenario_ids)
        if len(selected) != len(scenario_ids) or not selected <= {
            s["scenario_id"] for s in scenarios
        }:
            raise ValueError("selected scenario IDs must be unique eligible manifest scenarios")
        scenarios = [s for s in scenarios if s["scenario_id"] in selected]
    if not scenarios:
        raise ValueError("no eligible development fault scenarios selected")
    if any(
        s.get("evaluation_eligible") is False
        or s.get("truth_scope") == "unmodified_baseline_candidate"
        or not s["faults"]
        and not reviewed
        for s in scenarios
    ):
        raise ValueError("unreviewed baseline candidates are not normal ground truth")
    # Freeze all observation/metadata snapshots before any potentially billable call.
    stores, requests = {}, {}
    metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    if reviewed and metadata_sha != manifest["metadata_sha256"]:
        raise ValueError("benchmark metadata differs from the reviewed snapshot")
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        if not scenario_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("invalid scenario ID")
        path = scenario_root / scenario_id / "timeseries.parquet"
        if not path.resolve().is_relative_to(scenario_root.resolve()):
            raise ValueError("benchmark observation escapes its configured root")
        store = ObservationStore(path, metadata)
        request = DiagRequest(
            system_id=scenario["system_id"],
            device_ids=scenario["device_ids"],
            start=scenario["query_window"][0],
            end=scenario["query_window"][1],
        )
        store.check(request)
        if reviewed:
            expected = hashlib.sha256()
            for content_digest in (
                scenario["observation_sha256"],
                metadata_sha,
                hashlib.sha256(b"").hexdigest(),
            ):
                expected.update(bytes.fromhex(content_digest))
            if store.source_sha256 != expected.hexdigest():
                raise ValueError(
                    "benchmark observations differ from the reviewed candidate snapshot"
                )
        stores[scenario_id] = store
        requests[scenario_id] = request
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if preflight_only:
        return {
            "scope": scope,
            "preflight_only": True,
            "manifest_sha256": manifest_sha,
            "scenarios": len(scenarios),
            "fault_scenarios": sum(bool(s["faults"]) for s in scenarios),
            "excluded_scenario_ids": [s["scenario_id"] for s in all_scenarios if s not in scenarios],
            "observation_snapshots": {key: store.source_sha256 for key, store in stores.items()},
            "evaluation_consumed": False,
        }
    if final:
        consume_final_stage(final_ledger, manifest_sha, scope, stage)
    evaluation = (
        EvaluationRun(
            trace_store,
            manifest,
            scenarios,
            requests,
            {key: store.source_sha256 for key, store in stores.items()},
            ablation=ablation,
            verification=verification_mode,
        )
        if trace_store
        else None
    )
    output.mkdir(parents=True, exist_ok=True)
    cases, metadata_models, summary_snapshots = [], [], {}
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        store = stores[scenario_id]
        # IDs, fault types, injection parameters and baseline statistics are not
        # inputs to the diagnosis. Only the public system/device/query scope is.
        request = requests[scenario_id]
        result = await run_diagnosis(
            store,
            request,
            reasoner_factory(),
            verification_mode=verification_mode,
            knowledge_store=knowledge_store,
            ablation=ablation,
            trace_store=trace_store,
            memory_view=memory_view,
            harness_root=harness_root,
        )
        (output / f"{scenario_id}.json").write_text(result.model_dump_json(indent=2) + "\n")
        cases.append(score_case(scenario, result))
        if evaluation:
            evaluation.attach(scenario_id, result, cases[-1])
        metadata_models.append(result.model)
        if result.ablation:
            summary_snapshots[scenario_id] = result.ablation["summary_sha256"]
        print(
            f"{scenario_id}: {result.report.status}, tools={len(result.trace)}, loops={result.loops}"
        )
    summary = {
        "evaluation_scope": scope,
        "final_stage": stage if final else None,
        "scope": (
            "development_reviewed_candidates" if reviewed else "development_smoke_injected_faults_only"
        )
        if scope == "dev"
        else f"{scope}_{'reviewed_candidates' if reviewed else 'injected_faults_only'}",
        "metrics": aggregate(cases),
        "cases": cases,
        "models": metadata_models,
        "runtime": {
            name: version(name) for name in ("langgraph", "langchain", "fastmcp", "pydantic")
        },
        "python": platform.python_version(),
        "verification_mode": verification_mode,
        "ablation": policy.metadata() if policy else None,
        "knowledge_source_sha256": knowledge_store.source_sha256 if knowledge_store else None,
        "memory": memory_view.snapshot.model_dump(mode="json")
        | {"sha256": memory_view.snapshot.sha256}
        if memory_view
        else None,
        "manifest_sha256": manifest_sha,
        "observation_snapshots": {key: store.source_sha256 for key, store in stores.items()},
        "summary_snapshots": summary_snapshots,
        "observability": evaluation.receipt() if evaluation else None,
        "excluded_scenario_ids": [s["scenario_id"] for s in all_scenarios if s not in scenarios],
        "by_fault_type": {
            kind: aggregate(
                [
                    case
                    for case, scene in zip(cases, scenarios, strict=True)
                    if ("+".join(sorted(f["type"] for f in scene["faults"])) or "N") == kind
                ]
            )
            for kind in sorted(
                {"+".join(sorted(f["type"] for f in scene["faults"])) or "N" for scene in scenarios}
            )
        },
        "limitations": [
            "Human baseline review is scoped observation review, not production health certification."
            if reviewed
            else "Screened baseline is not manually verified healthy.",
            "No held-out or LLM-as-Judge evaluation in this development run."
            if scope == "dev"
            else f"One-shot {scope} evaluation for stage {stage}; no LLM-as-Judge.",
            "Fault-only subsets cannot measure normal-scene false alarms."
            if not any(not s["faults"] for s in scenarios)
            else "Normal labels are limited to locally reviewed baseline scopes.",
        ]
        + (
            [
                "A4 recalls development-run memories; only same-system overlapping windows are"
                + " excluded, so A4 development scores are not held-out generalization."
            ]
            if memory_view
            else []
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("bench/m1_smoke.json"))
    parser.add_argument("--scenario-root", type=Path, default=Path("data/scenarios"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/raw/2107/metadata/2107_system_metadata.json")
    )
    parser.add_argument("--output", type=Path, default=Path("data/eval/m1_smoke"))
    add_model_arguments(parser)
    add_knowledge_arguments(parser)
    parser.add_argument("--review-bundle", type=Path)
    parser.add_argument("--scope", choices=sorted(SCOPES), default="dev")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="check and freeze all snapshots, diagnose nothing, consume nothing",
    )
    parser.add_argument(
        "--allow-final-evaluation",
        action="store_true",
        help="explicitly permit one scored held-out/cross-site run for --stage",
    )
    parser.add_argument("--final-ledger", type=Path, default=Path("data/eval/final_ledger.json"))
    parser.add_argument("--stage", help="final evaluation stage label, e.g. round0 or final")
    parser.add_argument(
        "--faults-only",
        action="store_true",
        help="explicitly exclude all unmodified candidates; do not report normal-scene accuracy",
    )
    args = parser.parse_args(argv)
    if args.mode == "llm" and not args.allow_model_calls:
        parser.error("LLM benchmark requires --allow-model-calls; no model request was sent")
    knowledge_store = knowledge_from_args(args)

    async def benchmark():
        async with memory_from_args(args) as memory_view:
            return await run_benchmark(
                json.loads(args.manifest.read_text()),
                args.scenario_root,
                args.metadata,
                args.output,
                lambda: reasoner_from_args(args),
                verification_mode=args.verification,
                knowledge_store=knowledge_store,
                review_bundle=args.review_bundle,
                faults_only=args.faults_only,
                ablation=args.ablation,
                trace_store=TraceStore(args.trace_db) if args.trace_db else None,
                memory_view=memory_view,
                scope=args.scope,
                preflight_only=args.preflight_only,
                allow_final_evaluation=args.allow_final_evaluation,
                final_ledger=args.final_ledger,
                stage=args.stage,
            )

    summary = asyncio.run(benchmark())
    if summary.get("preflight_only"):
        print(json.dumps({k: v for k, v in summary.items() if k != "observation_snapshots"}, indent=2))
        return 0
    print(json.dumps(summary["metrics"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
