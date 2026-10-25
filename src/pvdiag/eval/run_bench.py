"""Development-only benchmark runner; passes public query scope, never truth, to the graph."""

import argparse
import asyncio
import json
import platform
from importlib.metadata import version
from pathlib import Path

from pvdiag.agent.__main__ import add_model_arguments, reasoner_from_args, run_diagnosis
from pvdiag.agent.state import DiagRequest
from pvdiag.eval.scorer import aggregate, score_case
from pvdiag.mcp_servers.store import ObservationStore


async def run_benchmark(
    manifest, scenario_root, metadata, output, reasoner_factory, *, verification_mode="structural"
):
    scenarios = manifest["scenarios"]
    if not scenarios or any(s["split"] != "dev" for s in scenarios):
        raise ValueError(
            "M1 runner only permits development scenarios; held-out evaluations are not enabled"
        )
    if len({s["scenario_id"] for s in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario IDs")
    if any(
        s.get("evaluation_eligible") is False
        or s.get("truth_scope") == "unmodified_baseline_candidate"
        for s in scenarios
    ):
        raise ValueError("unreviewed baseline candidates are not normal ground truth")
    output.mkdir(parents=True, exist_ok=True)
    cases, metadata_models = [], []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        if not scenario_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("invalid scenario ID")
        store = ObservationStore(scenario_root / scenario_id / "timeseries.parquet", metadata)
        # IDs, fault types, injection parameters and baseline statistics are not
        # inputs to the diagnosis. Only the public system/device/query scope is.
        request = DiagRequest(
            system_id=scenario["system_id"],
            device_ids=scenario["device_ids"],
            start=scenario["query_window"][0],
            end=scenario["query_window"][1],
        )
        result = await run_diagnosis(
            store, request, reasoner_factory(), verification_mode=verification_mode
        )
        (output / f"{scenario_id}.json").write_text(result.model_dump_json(indent=2) + "\n")
        cases.append(score_case(scenario, result))
        metadata_models.append(result.model)
        print(
            f"{scenario_id}: {result.report.status}, tools={len(result.trace)}, loops={result.loops}"
        )
    summary = {
        "scope": "development_smoke_injected_faults_only",
        "metrics": aggregate(cases),
        "cases": cases,
        "models": metadata_models,
        "runtime": {
            name: version(name) for name in ("langgraph", "langchain", "fastmcp", "pydantic")
        },
        "python": platform.python_version(),
        "verification_mode": verification_mode,
        "limitations": [
            "Screened baseline is not manually verified healthy.",
            "No held-out, normal-scene, or LLM-as-Judge evaluation in this smoke set.",
        ],
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
    args = parser.parse_args(argv)
    if args.mode == "llm" and not args.allow_model_calls:
        parser.error("LLM benchmark requires --allow-model-calls; no model request was sent")
    summary = asyncio.run(
        run_benchmark(
            json.loads(args.manifest.read_text()),
            args.scenario_root,
            args.metadata,
            args.output,
            lambda: reasoner_from_args(args),
            verification_mode=args.verification,
        )
    )
    print(json.dumps(summary["metrics"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
