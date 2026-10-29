"""§8 evaluate → propose → probe → compose → record, over development scenarios only."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from pvdiag.eval import scorer
from pvdiag.eval.outcome import case_passed
from pvdiag.evolve.harness import Harness, Patch, PatchConflict

POLICY_VERSION = "evolve-v1"
MAX_ROUNDS, MAX_PATCHES = 5, 4


def scorer_sha256():
    return hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest()


def category(scenario):
    return "+".join(sorted(f["type"] for f in scenario["faults"])) or "N"


CLOCK = re.compile(r"\d{4}-\d{2}-\d{2}|\b\d{1,2}:\d{2}\b")


def benchmark_identifiers(scenarios):
    values = set()
    for s in scenarios:
        values.add(s["scenario_id"])
        values.update(s.get("device_ids", []))
        if s.get("poa_device_id"):
            values.add(s["poa_device_id"])
        values.update(f["device_id"] for f in s["faults"] if f.get("device_id"))
    return values


def leaked(patch, identifiers):
    """Benchmark identifiers, dates or clock times newly written by a patch."""
    found = set()
    for edit in patch.edits:
        for value in identifiers:
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
            if re.search(pattern, edit.new) and not re.search(pattern, edit.old):
                found.add(value)
        found |= set(CLOCK.findall(edit.new)) - set(CLOCK.findall(edit.old))
    return sorted(found)


def max_diagnoses(scenarios, rounds, patches, generated=0):
    """Upper bound on diagnosis runs: round 0, then per round the generated probe scenes on the
    current harness, every patch probe (core + generated) and one composite over the core set."""
    return scenarios + rounds * (generated + patches * (scenarios + generated) + scenarios)


def summarize(evaluation, categories):
    passed = {sid: case_passed(item["case"]) for sid, item in evaluation.items()}
    groups = {}
    for sid, ok in passed.items():
        groups.setdefault(categories[sid], []).append(ok)
    cases = [item["case"] for item in evaluation.values()]
    return {
        "scenarios": len(passed),
        "passed": sum(passed.values()),
        "pass_rate": sum(passed.values()) / len(passed),
        "by_category": {key: sum(v) / len(v) for key, v in sorted(groups.items())},
        "failed_scenario_ids": sorted(sid for sid, ok in passed.items() if not ok),
        "known_input_tokens": sum(c["input_tokens"] for c in cases),
        "known_output_tokens": sum(c["output_tokens"] for c in cases),
        "usage_complete": not any(c["usage_missing_calls"] for c in cases),
    }


def regressed(candidate, base):
    return sorted(
        key
        for key, rate in base["by_category"].items()
        if candidate["by_category"].get(key, 0.0) < rate
    )


def restrict(evaluation, scenario_ids):
    return {sid: evaluation[sid] for sid in scenario_ids}


def write_json(path, value):
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    staging.replace(path)


async def run_evolution(
    manifest,
    evaluator,
    proposer,
    output,
    *,
    base=None,
    rounds=MAX_ROUNDS,
    patches_per_round=MAX_PATCHES,
    threshold=0.0,
    probe_sample=4,
    seed=0,
    faults_only=False,
    scene_source=None,
):
    scenarios = manifest.get("scenarios", [])
    if not scenarios or any(s["split"] != "dev" for s in scenarios):
        raise ValueError("self-evolution only reads development scenarios; held-out is reserved")
    if faults_only:
        scenarios = [s for s in scenarios if s["faults"]]
        if not scenarios:
            raise ValueError("no development fault scenarios selected")
    ids = [s["scenario_id"] for s in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate scenario IDs")
    if not 1 <= rounds <= MAX_ROUNDS or not 1 <= patches_per_round <= MAX_PATCHES:
        raise ValueError(f"at most {MAX_ROUNDS} rounds of {MAX_PATCHES} patches")
    if threshold < 0 or probe_sample < 0:
        raise ValueError("threshold and probe sample size cannot be negative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ledger.json").exists():
        raise ValueError("evolution output already exists; use a new directory")
    categories = {s["scenario_id"]: category(s) for s in scenarios}
    identifiers = benchmark_identifiers(scenarios)
    rng = random.Random(seed)
    frozen_scorer = scorer_sha256()
    current = base or Harness.from_root()

    async def evaluate(harness, scenario_ids, label):
        if scorer_sha256() != frozen_scorer:
            raise RuntimeError("the fixed scorer changed during self-evolution")
        result = await evaluator.evaluate(harness, sorted(scenario_ids), label)
        if set(result) != set(scenario_ids):
            raise ValueError("evaluator returned a different scenario set")
        return result

    current_eval = await evaluate(current, ids, "round0-base")
    current_summary = summarize(current_eval, categories)
    ledger = {
        "version": POLICY_VERSION,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "scorer_version": scorer.SCORER_VERSION,
        "scorer_sha256": frozen_scorer,
        "policy": {
            "rounds": rounds,
            "patches_per_round": patches_per_round,
            "threshold": threshold,
            "probe_sample": probe_sample,
            "seed": seed,
            "faults_only": faults_only,
            "generated_probe_scenes": getattr(scene_source, "per_round", 0),
        },
        "evaluator": getattr(evaluator, "metadata", {}),
        "proposer": getattr(proposer, "metadata", {}),
        "holdout": {
            "status": "not_enabled",
            "reason": "held-out manifests are refused; round 0 and final held-out runs remain separate",
        },
        "harness_versions": {current.sha256: {"round": 0, "parent": None, "patches": []}},
        "curve": [{"round": 0, "harness_sha256": current.sha256, "development": current_summary}],
        "rounds": [],
        "patches": {},
        "stopped": None,
    }
    write_json(output / "ledger.json", ledger)
    drafts = []

    for number in range(1, rounds + 1):
        failures = current_summary["failed_scenario_ids"]
        if not failures:
            ledger["stopped"] = "no_development_failures"
            break
        history = [
            {"patch_id": key, "failure_mode": value["failure_mode"], "outcome": value["outcome"]}
            for key, value in ledger["patches"].items()
        ]
        try:
            proposed = await proposer.propose(
                current,
                [
                    {
                        "scenario_id": sid,
                        "case": current_eval[sid]["case"],
                        "run": current_eval[sid]["run"],
                    }
                    for sid in failures
                ],
                patches_per_round,
                history,
            )
        except (ValueError, RuntimeError) as error:
            # Adapter errors are already redacted (budget, provider failure, bad output).
            ledger["stopped"] = {"reason": "proposer_failed", "round": number, "error": str(error)}
            break
        if len(proposed) > patches_per_round:
            raise ValueError("proposer exceeded the per-round patch budget")
        passing = sorted(set(ids) - set(failures))
        probe_ids = sorted(
            set(failures) | set(rng.sample(passing, min(probe_sample, len(passing))))
        )
        probe_base = restrict(current_eval, probe_ids)
        generated = {"scenario_ids": [], "report": None}
        if scene_source is not None:
            # Generated scenes only widen this round's probes; adoption stays on the fixed
            # core set, so the development curve remains comparable across rounds.
            try:
                extra, report = await scene_source.extend(
                    number,
                    {
                        "manifest_sha256": ledger["manifest_sha256"],
                        "cases": [current_eval[sid]["case"] for sid in ids],
                    },
                )
            except ValueError as error:
                generated["error"] = str(error)
            else:
                extra_ids = [scene["scenario_id"] for scene in extra]
                if (
                    any(scene["split"] != "dev" or not scene["faults"] for scene in extra)
                    or len(set(extra_ids)) != len(extra_ids)
                    or set(extra_ids) & set(categories)
                ):
                    raise ValueError("generated probe scenes must be new development fault scenes")
                categories.update({scene["scenario_id"]: category(scene) for scene in extra})
                identifiers |= benchmark_identifiers(extra)
                if extra_ids:
                    probe_base |= await evaluate(current, extra_ids, f"round{number}-generated")
                generated = {"scenario_ids": sorted(extra_ids), "report": report}
        probe_ids = sorted(probe_base)
        base_probe = summarize(probe_base, categories)
        record = {
            "round": number,
            "parent_harness_sha256": current.sha256,
            "probe_scenario_ids": probe_ids,
            "generated_probe_scenes": generated,
            "base_probe": base_probe,
            "patches": [],
            "composite": None,
            "adopted": False,
        }
        effective = []
        for raw in proposed:
            patch = Patch.model_validate(
                raw.model_dump(mode="json") if isinstance(raw, Patch) else raw
            )
            entry = patch.model_dump(mode="json") | {"round": number}
            if patch.identity in ledger["patches"]:
                entry["outcome"], entry["reason"] = "rejected", "duplicate_of_earlier_proposal"
                record["patches"].append({"patch_id": patch.identity, "outcome": "duplicate"})
                continue
            if leaks := leaked(patch, identifiers):
                entry |= {
                    "outcome": "rejected",
                    "reason": "references_benchmark_specifics",
                    "detail": leaks[:10],
                }
                ledger["patches"][patch.identity] = entry
                record["patches"].append({"patch_id": patch.identity, "outcome": "rejected"})
                continue
            try:
                candidate = current.apply(patch)
            except PatchConflict as error:
                entry |= {"outcome": "rejected", "reason": "does_not_apply", "detail": str(error)}
            else:
                probe = summarize(
                    await evaluate(candidate, probe_ids, f"round{number}-{patch.identity}"),
                    categories,
                )
                entry |= {"probe": probe, "regressed_categories": regressed(probe, base_probe)}
                if (
                    probe["pass_rate"] > base_probe["pass_rate"]
                    and not entry["regressed_categories"]
                ):
                    entry["outcome"] = "probe_effective"
                    effective.append((probe["pass_rate"], patch))
                else:
                    entry |= {"outcome": "rejected", "reason": "not_effective_on_probe"}
            ledger["patches"][patch.identity] = entry
            record["patches"].append({"patch_id": patch.identity, "outcome": entry["outcome"]})

        composite, merged = current, []
        for _, patch in sorted(effective, key=lambda item: (-item[0], item[1].identity)):
            try:
                composite = composite.apply(patch)
                merged.append(patch.identity)
            except PatchConflict:
                ledger["patches"][patch.identity] |= {
                    "outcome": "rejected",
                    "reason": "conflicts_with_higher_ranked_patch",
                }
        if merged:
            composite_eval = await evaluate(composite, ids, f"round{number}-composite")
            summary = summarize(composite_eval, categories)
            worse = regressed(summary, current_summary)
            adopted = summary["pass_rate"] > current_summary["pass_rate"] + threshold and not worse
            record["composite"] = {
                "harness_sha256": composite.sha256,
                "patch_ids": merged,
                "development": summary,
                "regressed_categories": worse,
            }
            for key in merged:
                ledger["patches"][key] |= (
                    {"outcome": "adopted"}
                    if adopted
                    else {"outcome": "rejected", "reason": "composite_not_strictly_better"}
                )
            if adopted:
                record["adopted"] = True
                ledger["harness_versions"][composite.sha256] = {
                    "round": number,
                    "parent": current.sha256,
                    "patches": merged,
                    "changed_files": composite.changed(current),
                }
                drafts.extend(
                    {
                        "patch_id": key,
                        "title": ledger["patches"][key]["failure_mode"][:160],
                        "text": ledger["patches"][key]["rationale"],
                        "failure_scenario_ids": failures,
                        "harness_sha256": composite.sha256,
                    }
                    for key in merged
                )
                current, current_eval, current_summary = composite, composite_eval, summary
        ledger["rounds"].append(record)
        ledger["curve"].append(
            {"round": number, "harness_sha256": current.sha256, "development": current_summary}
        )
        write_json(output / "ledger.json", ledger)
    else:
        ledger["stopped"] = "round_limit"
    ledger["final_harness_sha256"] = current.sha256
    ledger["final_harness_root"] = str(current.materialize(output / "harness"))
    write_json(output / "ledger.json", ledger)
    # Lessons still need the memory CLI's interactive review and stored support cases.
    write_json(output / "lesson_drafts.json", drafts)
    return ledger
