"""Post-diagnosis evaluation binding; expected outputs never enter model context."""

import hashlib
from pathlib import Path
from uuid import uuid4

from pvdiag.eval import scorer
from pvdiag.observability.store import digest


class EvaluationRun:
    def __init__(self, store, manifest, scenarios, requests, snapshots, *, ablation, verification):
        self.store = store
        identity = {
            "manifest_sha256": digest(manifest),
            "snapshots": snapshots,
            "selected_scenario_ids": [s["scenario_id"] for s in scenarios],
        }
        self.dataset_id = digest(identity)
        self.dataset_name = "pvdiag-dev-" + self.dataset_id[:24]
        store.put(
            "dataset",
            self.dataset_id,
            {
                "local_id": self.dataset_id,
                "name": self.dataset_name,
                "description": "Development evaluation; not held-out generalization results.",
                "metadata": identity | {"pvdiag_dataset_sha256": self.dataset_id},
            },
            immutable=True,
        )
        self.items = {}
        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            item_id = digest([self.dataset_id, scenario_id])
            self.items[scenario_id] = item_id
            store.put(
                "dataset_item",
                item_id,
                {
                    "id": item_id,
                    "dataset_local_id": self.dataset_id,
                    "datasetName": self.dataset_name,
                    "input": requests[scenario_id].model_dump(mode="json"),
                    "expectedOutput": {
                        "faults": scenario["faults"],
                        "is_normal": not scenario["faults"],
                        "truth_scope": scenario.get("truth_scope", "injected_development_faults"),
                    },
                    "metadata": {
                        "scenario_id": scenario_id,
                        "split": "dev",
                        "observation_source_sha256": snapshots[scenario_id],
                    },
                },
                immutable=True,
            )
        self.experiment_id = uuid4().hex
        self.experiment = {
            "id": self.experiment_id,
            "name": f"{ablation or 'unprofiled'}-{verification}-{self.experiment_id[:12]}",
            "dataset_local_id": self.dataset_id,
            "metadata": {
                "ablation": ablation,
                "verification": verification,
                "manifest_sha256": identity["manifest_sha256"],
                "scorer_version": scorer.SCORER_VERSION,
                "scorer_sha256": hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest(),
            },
        }
        store.put("experiment", self.experiment_id, self.experiment, immutable=True)

    def attach(self, scenario_id, result, case):
        if not result.observability:
            raise ValueError("evaluation requires an instrumented diagnosis")
        receipt = result.observability
        trace_id = receipt["trace_id"]
        root = self.store.get("span", trace_id + ":" + receipt["root_span_id"])
        item = self.store.get("dataset_item", self.items[scenario_id])
        if root is None or root["input"] != item["input"] or case["scenario_id"] != scenario_id:
            raise ValueError("evaluation_trace_input_mismatch")
        self.store.put(
            "evaluation",
            trace_id,
            {
                "trace_id": trace_id,
                "root_span_id": receipt["root_span_id"],
                "experiment_id": self.experiment_id,
                "item_id": item["id"],
                "case": case,
            },
            immutable=True,
        )

    def receipt(self):
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment["name"],
            "dataset_local_id": self.dataset_id,
        }
