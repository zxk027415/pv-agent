"""Editable harness snapshots, exact-replacement patches and isolated variant directories."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pydantic import Field, model_validator

from pvdiag.mcp_servers.models import Model
from pvdiag.observability.store import digest

REPO = Path(__file__).resolve().parents[3]
# The scorer, code, tool contracts and licensed excerpts are never patch targets.
EDITABLE = (
    "harness/system_prompt.md",
    "harness/verification_prompt.md",
    *(f"playbooks/F{i}.md" for i in range(1, 7)),
)


class PatchConflict(ValueError):
    pass


class Edit(Model):
    path: str
    old: str = Field(min_length=1, max_length=4000)
    new: str = Field(max_length=4000)

    @model_validator(mode="after")
    def editable(self):
        if self.path not in EDITABLE:
            raise ValueError("patches may only edit the system/verification prompts or playbooks")
        if self.old == self.new:
            raise ValueError("an edit must change its target text")
        return self


class Patch(Model):
    failure_mode: str = Field(min_length=1, max_length=300)
    rationale: str = Field(min_length=1, max_length=1000)
    edits: list[Edit] = Field(min_length=1, max_length=8)

    @property
    def identity(self):
        return "patch_" + digest(self.model_dump(mode="json"))[:24]


class Harness:
    def __init__(self, files):
        if set(files) != set(EDITABLE):
            raise ValueError("a harness snapshot contains exactly the editable files")
        self.files = dict(sorted(files.items()))
        self.sha256 = digest(self.files)

    @classmethod
    def from_root(cls, root=REPO):
        return cls({path: (Path(root) / path).read_text() for path in EDITABLE})

    def apply(self, patch):
        files = dict(self.files)
        for edit in patch.edits:
            if files[edit.path].count(edit.old) != 1:
                raise PatchConflict(f"{patch.identity}: target text is not unique in {edit.path}")
            files[edit.path] = files[edit.path].replace(edit.old, edit.new)
        return Harness(files)

    def changed(self, other):
        return [path for path in EDITABLE if self.files[path] != other.files[path]]

    def materialize(self, directory, kb_root=REPO / "kb"):
        """Write prompts, playbooks and a kb copy whose manifest binds the variant playbooks."""
        target = Path(directory) / self.sha256
        if (target / "harness.json").exists():
            if Harness.from_root(target).sha256 != self.sha256:
                raise ValueError("materialized harness was modified")
            return target
        staging = target.with_name(target.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        for path, text in self.files.items():
            (staging / path).parent.mkdir(parents=True, exist_ok=True)
            (staging / path).write_text(text)
        shutil.copytree(kb_root, staging / "kb", ignore=shutil.ignore_patterns("manifest.json"))
        manifest = json.loads((Path(kb_root) / "manifest.json").read_text())
        for document in manifest["documents"]:
            if document["kind"] == "project_playbook":
                content = hashlib.sha256((staging / document["path"]).read_bytes()).hexdigest()
                document["content_sha256"] = document["source_sha256"] = content
        (staging / "kb" / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        (staging / "harness.json").write_text(json.dumps({"sha256": self.sha256}) + "\n")
        staging.rename(target)
        return target
