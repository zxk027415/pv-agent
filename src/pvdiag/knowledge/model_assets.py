"""Explicit, hash-checked downloads of the two reviewed local BGE snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, model_validator

from pvdiag.mcp_servers.models import Model

DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "kb" / "model_assets.json"
REPOS = {
    "embedding": "BAAI/bge-small-zh-v1.5",
    "reranker": "BAAI/bge-reranker-base",
}


class AssetFile(Model):
    path: str = Field(pattern=r"^[a-zA-Z0-9_./-]+$")
    size_bytes: int = Field(gt=0, le=1_500_000_000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def safe_path(self):
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts or len(path.parts) > 2:
            raise ValueError("model file path must be a bounded relative path")
        allowed = {".json", ".md", ".txt", ".safetensors", ".model"}
        if path.suffix not in allowed:
            raise ValueError("unreviewed model file type; only safetensors weights are allowed")
        return self


class ModelAsset(Model):
    key: Literal["embedding", "reranker"]
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: Literal["MIT"]
    license_source_url: str = Field(
        pattern=r"^https://raw\.githubusercontent\.com/FlagOpen/FlagEmbedding/[0-9a-f]{40}/LICENSE$"
    )
    license_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[AssetFile] = Field(min_length=3, max_length=32)

    @model_validator(mode="after")
    def reviewed_repository(self):
        if self.repo_id != REPOS[self.key]:
            raise ValueError("model repository has not been reviewed")
        names = [file.path for file in self.files]
        if len(set(names)) != len(names):
            raise ValueError("duplicate model file paths")
        if not {"model.safetensors", "config.json", "README.md"} <= set(names):
            raise ValueError("model snapshot lacks weights, config or model card")
        return self


class AssetRegistry(Model):
    version: Literal["bge-assets-v1"]
    assets: list[ModelAsset] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def distinct_roles(self):
        if {asset.key for asset in self.assets} != set(REPOS):
            raise ValueError("both model roles must appear exactly once")
        return self


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_registry(path=DEFAULT_REGISTRY):
    registry = AssetRegistry.model_validate_json(Path(path).read_bytes())
    license_bytes = (Path(path).parent / "licenses" / "FlagEmbedding-LICENSE.txt").read_bytes()
    digest = hashlib.sha256(license_bytes).hexdigest()
    if any(asset.license_sha256 != digest for asset in registry.assets):
        raise ValueError("reviewed model license hash mismatch")
    return registry, license_bytes


def checked_path(root, relative):
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("model path escapes snapshot directory")
    return target


def verify_snapshot(root, role, *, registry_path=DEFAULT_REGISTRY):
    """Verify the model's pinned source, card, license, config, tokenizer and weights."""
    root = Path(root).resolve(strict=True)
    registry, license_bytes = read_registry(registry_path)
    asset = next(a for a in registry.assets if a.key == role)
    receipt = checked_path(root, "snapshot.json")
    if json.loads(receipt.read_text()) != asset.model_dump(mode="json"):
        raise ValueError("model receipt does not match the reviewed revision")
    license_path = checked_path(root, "LICENSE")
    if license_path.read_bytes() != license_bytes:
        raise ValueError("model snapshot license mismatch")
    for file in asset.files:
        target = checked_path(root, file.path)
        if target.stat().st_size != file.size_bytes or file_sha256(target) != file.sha256:
            raise ValueError("model snapshot file hash mismatch")
    return asset


def _write_new(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("existing model metadata differs; choose a new snapshot directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def prepare(root=Path("data/models"), *, registry_path=DEFAULT_REGISTRY, client=None):
    """Network access occurs only here, never during retrieval or model construction."""
    registry, license_bytes = read_registry(registry_path)
    if client is None:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, connect=20)) as session:
            return prepare(root, registry_path=registry_path, client=session)
    outputs = {}
    for asset in registry.assets:
        directory = Path(root) / asset.repo_id.split("/")[1]
        directory.mkdir(parents=True, exist_ok=True)
        for file in asset.files:
            target = checked_path(directory, file.path)
            if target.exists():
                if target.stat().st_size == file.size_bytes and file_sha256(target) == file.sha256:
                    continue
                raise ValueError("existing model file differs; choose a new snapshot directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, suffix=".part", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    digest, count = hashlib.sha256(), 0
                    url = f"https://huggingface.co/{asset.repo_id}/resolve/{asset.revision}/{file.path}"
                    print(
                        f"Downloading {asset.repo_id}/{file.path} ({file.size_bytes} bytes)",
                        flush=True,
                    )
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        for block in response.iter_bytes(chunk_size=1024 * 1024):
                            count += len(block)
                            if count > file.size_bytes:
                                raise ValueError("model download exceeds recorded size")
                            digest.update(block)
                            handle.write(block)
                    if count != file.size_bytes or digest.hexdigest() != file.sha256:
                        raise ValueError("downloaded model hash or size mismatch")
                # Linking creates the target only if absent; never overwrite another writer.
                os.link(temporary, target)
            except (httpx.HTTPError, OSError) as exc:
                raise RuntimeError(
                    f"model_asset_download_failed: {asset.key}/{file.path}: {type(exc).__name__}"
                ) from None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        _write_new(checked_path(directory, "LICENSE"), license_bytes)
        _write_new(
            checked_path(directory, "snapshot.json"),
            (json.dumps(asset.model_dump(mode="json"), sort_keys=True, indent=2) + "\n").encode(),
        )
        verify_snapshot(directory, asset.key, registry_path=registry_path)
        outputs[asset.key] = str(directory)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/models"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.root, registry_path=args.registry), indent=2))


if __name__ == "__main__":
    main()
