import hashlib
import json
from pathlib import Path

import httpx
import pytest

from pvdiag.knowledge.model_assets import AssetFile, ModelAsset, prepare, verify_snapshot


@pytest.fixture
def model_registry(tmp_path):
    root = tmp_path / "registry"
    (root / "licenses").mkdir(parents=True)
    license_bytes = b"MIT license fixture; not actual weights"
    (root / "licenses/FlagEmbedding-LICENSE.txt").write_bytes(license_bytes)
    content = {
        "README.md": b"license: mit",
        "config.json": b"{}",
        "model.safetensors": b"mock weight bytes",
    }
    assets, responses = [], {}
    for role, name, revision in [
        ("embedding", "bge-small-zh-v1.5", "a" * 40),
        ("reranker", "bge-reranker-base", "b" * 40),
    ]:
        asset = {
            "key": role,
            "repo_id": f"BAAI/{name}",
            "revision": revision,
            "license": "MIT",
            "license_source_url": f"https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/{'c' * 40}/LICENSE",
            "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
            "files": [
                {
                    "path": filename,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for filename, data in content.items()
            ],
        }
        assets.append(asset)
        for filename, data in content.items():
            responses[f"https://huggingface.co/BAAI/{name}/resolve/{revision}/{filename}"] = data
    path = root / "model_assets.json"
    path.write_text(json.dumps({"version": "bge-assets-v1", "assets": assets}))
    return path, responses


def test_asset_downloads_are_verified_and_repeated_preparation_is_offline(model_registry, tmp_path):
    registry, responses = model_registry
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=responses[str(request.url)])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        paths = prepare(tmp_path / "models", registry_path=registry, client=client)
        assert len(calls) == 6
        assert prepare(tmp_path / "models", registry_path=registry, client=client) == paths
        assert len(calls) == 6
        for role, directory in paths.items():
            asset = verify_snapshot(directory, role, registry_path=registry)
            assert asset.key == role
            assert (Path(directory) / "LICENSE").exists()
        weights = Path(paths["embedding"]) / "model.safetensors"
        weights.write_bytes(b"existing user or other task content")
        with pytest.raises(ValueError, match="existing model file differs"):
            prepare(tmp_path / "models", registry_path=registry, client=client)
        assert weights.read_bytes() == b"existing user or other task content"


@pytest.mark.parametrize("failure", ["hash", "oversized", "network"])
def test_failed_download_never_publishes_bad_weights_or_leaves_temporary_files(
    model_registry, tmp_path, failure
):
    registry, responses = model_registry

    def handler(request):
        content = responses[str(request.url)]
        if request.url.path.endswith("model.safetensors"):
            if failure == "network":
                raise httpx.ReadTimeout("sensitive transport error", request=request)
            content = b"x" * (len(content) + (failure == "oversized"))
        return httpx.Response(200, content=content)

    root = tmp_path / "models"
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises((ValueError, RuntimeError)) as error,
    ):
        prepare(root, registry_path=registry, client=client)
    assert "sensitive transport error" not in str(error.value)
    assert not list(root.rglob("*.part"))
    assert not list(root.rglob("model.safetensors"))
    assert not list(root.rglob("snapshot.json"))
    # Completed files survive and are verified/skipped on the next attempt.
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=responses[str(r.url)]))
    ) as client:
        paths = prepare(root, registry_path=registry, client=client)
    assert verify_snapshot(paths["reranker"], "reranker", registry_path=registry)


@pytest.mark.parametrize("field", ["receipt", "license", "weights"])
def test_loading_rejects_changed_receipt_license_and_weights(model_registry, tmp_path, field):
    registry, responses = model_registry
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=responses[str(r.url)]))
    ) as client:
        paths = prepare(tmp_path / "models", registry_path=registry, client=client)
    root = Path(paths["embedding"])
    if field == "receipt":
        receipt = json.loads((root / "snapshot.json").read_text())
        receipt["revision"] = "d" * 40
        (root / "snapshot.json").write_text(json.dumps(receipt))
    elif field == "license":
        (root / "LICENSE").write_text("changed license")
    else:
        (root / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="match"):
        verify_snapshot(root, "embedding", registry_path=registry)


@pytest.mark.parametrize(
    "path", ["../outside.json", "/tmp/outside.json", "a/b/c.json", "weights.bin", "custom_code.py"]
)
def test_model_registry_refuses_escaped_paths_pickle_and_code(path):
    with pytest.raises(ValueError):
        AssetFile(path=path, size_bytes=1, sha256="a" * 64)


def test_registry_rejects_unreviewed_repository_and_symlink_escape(model_registry, tmp_path):
    registry, _ = model_registry
    payload = json.loads(registry.read_text())
    item = payload["assets"][0] | {"repo_id": "someone/other-model"}
    with pytest.raises(ValueError, match="reviewed"):
        ModelAsset.model_validate(item)
    directory = tmp_path / "models/bge-small-zh-v1.5"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    (directory / "README.md").symlink_to(outside)
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: pytest.fail("network must not be reached"))
        ) as client,
        pytest.raises(ValueError, match="escapes"),
    ):
        prepare(tmp_path / "models", registry_path=registry, client=client)
