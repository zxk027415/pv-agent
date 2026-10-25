"""Rebuild only the reviewed pvlib excerpts and existing project playbooks offline."""

import argparse
import hashlib
import inspect
import json
from importlib.metadata import distribution
from pathlib import Path

from pvlib import inverter, pvsystem, shading, soiling, temperature

from pvdiag.knowledge.models import Document, Manifest
from pvdiag.knowledge.store import DEFAULT_KB

LICENSE_SHA256 = "a02e12ddaada3cf0d5dbdd8affdd577c2eec640758a95842a304f7513b8c0be4"
EXCERPTS = [
    ("pvlib-soiling-hsu", "HSU soiling ratio and rain cleaning inputs", soiling.hsu, ["F1"]),
    (
        "pvlib-shading-masking",
        "Row shading and average masking angle",
        shading.masking_angle_passias,
        ["F2"],
    ),
    (
        "pvlib-pvwatts-dc",
        "PVWatts DC power, effective irradiance and temperature inputs",
        pvsystem.pvwatts_dc,
        ["F1", "F4", "F5"],
    ),
    (
        "pvlib-pvwatts-ac",
        "PVWatts inverter efficiency and output clipping",
        inverter.pvwatts,
        ["F3", "F4", "F5"],
    ),
    (
        "pvlib-faiman",
        "Faiman module temperature and empirical heat-loss assumptions",
        temperature.faiman,
        ["F5"],
    ),
]
TITLES = {
    "F1": "积灰与清洗恢复 / Soiling",
    "F2": "局部遮挡 / Shading",
    "F3": "逆变器停机 / Inverter outage",
    "F4": "组串失效 / String loss",
    "F5": "辐照传感器漂移 / Irradiance sensor drift",
    "F6": "通讯中断 / Communication gap",
}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def build(root=DEFAULT_KB):
    root = Path(root)
    package = distribution("pvlib")
    if package.version != "0.16.1":
        raise ValueError("only pvlib 0.16.1 has been reviewed for this corpus")
    license_bytes = package.locate_file("pvlib-0.16.1.dist-info/licenses/LICENSE").read_bytes()
    if sha(license_bytes) != LICENSE_SHA256:
        raise ValueError("pvlib license differs from the reviewed snapshot")
    docs, contents = [], {}
    for identifier, title, function, faults in EXCERPTS:
        module = Path(inspect.getfile(function))
        source = module.read_bytes()
        # Full function docstrings only: references are bibliography, not copied articles.
        content = (inspect.getdoc(function) + "\n").encode()
        if len(content.decode()) > 7000:
            raise ValueError("reviewed docstring exceeds passage budget")
        relative = f"excerpts/{identifier}.txt"
        contents[relative] = content
        docs.append(
            Document(
                document_id=identifier,
                title=title,
                kind="upstream_excerpt",
                fault_types=faults,
                path=relative,
                content_sha256=sha(content),
                source_url=f"https://github.com/pvlib/pvlib-python/blob/v0.16.1/pvlib/{module.name}#L{inspect.getsourcelines(function)[1]}",
                source_version="pvlib==0.16.1",
                source_sha256=sha(source),
                license="BSD-3-Clause",
                license_path="licenses/pvlib-LICENSE.txt",
                license_sha256=LICENSE_SHA256,
                attribution="pvlib python Contributors; PVLIB python Development Team; Sandia National Laboratories",
            )
        )
    for fault, title in TITLES.items():
        relative = f"playbooks/{fault}.md"
        content = (root.parent / relative).read_bytes()
        docs.append(
            Document(
                document_id=f"playbook-{fault.lower()}",
                title=title,
                kind="project_playbook",
                fault_types=[fault],
                path=relative,
                content_sha256=sha(content),
                source_url=f"project:{relative}",
                source_version="playbook-v1",
                source_sha256=sha(content),
                license="project-authored",
                attribution="PV Diagnosis Agent project-authored diagnostic procedure",
            )
        )
    manifest = Manifest(version="kb-v1", documents=docs)
    # All input checks finish before writing; unrelated files are never scanned or removed.
    contents["licenses/pvlib-LICENSE.txt"] = license_bytes
    contents["manifest.json"] = (
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False) + "\n"
    ).encode()
    for relative, content in contents.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_KB)
    args = parser.parse_args(argv)
    result = build(args.root)
    print(f"{args.root / 'manifest.json'}: {len(result.documents)} reviewed passages")


if __name__ == "__main__":
    main()
