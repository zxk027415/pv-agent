"""Download PVDAQ 2023 Solar Data Prize files from the public OEDI S3 bucket.

Anonymous HTTPS access, no AWS credentials needed.

Examples:
    uv run python -m pvdiag.data.download 2107 --list
    uv run python -m pvdiag.data.download 2107 --metadata-only
    uv run python -m pvdiag.data.download 2107
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx

BUCKET_URL = "https://oedi-data-lake.s3.amazonaws.com"
PREFIX_TMPL = "pvdaq/2023-solar-data-prize/{system_id}_OEDI/"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DEFAULT_DEST = Path("data/raw")


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int


def system_prefix(system_id: int) -> str:
    return PREFIX_TMPL.format(system_id=system_id)


def parse_list_page(xml_text: str) -> tuple[list[S3Object], str | None]:
    """Parse one ListObjectsV2 response page -> (objects, next continuation token)."""
    root = ET.fromstring(xml_text)
    objs: list[S3Object] = []
    for c in root.findall("s3:Contents", S3_NS):
        key = c.findtext("s3:Key", default="", namespaces=S3_NS)
        size = int(c.findtext("s3:Size", default="0", namespaces=S3_NS))
        if key and not key.endswith("/"):
            objs.append(S3Object(key, size))
    truncated = root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS) == "true"
    token = root.findtext("s3:NextContinuationToken", default=None, namespaces=S3_NS)
    return objs, (token if truncated else None)


def list_objects(client: httpx.Client, prefix: str) -> list[S3Object]:
    objs: list[S3Object] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = client.get(f"{BUCKET_URL}/", params=params)
        r.raise_for_status()
        page, token = parse_list_page(r.text)
        objs.extend(page)
        if not token:
            return objs


def local_path(obj: S3Object, prefix: str, dest_root: Path, system_id: int) -> Path:
    rel = obj.key[len(prefix):] if obj.key.startswith(prefix) else Path(obj.key).name
    return dest_root / str(system_id) / rel


def download_object(client: httpx.Client, obj: S3Object, path: Path) -> bool:
    """Download one object. Returns False if skipped (same size already on disk)."""
    if path.exists() and path.stat().st_size == obj.size:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with client.stream("GET", f"{BUCKET_URL}/{obj.key}") as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    if tmp.stat().st_size != obj.size:
        tmp.unlink()
        raise OSError(f"size mismatch for {obj.key}")
    tmp.replace(path)
    return True


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system_id", type=int)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--list", action="store_true", help="only list remote files")
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="download at most N data files")
    ap.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATTERN",
        help="only data files whose path contains PATTERN (case-insensitive, repeatable); metadata always kept",
    )
    args = ap.parse_args(argv)

    prefix = system_prefix(args.system_id)
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=True) as client:
        objs = list_objects(client, prefix)
        if not objs:
            print(f"no objects under {prefix}", file=sys.stderr)
            return 1
        if args.include:
            pats = [p.lower() for p in args.include]
            objs = [
                o
                for o in objs
                if "/metadata/" in o.key or any(p in o.key[len(prefix):].lower() for p in pats)
            ]
        if args.metadata_only:
            objs = [o for o in objs if "/metadata/" in o.key]
        elif args.limit is not None:
            meta = [o for o in objs if "/metadata/" in o.key]
            data = [o for o in objs if "/metadata/" not in o.key][: args.limit]
            objs = meta + data

        total = sum(o.size for o in objs)
        print(f"{len(objs)} files, {human(total)} under s3://oedi-data-lake/{prefix}")
        if args.list:
            for o in objs:
                print(f"{human(o.size):>10}  {o.key[len(prefix):]}")
            return 0

        done = skipped = 0
        for i, o in enumerate(objs, 1):
            path = local_path(o, prefix, args.dest, args.system_id)
            if download_object(client, o, path):
                done += 1
            else:
                skipped += 1
            print(f"[{i}/{len(objs)}] {path}", flush=True)
        print(f"downloaded {done}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
