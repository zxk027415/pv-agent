"""Bind raw baseline review packets, local decisions and development benchmark labels.

Preparation reconstructs every candidate from source observations and injection
definitions. Human review sees pre-injection data with neutral content IDs. No
candidate is automatically certified normal, and no model is invoked here.
"""

import argparse
import getpass
import hashlib
import html
import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pvdiag.data.normalize import KEY
from pvdiag.sim.baseline import aligned_signals, screen_baseline
from pvdiag.sim.scenarios import (
    Scenario,
    materialize_scenario,
    scenario_window,
    screening_options,
    validate_split_isolation,
)

VERSION = "baseline-review-v1"
CHECKS = (
    "full_query_window",
    "missing_zero_and_night_data",
    "peer_comparison",
    "irradiance_context",
    "limitations_acknowledged",
)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def local_file(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("review or scenario artifact escapes its configured root")
    return path


def parquet_bytes(frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def values(series):
    return [float(v) if pd.notna(v) else None for v in series]


def packet_for(baseline, scope, source_sha256, metadata_sha256, screening=None):
    power, poa = aligned_signals(
        baseline,
        *scope["query_window"],
        tuple(scope["device_ids"]),
        scope["poa_device_id"],
    )
    daylight = poa > 200
    scales = power[daylight].quantile(0.95)
    normalized = power.div(scales.where(scales > 0), axis="columns")
    ratios = {}
    for device in scope["device_ids"]:
        peers = normalized.drop(columns=device)
        denominator = peers.median(axis=1).where(peers.notna().sum(axis=1) >= 2)
        ratios[device] = values(
            (normalized[device] / denominator.where(denominator > 0)).where(daylight)
        )
    raw = parquet_bytes(baseline)
    return {
        "version": VERSION,
        "scope": scope,
        "source_sha256": source_sha256,
        "metadata_sha256": metadata_sha256,
        "baseline_sha256": digest(raw),
        "screening": screen_baseline(
            baseline,
            scope["system_id"],
            *scope["query_window"],
            tuple(scope["device_ids"]),
            scope["poa_device_id"],
            **(screening or {}),
        ),
        "signals": {
            "utc": [t.isoformat() for t in power.index],
            "ac_power_kw": {device: values(power[device]) for device in scope["device_ids"]},
            "poa_wm2": values(poa),
            "empirical_peer_ratio": ratios,
        },
        "limitations": [
            "未注入不等于正常；自动筛查不能排除恒定损失、传感器漂移或自然故障。",
            "同站比值按本窗口经验 P95 归一化，不是额定容量或独立物理性能比。",
            "缺测保留为空；图为 UTC 十五分钟均值，原始记录另存 Parquet。",
            "人工接受仅表示当前范围内未观察到故障，不是生产健康认证。",
        ],
    }, raw


class ReviewLedger:
    """Trusted local control plane; identities are supplied only by the local CLI."""

    def __init__(self, path):
        self.path = path

    def initialize(self, catalog):
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    review_id TEXT PRIMARY KEY,
                    decision_json TEXT NOT NULL
                )
            """)
            connection.execute("CREATE TABLE catalog (sha256 TEXT NOT NULL)")
            connection.execute("INSERT INTO catalog VALUES (?)", (digest(canonical(catalog)),))
            connection.execute("CREATE TABLE packets (review_id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO packets VALUES (?)", [(k,) for k in catalog["packet_ids"]]
            )

    def validate_catalog(self, catalog):
        with sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT sha256 FROM catalog").fetchall()
        if rows != [(digest(canonical(catalog)),)]:
            raise ValueError("review catalog differs from its original ledger binding")

    def decisions(self):
        with sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            return {
                row[0]: json.loads(row[1]) for row in connection.execute("SELECT * FROM decisions")
            }

    def record(self, review_id, *, accepted, reviewer, reason, checks):
        if (
            type(accepted) is not bool
            or not reviewer.strip()
            or len(reviewer) > 80
            or any(ord(c) < 32 for c in reviewer)
            or not reason.strip()
            or len(reason) > 2000
            or not set(checks) <= set(CHECKS)
            or accepted
            and set(checks) != set(CHECKS)
        ):
            raise ValueError(
                "a named reviewer, reason and complete acceptance checklist are required"
            )
        content = {
            "review_id": review_id,
            "accepted": accepted,
            "reviewer": reviewer,
            "reason": reason,
            "checks": sorted(checks),
        }
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    "SELECT 1 FROM packets WHERE review_id=?", (review_id,)
                ).fetchone()
                is None
            ):
                raise ValueError("baseline packet is not registered in this review ledger")
            previous = connection.execute(
                "SELECT decision_json FROM decisions WHERE review_id=?",
                (review_id,),
            ).fetchone()
            if previous:
                recorded = json.loads(previous[0])
                if {k: recorded[k] for k in content} != content:
                    raise ValueError("a different baseline decision is already recorded")
                return recorded
            content["decided_at"] = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT INTO decisions VALUES (?,?)", (review_id, canonical(content).decode())
            )
        return content


def prepare(manifest, source, metadata, scenario_root, bundle):
    if bundle.exists():
        raise ValueError(
            "use a new bundle destination; existing review artifacts are not overwritten"
        )
    scenes = [Scenario.from_dict(s) for s in manifest["scenarios"]]
    screening = screening_options(manifest)
    validate_split_isolation(scenes)
    # Reviewers see only pre-injection baselines; held-out normals need the same review.
    # One split per bundle, and released held-out manifests stay refused by run_bench.
    if not scenes or len({scene.split for scene in scenes}) != 1 or scenes[0].split not in {
        "dev",
        "test",
    }:
        raise ValueError("a review bundle holds candidates from exactly one dev or test split")
    source_bytes, metadata_bytes = source.read_bytes(), metadata.read_bytes()
    source_sha, metadata_sha = digest(source_bytes), digest(metadata_bytes)
    if source_sha != manifest.get("source_sha256"):
        raise ValueError("candidate source fingerprint differs from the original source file")
    frame, site = pd.read_parquet(io.BytesIO(source_bytes)), json.loads(metadata_bytes)
    if {scene.system_id for scene in scenes} != {int(site["System"]["system_id"])}:
        raise ValueError("candidate systems differ from metadata")
    packets, by_scope, bindings = {}, {}, {}
    for scene, original in zip(scenes, manifest["scenarios"], strict=True):
        scope = {
            "system_id": scene.system_id,
            "query_window": list(scene.query_window),
            "device_ids": sorted(scene.device_ids),
            "poa_device_id": scene.poa_device_id,
        }
        scope_key = digest(canonical(scope))
        if scope_key not in by_scope:
            baseline = scenario_window(frame, scene)
            packet, raw = packet_for(baseline, scope, source_sha, metadata_sha, screening)
            if set(packet["screening"]["eligible_devices"]) != set(scene.device_ids):
                raise ValueError("original baseline no longer passes the frozen screening criteria")
            review_id = digest(canonical(packet))
            by_scope[scope_key] = (review_id, baseline)
            packets[review_id] = (packet, raw)
        review_id, baseline = by_scope[scope_key]
        expected, _ = materialize_scenario(baseline, scene, **screening)
        path = local_file(scenario_root, f"{scene.scenario_id}/timeseries.parquet")
        content = path.read_bytes()
        actual = pd.read_parquet(io.BytesIO(content))
        try:
            pd.testing.assert_frame_equal(
                actual.sort_values(KEY).reset_index(drop=True),
                expected.sort_values(KEY).reset_index(drop=True),
                check_exact=True,
            )
        except AssertionError:
            raise ValueError(
                f"candidate observations differ from reproducible injection: {scene.scenario_id}"
            ) from None
        if not scene.faults and (
            original.get("truth_scope") != "unmodified_baseline_candidate"
            or original.get("evaluation_eligible") is not False
        ):
            raise ValueError("unreviewed unmodified candidates must remain ineligible")
        bindings[scene.scenario_id] = {
            "review_id": review_id,
            "observation_sha256": digest(content),
        }
    catalog = {
        "version": VERSION,
        "source_sha256": source_sha,
        "metadata_sha256": metadata_sha,
        "candidate_manifest": manifest,
        "bindings": bindings,
        "packet_ids": sorted(packets),
    }
    # No artifacts or approvals are created until every candidate has been checked.
    (bundle / "packets").mkdir(parents=True)
    for review_id, (packet, raw) in packets.items():
        write_json(bundle / "packets" / f"{review_id}.json", packet)
        (bundle / "packets" / f"{review_id}.parquet").write_bytes(raw)
    write_json(bundle / "catalog.json", catalog)
    ReviewLedger(bundle / "reviews.sqlite3").initialize(catalog)
    return catalog


def load_bundle(bundle):
    catalog = read_json(bundle / "catalog.json")
    if catalog["version"] != VERSION:
        raise ValueError("unsupported baseline review version")
    ReviewLedger(bundle / "reviews.sqlite3").validate_catalog(catalog)
    packets = {}
    for review_id in catalog["packet_ids"]:
        if len(review_id) != 64 or any(c not in "0123456789abcdef" for c in review_id):
            raise ValueError("invalid baseline review ID")
        path = local_file(bundle, f"packets/{review_id}.json")
        packet = read_json(path)
        if digest(canonical(packet)) != review_id or (
            digest(local_file(bundle, f"packets/{review_id}.parquet").read_bytes())
            != packet["baseline_sha256"]
        ):
            raise ValueError("baseline review packet or raw snapshot was modified")
        if (packet["source_sha256"], packet["metadata_sha256"]) != (
            catalog["source_sha256"],
            catalog["metadata_sha256"],
        ):
            raise ValueError("baseline review provenance mismatch")
        packets[review_id] = packet
    return catalog, packets


def released_manifest(bundle):
    catalog, packets = load_bundle(bundle)
    decisions = ReviewLedger(bundle / "reviews.sqlite3").decisions()
    if any(not decisions.get(k, {}).get("accepted") for k in packets):
        raise ValueError("every original baseline needs an accepted human review before release")
    scenes = []
    for original in catalog["candidate_manifest"]["scenarios"]:
        binding = catalog["bindings"][original["scenario_id"]]
        packet = packets[binding["review_id"]]
        scene = (
            original
            | binding
            | {
                "evaluation_eligible": True,
                "truth_scope": "injected_faults_on_reviewed_baseline"
                if original["faults"]
                else "reviewed_baseline_no_observed_fault",
                "baseline_review": decisions[binding["review_id"]],
                "baseline_sha256": packet["baseline_sha256"],
            }
        )
        scenes.append(scene)
    return catalog["candidate_manifest"] | {
        "schema_version": 3,
        "purpose": "reviewed_development_candidates_not_final_benchmark",
        "review_catalog_sha256": digest(canonical(catalog)),
        "metadata_sha256": catalog["metadata_sha256"],
        "scenarios": scenes,
    }


def verify_release(manifest, bundle):
    expected = released_manifest(bundle)
    if canonical(manifest) != canonical(expected):
        raise ValueError(
            "reviewed manifest does not match the original candidates and local decisions"
        )


def render(bundle):
    """Render in the existing optional UI environment; no Parquet engine needed."""
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    catalog, packets = load_bundle(bundle)
    decisions = ReviewLedger(bundle / "reviews.sqlite3").decisions()
    (bundle / "plotly.min.js").write_text(get_plotlyjs())
    links = []
    for review_id, packet in packets.items():
        signals, scope = packet["signals"], packet["scope"]
        utc = [pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ") for t in signals["utc"]]
        figures = []
        for title, series in [
            ("逆变器功率（kW）", signals["ac_power_kw"]),
            ("POA 辐照度（W/m²）", {scope["poa_device_id"]: signals["poa_wm2"]}),
            ("经验 P95 归一化同站比值（仅日间）", signals["empirical_peer_ratio"]),
        ]:
            figure = go.Figure()
            for device, points in series.items():
                figure.add_trace(
                    go.Scatter(
                        x=utc,
                        y=points,
                        name=device,
                        mode="lines",
                        connectgaps=False,
                        hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>%{y}<extra>%{fullData.name}</extra>",
                    )
                )
            figure.update_layout(
                title=title,
                height=340,
                autosize=True,
                xaxis={"type": "date", "title": "UTC"},
                template="plotly_white",
                legend={"orientation": "h", "y": -0.3},
                margin={"l": 60, "r": 25, "t": 50, "b": 60},
            )
            # Script-safe JSON prevents device labels from terminating an inline script.
            figures.append(figure.to_json().replace("<", "\\u003c"))
        status = "未审核"
        if review_id in decisions:
            status = "已接受" if decisions[review_id]["accepted"] else "已拒绝"
        details = html.escape(json.dumps(scope, ensure_ascii=False, indent=2))
        warnings = "".join(f"<li>{html.escape(text)}</li>" for text in packet["limitations"])
        plots = "".join(
            f'<div id="p{i}"></div><script>const f{i}={figure};Plotly.newPlot("p{i}",f{i}.data,f{i}.layout,{{responsive:true,displaylogo:false}});</script>'
            for i, figure in enumerate(figures)
        )
        body = (
            f"<h1>原始基线审核 · {status}</h1><p>审核 ID：<code>{review_id}</code></p>"
            f"<pre>{details}</pre><ul>{warnings}</ul><p>以下全部为注入前源数据，不包含诊断预测或故障标签。</p>"
            f'<p><a href="packets/{review_id}.parquet">原始基线 Parquet</a> · '
            f'<a href="packets/{review_id}.json">完整审核数据与筛查统计</a></p>{plots}'
            "<h2>人工核对清单</h2><ol><li>核对完整查询窗口</li><li>检查缺测、零值及夜间记录</li>"
            "<li>核对同站差异与稳定损失</li><li>检查辐照变化、日间关系及传感器异常可能</li>"
            "<li>确认上述数据和方法限制，证据不足时不接受</li></ol>"
        )
        document = (
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>原始基线审核</title><script src="plotly.min.js"></script>'
            "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.6}"
            "pre,code{overflow-wrap:anywhere;white-space:pre-wrap}a{color:#2563eb}</style>"
            f"{body}</html>"
        )
        (bundle / f"{review_id}.html").write_text(document)
        links.append(
            f'<li><a href="{review_id}.html">{html.escape(str(scope["query_window"]))} · {html.escape(", ".join(scope["device_ids"]))}</a> · {status}</li>'
        )
    (bundle / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>开发基线审核目录</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.8}</style>"
        f"<h1>开发基线审核目录</h1><p>{len(packets)} 个独立原始范围，关联 {len(catalog['bindings'])} 个开发候选；人工审核前不会产生正常真值。</p><ul>{''.join(links)}</ul></html>"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "render", "review", "release", "status"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("bench/m2_dev_candidates.json"))
    parser.add_argument("--source", type=Path, default=Path("data/clean/2107.parquet"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/raw/2107/metadata/2107_system_metadata.json")
    )
    parser.add_argument("--scenario-root", type=Path, default=Path("data/m2_candidates"))
    parser.add_argument("--review-id")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        catalog = prepare(
            read_json(args.manifest), args.source, args.metadata, args.scenario_root, args.bundle
        )
        print(
            f"prepared {len(catalog['packet_ids'])} neutral baseline packets; no review decisions recorded"
        )
    elif args.command == "render":
        render(args.bundle)
        print(args.bundle / "index.html")
    elif args.command == "release":
        manifest = released_manifest(args.bundle)
        path = args.output or args.bundle / "reviewed_manifest.json"
        if path.exists():
            raise ValueError("use a new output path; existing manifests are not overwritten")
        write_json(path, manifest)
        print(f"{path}: {len(manifest['scenarios'])} reviewed development candidates")
    else:
        catalog, packets = load_bundle(args.bundle)
        ledger = ReviewLedger(args.bundle / "reviews.sqlite3")
        if args.command == "status":
            decisions = ledger.decisions()
            print(
                json.dumps(
                    {
                        "packets": len(packets),
                        "accepted": sum(d["accepted"] for d in decisions.values()),
                        "rejected": sum(not d["accepted"] for d in decisions.values()),
                        "pending": len(packets.keys() - decisions.keys()),
                    }
                )
            )
            return 0
        if not args.interactive or args.review_id not in packets:
            raise ValueError("review requires --interactive and an existing neutral review ID")
        print(json.dumps(packets[args.review_id]["scope"], indent=2, ensure_ascii=False))
        print(args.bundle / f"{args.review_id}.html")
        print(
            "请核对完整窗口、缺测/零值/夜间、同站关系、辐照与传感器上下文及限制；证据不足时拒绝。"
        )
        answer = input("输入 accept 接受该原始基线；reject 拒绝；其他输入不记录：").strip()
        if answer not in {"accept", "reject"}:
            print("no decision recorded")
            return 0
        reason = input("核对依据与限制（必填）：").strip()
        ledger.record(
            args.review_id,
            accepted=answer == "accept",
            reviewer=getpass.getuser(),
            reason=reason,
            checks=CHECKS if answer == "accept" else [],
        )
        print("baseline decision recorded")
    return 0


if __name__ == "__main__":
    main()
