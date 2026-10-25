"""Per-browser run state and evidence views, without model or database imports."""

import copy
import hashlib
import json
from uuid import uuid4

import pandas as pd

from pvdiag.ui.client import ApiError, Event, run_path

STATUS = {
    "running": "诊断进行中",
    "completed": "诊断结束",
    "awaiting_approval": "等待人工审批",
    "interrupted": "运行已中断，可恢复",
    "fault": "发现故障",
    "normal": "未发现故障",
    "inconclusive": "证据不足",
    "needs_review": "需要复核",
}


def empty_session():
    return {"run_id": "", "cursor": None, "progress": [], "reply": None, "request": None}


def request_body(profile, device_ids, start, end, question):
    if not profile or not device_ids or not question.strip():
        raise ApiError("请选择数据配置、至少一台设备并填写诊断问题。")
    try:
        bounds = [pd.Timestamp(value) for value in (start, end)]
        if any(pd.isna(t) or t.tzinfo is None for t in bounds):
            raise ValueError
        bounds = [t.tz_convert("UTC") for t in bounds]
        if not pd.Timedelta(0) < bounds[1] - bounds[0] <= pd.Timedelta(days=90):
            raise ValueError
        if any(t.minute % 15 or t.second or t.microsecond or t.nanosecond for t in bounds):
            raise ValueError
    except (TypeError, ValueError):
        raise ApiError("时间须包含时区、按十五分钟对齐，且窗口为正、不超过九十天。") from None
    if not set(device_ids) <= set(profile["device_ids"]) or len(set(device_ids)) != len(device_ids):
        raise ApiError("设备必须在当前配置的可见范围内，且不能重复。")
    return {
        "profile_id": profile["profile_id"],
        "request": {
            "system_id": profile["system_id"],
            "device_ids": device_ids,
            "start": bounds[0].isoformat(),
            "end": bounds[1].isoformat(),
            "question": question,
        },
    }


def prepare_request(session, body):
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    previous = session.get("request")
    if not previous or previous["digest"] != digest:
        session = empty_session()
        session["request"] = {"digest": digest, "key": uuid4().hex}
    return session


def apply_event(session, event):
    if event.kind == "accepted":
        run_id = event.data["run_id"]
        run_path(run_id)
        if session["run_id"] and session["run_id"] != run_id:
            raise ApiError("收到不属于当前运行的事件，请重新读取运行。")
        session["run_id"] = run_id
        session["reply"] = None
    if event.cursor:
        prefix, separator, sequence = event.cursor.partition(":")
        if not separator or prefix != session["run_id"] or not sequence.isdigit():
            raise ApiError("服务返回的事件游标与当前运行不一致。")
        if session["cursor"] and int(session["cursor"].rsplit(":", 1)[1]) >= int(sequence):
            return
        session["cursor"] = event.cursor
    if event.kind == "node":
        session["reply"] = None
        session["progress"].append(
            [
                event.cursor or "",
                event.data.get("node", ""),
                event.data.get("loops", ""),
                event.data.get("tool_calls", ""),
                event.data.get("report_status", ""),
            ]
        )
        session["progress"] = session["progress"][-200:]
    elif event.kind in {"completed", "awaiting_approval", "running", "interrupted"}:
        if event.data.get("run_id") != session["run_id"]:
            raise ApiError("服务返回的结果与当前运行不一致。")
        session["reply"] = event.data
    elif event.kind == "error":
        session["reply"] = None
        raise ApiError("运行或进度连接已中断，请读取当前运行，必要时恢复。")


class Controller:
    def __init__(self, client):
        self.client = client

    async def start(self, token, session, body):
        # The caller persists this initial yield before any HTTP request can fail.
        session = prepare_request(copy.deepcopy(session), body)
        yield session
        async for state in self._consume(
            token,
            session,
            self.client.stream(
                token, "POST", "/diagnose", body=body, key=session["request"]["key"]
            ),
        ):
            yield state

    async def _consume(self, token, session, events):
        received = False
        async for event in events:
            received = True
            apply_event(session, event)
            yield session
        if not received or session["reply"] is None:
            if not session["run_id"]:
                raise ApiError("没有收到运行 ID，请用相同输入重试。")
            reply = await self.client.view(token, session["run_id"])
            apply_event(session, Event(reply["status"], reply))
            yield session

    async def read(self, token, session, run_id):
        run_path(run_id)
        if session["run_id"] != run_id:
            session = empty_session()
            session["run_id"] = run_id
        else:
            session = copy.deepcopy(session)
        # Clear any old proposal before attempting a read under new credentials.
        session["reply"] = None
        yield session
        reply = await self.client.view(token, run_id)
        apply_event(session, Event(reply["status"], reply))
        yield session

    async def reconnect(self, token, session, *, resume=False):
        session = copy.deepcopy(session)
        path = run_path(session["run_id"])
        session["reply"] = None
        yield session
        events = self.client.stream(
            token,
            "POST" if resume else "GET",
            path + ("/resume" if resume else "/events"),
            cursor=None if resume else session["cursor"],
        )
        async for state in self._consume(token, session, events):
            yield state

    async def decide(self, token, session, *, approved, confirmed, reason):
        if not confirmed:
            raise ApiError("请先核对完整草案，并勾选人工确认。")
        reply = session.get("reply")
        if not reply or reply["status"] != "awaiting_approval":
            raise ApiError("请先读取当前待审批运行并核对草案。")
        proposal = reply["approval_request"]["proposal"]
        # Bind to the displayed snapshot, never to the editable run ID textbox.
        body = {
            "approved": approved,
            "action_id": proposal["action_id"],
            "payload_sha256": proposal["payload_sha256"],
            "reason": reason,
        }
        session = copy.deepcopy(session)
        session["reply"] = None
        yield session
        async for state in self._consume(
            token,
            session,
            self.client.stream(token, "POST", run_path(session["run_id"]) + "/approve", body=body),
        ):
            yield state


def report_view(session):
    reply = session.get("reply") or {}
    result = reply.get("result") or {}
    approval = reply.get("approval_request") or {}
    proposal = approval.get("proposal") or {}
    payload = proposal.get("payload") or {}
    report = result.get("report") or payload.get("report") or {}
    evidence = result.get("evidence") or payload.get("evidence") or {}
    faults = [
        [
            f["device_id"],
            f["fault_type"],
            f["window"]["start"]["value"],
            f["window"]["end"]["value"],
            f["confidence"],
            f["rationale"],
        ]
        for f in report.get("findings", [])
    ]
    fallback = "正在连接或读取运行" if session.get("run_id") else "等待诊断或读取运行"
    state = STATUS.get(reply.get("status"), fallback)
    if report:
        state += " · " + STATUS.get(report["status"], report["status"])
    return {
        "status": state,
        "summary": report.get("summary", "选择数据配置后开始诊断，或输入运行 ID 读取已有结果。"),
        "faults": faults,
        "evidence": evidence,
        "proposal": proposal or None,
        "approval_enabled": reply.get("status") == "awaiting_approval" and bool(proposal),
        "result": result,
    }


def chart_data(evidence, evidence_id, metric):
    item = evidence.get(evidence_id, {})
    rows = []
    if item.get("tool") == "get_timeseries":
        for series in item["result"].get("series", []):
            if series["metric"] == metric:
                for point in series["points"]:
                    ts = pd.to_datetime(point["ts"], utc=True, errors="coerce")
                    if not pd.isna(ts):
                        rows.append([ts, point["value"], series["device_id"]])
    return pd.DataFrame(rows, columns=["UTC", "数值", "设备"])


def peer_data(evidence, evidence_id):
    item = evidence.get(evidence_id, {})
    rows = []
    if item.get("tool") == "peer_compare":
        rows = [
            [device["device_id"], device["low_ratio_fraction"]]
            for device in item["result"]["devices"]
        ]
    return pd.DataFrame(rows, columns=["设备", "低值占比"])
