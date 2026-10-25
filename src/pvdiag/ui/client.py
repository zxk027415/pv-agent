"""Bounded SSE transport to a server-owned API URL, with per-request credentials."""

import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

MAX_FRAME_BYTES = 4 * 1024 * 1024


class ApiError(Exception):
    """Safe, localized error text; never includes response or request internals."""


@dataclass
class Event:
    kind: str
    data: dict
    cursor: str | None = None


async def read_events(lines):
    fields, data, size = {}, [], 0
    async for line in lines:
        size += len(line.encode())
        if size > MAX_FRAME_BYTES:
            raise ApiError("进度事件过大，请使用运行 ID 查询状态。")
        if not line:
            if data:
                try:
                    payload = json.loads("\n".join(data))
                except ValueError:
                    raise ApiError("服务返回了无效的进度事件。") from None
                if not isinstance(payload, dict):
                    raise ApiError("服务返回了无效的进度事件。")
                yield Event(fields.get("event", "message"), payload, fields.get("id"))
            fields, data, size = {}, [], 0
            continue
        if line.startswith(":"):
            continue
        key, separator, value = line.partition(":")
        if separator:
            value = value.removeprefix(" ")
            if key == "data":
                data.append(value)
            elif key in {"event", "id"}:
                fields[key] = value
    if data:
        raise ApiError("进度连接提前中断，请使用当前运行 ID 重连。")


def run_path(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", run_id):
        raise ApiError("请输入有效的运行 ID。")
    return f"/runs/{run_id}"


class ApiClient:
    def __init__(self, base_url, *, transport=None):
        url = urlsplit(base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
            or url.scheme == "http"
            and url.hostname not in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("use a loopback HTTP or HTTPS API origin without credentials")
        self.base_url, self.transport = base_url.rstrip("/"), transport

    @asynccontextmanager
    async def connection(self, token):
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 1024
            or not token.isascii()
            or any(ord(c) < 32 or ord(c) == 127 for c in token)
        ):
            raise ApiError("请填写有效的 API 访问令牌。")
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {token}"},
                transport=self.transport,
                follow_redirects=False,
                trust_env=False,
                timeout=httpx.Timeout(15, read=30),
            ) as client:
                yield client
        except httpx.HTTPError:
            raise ApiError("无法连接诊断服务；已有运行可凭运行 ID 查询或重连。") from None

    @staticmethod
    def check(response):
        if response.status_code not in {200, 204}:
            message = {
                401: "身份无效或已过期，请检查访问令牌。",
                403: "当前身份没有执行此操作的权限。",
                404: "找不到该运行或数据配置，或当前身份无权访问。",
                409: "请求与已有运行或审批决定冲突，请先重新读取运行。",
                422: "请求超出可用范围或格式不正确，请检查输入。",
                429: "诊断容量已满，请稍后重试当前请求。",
            }.get(response.status_code, "诊断服务暂不可用，请稍后查询当前运行。")
            raise ApiError(message)

    async def profiles(self, token):
        async with self.connection(token) as client:
            reply = await client.get("/profiles")
            self.check(reply)
            try:
                profiles = reply.json()
                if not isinstance(profiles, list) or not all(
                    isinstance(p, dict)
                    and {"profile_id", "system_id", "device_ids", "start", "end", "mode"}
                    <= p.keys()
                    for p in profiles
                ):
                    raise ValueError
            except ValueError:
                raise ApiError("服务返回了无效的数据配置。") from None
            return profiles

    async def view(self, token, run_id):
        async with self.connection(token) as client:
            reply = await client.get(run_path(run_id))
            self.check(reply)
            return self.reply_json(reply)

    @staticmethod
    def reply_json(response):
        try:
            reply = response.json()
            if not isinstance(reply, dict) or not {"run_id", "status"} <= reply.keys():
                raise ValueError
        except ValueError:
            raise ApiError("服务返回了无效的运行状态。") from None
        return reply

    async def stream(self, token, method, path, *, body=None, key=None, cursor=None):
        headers = {"Accept": "text/event-stream"}
        if key:
            headers["Idempotency-Key"] = key
        if cursor:
            headers["Last-Event-ID"] = cursor
        async with (
            self.connection(token) as client,
            client.stream(method, path, json=body, headers=headers) as response,
        ):
            self.check(response)
            if response.status_code == 204:
                return
            if response.headers.get("content-type", "").startswith("application/json"):
                await response.aread()
                reply = self.reply_json(response)
                yield Event(reply["status"], reply)
                return
            if not response.headers.get("content-type", "").startswith("text/event-stream"):
                raise ApiError("诊断服务没有返回预期的进度流。")
            async for event in read_events(response.aiter_lines()):
                yield event
