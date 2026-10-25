import asyncio
import copy
import json
import os

import httpx
import pandas as pd
import pytest

from pvdiag.ui.client import MAX_FRAME_BYTES, ApiClient, ApiError, Event, read_events, run_path
from pvdiag.ui.controller import (
    Controller,
    apply_event,
    chart_data,
    empty_session,
    peer_data,
    prepare_request,
    report_view,
    request_body,
)


def test_ui_sse_parser_handles_comments_multiline_utf8_and_rejects_truncation():
    async def lines(values):
        for value in values:
            yield value

    async def verify():
        events = [
            e
            async for e in read_events(
                lines(
                    [
                        ": heartbeat",
                        "",
                        "id: run:1",
                        "event: node",
                        'data: {"node":',
                        'data: "诊断"}',
                        "",
                        "id: unused-comment",
                        ": ping",
                        "",
                    ]
                )
            )
        ]
        assert events == [Event("node", {"node": "诊断"}, "run:1")]
        for payload in [
            ['data: {"incomplete":true}'],
            ["data: []", ""],
            ["data: invalid", ""],
            ["data: " + "x" * MAX_FRAME_BYTES],
        ]:
            with pytest.raises(ApiError):
                _ = [event async for event in read_events(lines(payload))]

    asyncio.run(verify())


def test_ui_transport_never_follows_redirects_or_surfaces_error_bodies():
    async def verify():
        seen = []

        def route(request):
            seen.append(request)
            return httpx.Response(
                302,
                headers={"location": "https://other.invalid"},
                text="controlled-secret-in-upstream-error",
            )

        client = ApiClient("http://127.0.0.1:8000", transport=httpx.MockTransport(route))
        with pytest.raises(ApiError) as error:
            await client.profiles("controlled-per-user-token")
        assert len(seen) == 1
        assert seen[0].headers["authorization"] == "Bearer controlled-per-user-token"
        assert "controlled" not in str(error.value)
        assert "controlled" not in str(error.value.__cause__)
        for invalid in ["", "bad\r\nheader", "x" * 1025, "非 ASCII 令牌", "bad\x00token"]:
            with pytest.raises(ApiError):
                await client.profiles(invalid)
        assert len(seen) == 1

    asyncio.run(verify())
    for url in [
        "http://other.invalid",
        "http://user:secret@127.0.0.1",
        "https://api.invalid?q=key",
    ]:
        with pytest.raises(ValueError):
            ApiClient(url)
    for run_id in ["../../other", "run?token=x", "", "a" * 101]:
        with pytest.raises(ApiError):
            run_path(run_id)


def test_ui_retry_keeps_idempotency_key_after_failure_and_states_are_isolated():
    async def verify():
        keys = []

        def route(request):
            keys.append(request.headers["idempotency-key"])
            if len(keys) == 1:
                raise httpx.ReadError("controlled-secret-do-not-echo")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='id: run:1\nevent: accepted\ndata: {"run_id":"run"}\n\n'
                'id: run:2\nevent: completed\ndata: {"run_id":"run","status":"completed"}\n\n',
            )

        controller = Controller(ApiClient("http://localhost", transport=httpx.MockTransport(route)))
        state, other = empty_session(), empty_session()
        body = {"profile_id": "fixture", "request": {"question": "scope"}}
        with pytest.raises(ApiError):
            async for current in controller.start("token", state, body):
                state = copy.deepcopy(current)
        assert state["request"]["key"] and not state["run_id"]
        async for current in controller.start("token", state, body):
            state = copy.deepcopy(current)
        assert keys[0] == keys[1]
        assert state["run_id"] == "run" and state["reply"]["status"] == "completed"
        assert other == empty_session()
        new = prepare_request(state, {"profile_id": "different"})
        assert new["request"]["key"] != keys[0] and new["reply"] is None
        apply_event(state, Event("node", {"node": "old"}, "run:1"))
        assert state["progress"] == []
        with pytest.raises(ApiError):
            apply_event(state, Event("node", {"node": "wrong"}, "other:3"))

    asyncio.run(verify())


def test_ui_reconnect_handles_exhausted_cursor_and_resume_json_without_new_diagnosis():
    async def verify():
        seen = []

        def route(request):
            seen.append(request)
            if request.url.path.endswith("/events"):
                assert request.headers["last-event-id"] == "run:2"
                return httpx.Response(204)
            return httpx.Response(200, json={"run_id": "run", "status": "completed"})

        c = Controller(ApiClient("http://localhost", transport=httpx.MockTransport(route)))
        state = empty_session() | {"run_id": "run", "cursor": "run:2"}
        for resume in [False, True]:
            async for current in c.reconnect("token", state, resume=resume):
                state = copy.deepcopy(current)
            assert state["reply"]["status"] == "completed"
        assert [r.url.path for r in seen] == ["/runs/run/events", "/runs/run", "/runs/run/resume"]

    asyncio.run(verify())


def test_ui_approval_is_bound_to_displayed_snapshot_and_requires_explicit_confirmation():
    async def verify():
        seen = []

        def route(request):
            seen.append(request)
            return httpx.Response(200, json={"run_id": "displayed-run", "status": "completed"})

        c = Controller(ApiClient("http://localhost", transport=httpx.MockTransport(route)))
        state = empty_session() | {
            "run_id": "displayed-run",
            "reply": {
                "run_id": "displayed-run",
                "status": "awaiting_approval",
                "approval_request": {
                    "proposal": {
                        "action_id": "displayed-action",
                        "payload_sha256": "displayed-digest",
                    }
                },
            },
        }
        with pytest.raises(ApiError):
            _ = [
                v async for v in c.decide("token", state, approved=True, confirmed=False, reason="")
            ]
        assert seen == []
        values = [
            v
            async for v in c.decide(
                "reviewer-token", state, approved=False, confirmed=True, reason="检查后拒绝"
            )
        ]
        assert values[-1]["reply"]["status"] == "completed"
        assert seen[0].url.path == "/runs/displayed-run/approve"
        body = json.loads(seen[0].content)
        assert body == {
            "action_id": "displayed-action",
            "payload_sha256": "displayed-digest",
            "approved": False,
            "reason": "检查后拒绝",
        }
        assert "reviewer" not in body
        assert state["reply"]["status"] == "awaiting_approval"

    asyncio.run(verify())


def test_ui_plot_preserves_returned_points_nulls_metrics_and_exact_evidence_selection():
    evidence = {
        "series": {
            "tool": "get_timeseries",
            "result": {
                "series": [
                    {
                        "device_id": "inv01",
                        "metric": "ac_power_kw",
                        "points": [
                            {"ts": "2019-01-01T01:00:00+01:00", "value": 2.0},
                            {"ts": "2019-01-01T00:15:00Z", "value": None},
                        ],
                    },
                    {
                        "device_id": "poa",
                        "metric": "poa_wm2",
                        "points": [{"ts": "2019-01-01T00:00:00Z", "value": 600}],
                    },
                ]
            },
        },
        "peers": {
            "tool": "peer_compare",
            "result": {"devices": [{"device_id": "inv01", "low_ratio_fraction": 0.5}]},
        },
    }
    actual = chart_data(evidence, "series", "ac_power_kw")
    assert len(actual) == 2 and actual.iloc[0]["UTC"] == pd.Timestamp("2019-01-01", tz="UTC")
    assert actual.iloc[0]["数值"] == 2 and pd.isna(actual.iloc[1]["数值"])
    assert chart_data(evidence, "peers", "ac_power_kw").empty
    assert peer_data(evidence, "peers")["低值占比"].tolist() == [0.5]
    assert peer_data(evidence, "series").empty
    assert report_view(empty_session())["approval_enabled"] is False


def test_ui_input_requires_timezone_alignment_and_visible_devices():
    profile = {"profile_id": "fixture", "system_id": 2107, "device_ids": ["inv01"]}
    good = (profile, ["inv01"], "2019-01-01T00:00:00Z", "2019-01-02T00:00:00Z", "检查")
    assert request_body(*good)["request"]["system_id"] == 2107
    for start in ["2019-01-01", "2019-01-01T00:01:00Z", "NaT", "invalid"]:
        with pytest.raises(ApiError):
            request_body(profile, ["inv01"], start, good[3], "检查")
    with pytest.raises(ApiError):
        request_body(profile, ["inv02"], good[2], good[3], "检查")


def test_api_profile_defaults_align_non_quarter_hour_observations_for_the_ui(tmp_path):
    from test_verification import context

    from pvdiag.agent.state import DiagRequest
    from pvdiag.api.config import Profile

    context(tmp_path, "F3")
    observations = tmp_path / "observations.parquet"
    frame = pd.read_parquet(observations)
    frame["ts"] += pd.Timedelta(minutes=5)
    frame.to_parquet(observations, index=False)
    profile = Profile(
        {
            "profile_id": "fixture",
            "observations": observations,
            "metadata": tmp_path / "metadata.json",
            "allowed_device_ids": ["inv01"],
        }
    )
    public = profile.public()
    body = request_body(public, ["inv01"], public["start"], public["end"], "检查")
    request = DiagRequest.model_validate(body["request"])
    assert request.start == frame.ts.min().ceil("15min")
    assert request.end == frame.ts.max().floor("15min") + pd.Timedelta(minutes=15)
    profile.validate(request)


def test_ui_http_controller_uses_real_api_ownership_sse_and_identity_bound_decisions(tmp_path):
    if not os.getenv("PVDIAG_TEST_POSTGRES_DSN"):
        pytest.skip("an isolated PVDIAG_TEST_POSTGRES_DSN is required")
    from test_api import api_fixture, application

    async def verify():
        fixture = api_fixture.__wrapped__(tmp_path)
        app = await application(fixture)
        _, _, profile, request = fixture
        controller = Controller(
            ApiClient("http://localhost", transport=httpx.ASGITransport(app=app))
        )
        state = empty_session()
        body = {"profile_id": "fixture", "request": request.model_dump(mode="json")}
        async with app.router.lifespan_context(app):
            async for current in controller.start("controlled-owner-fixture", state, body):
                state = copy.deepcopy(current)
            assert len(state["progress"]) > 0
            assert report_view(state)["approval_enabled"]
            assert profile.ops.work_orders() == []
            run_id = state["run_id"]
            with pytest.raises(ApiError, match="找不到"):
                _ = [v async for v in controller.read("controlled-outsider-fixture", state, run_id)]
            # Requester cannot approve even with the exact displayed content.
            with pytest.raises(ApiError, match="权限"):
                _ = [
                    v
                    async for v in controller.decide(
                        "controlled-owner-fixture", state, approved=True, confirmed=True, reason=""
                    )
                ]
            async for current in controller.decide(
                "controlled-reviewer-fixture",
                state,
                approved=True,
                confirmed=True,
                reason="受控界面测试",
            ):
                final = copy.deepcopy(current)
            assert final["reply"]["status"] == "completed"
            assert not report_view(final)["approval_enabled"]
            assert final["reply"]["result"]["approval"]["reviewer"] == "reviewer"
            assert len(profile.ops.work_orders()) == 1
            async for current in controller.start("controlled-owner-fixture", state, body):
                retried = copy.deepcopy(current)
            assert retried["run_id"] == run_id
            assert len(profile.ops.work_orders()) == 1

    asyncio.run(verify())
