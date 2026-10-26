import asyncio
import hashlib
import json
import os
import socket
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("fastapi")

from test_verification import context

from pvdiag.agent.persistence import PostgresStore
from pvdiag.agent.rules import RuleReasoner
from pvdiag.api.app import create_app
from pvdiag.api.config import Authentication, Profile
from pvdiag.api.store import ApiStore
from pvdiag.observability.store import TraceStore, canonical


def identities():
    return Authentication(
        [
            {
                "subject": subject,
                "token_sha256": hashlib.sha256(
                    f"controlled-{subject}-fixture".encode()
                ).hexdigest(),
                "roles": roles,
                "profiles": ["fixture"],
            }
            for subject, roles in [
                ("owner", ["diagnose", "read"]),
                ("reviewer", ["approve"]),
                ("outsider", ["diagnose", "read"]),
            ]
        ]
    )


def headers(subject="owner", key=None):
    result = {"Authorization": f"Bearer controlled-{subject}-fixture"}
    if key:
        result["Idempotency-Key"] = key
    return result


def messages(text):
    result = []
    for frame in text.split("\n\n"):
        fields = dict(
            line.split(": ", 1)
            for line in frame.splitlines()
            if ": " in line and not line.startswith(":")
        )
        if "event" in fields and "data" in fields:
            result.append(
                {
                    "id": fields.get("id"),
                    "event": fields["event"],
                    "data": json.loads(fields["data"]),
                }
            )
    return result


@pytest.fixture
def api_fixture(tmp_path):
    dsn = os.getenv("PVDIAG_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("an isolated PVDIAG_TEST_POSTGRES_DSN is required")
    _, request, _, _, _ = context(tmp_path, "F3")
    profile = Profile(
        {
            "profile_id": "fixture",
            "observations": tmp_path / "observations.parquet",
            "metadata": tmp_path / "metadata.json",
            "allowed_device_ids": request.device_ids,
            "ops_db": tmp_path / "orders.sqlite3",
        }
    )
    return dsn, "pvdiag_api_" + uuid4().hex, profile, request


async def application(fixture, **kwargs):
    dsn, schema, profile, _ = fixture
    async with PostgresStore.open(dsn, schema=schema, initialize=True) as persistence:
        await ApiStore(persistence).initialize()
    return create_app(
        {"fixture": profile}, identities(), dsn, schema=schema, poll_interval=0.01, **kwargs
    )


def test_authentication_has_no_default_token_and_never_uses_query_credentials():
    with pytest.raises(ValueError):
        Authentication([])
    auth = identities()
    assert auth.authenticate(None) is None
    assert auth.authenticate("controlled-owner-fixture") is None
    assert auth.authenticate("Bearer wrong-fixture") is None
    assert auth.authenticate(headers()["Authorization"]).subject == "owner"


def test_api_trace_survives_approval_resume_and_excludes_authentication(api_fixture, tmp_path):
    traces = TraceStore(tmp_path / "api-traces.sqlite3")

    async def verify():
        app = await application(api_fixture, trace_store=traces)
        request = api_fixture[3]
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://fixture"
            ) as client,
        ):
            response = await client.post(
                "/diagnose",
                headers=headers(key="controlled-trace-0001"),
                json={"profile_id": "fixture", "request": request.model_dump(mode="json")},
            )
            assert response.status_code == 200
            pending = messages(response.text)[-1]["data"]["approval_request"]["proposal"]
            roots = [s for s in traces.all("span") if s["parent_id"] is None]
            assert len(roots) == 1 and roots[0]["metadata"]["phase"] == "awaiting_approval"
            run_id = response.headers["x-run-id"]
            response = await client.post(
                f"/runs/{run_id}/approve",
                headers=headers("reviewer"),
                json={
                    "approved": False,
                    "action_id": pending["action_id"],
                    "payload_sha256": pending["payload_sha256"],
                },
            )
            assert response.status_code == 200
            result = messages(response.text)[-1]["data"]["result"]
            assert result["action_outcome"]["status"] == "rejected"
            assert result["observability"]["trace_id"] == roots[0]["trace_id"]
            assert len(traces.all("span")) == len(result["trace"]) + 1
            assert "controlled-owner-fixture" not in canonical(traces.snapshot())
            assert "controlled-reviewer-fixture" not in canonical(traces.snapshot())

    asyncio.run(verify())


def test_http_diagnosis_progress_identity_bound_approval_and_idempotent_retries(api_fixture):
    async def verify():
        app = await application(api_fixture)
        _, _, profile, request = api_fixture
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://fixture"
            ) as client,
        ):
            assert (await client.get("/profiles")).status_code == 401
            public = await client.get("/profiles", headers=headers())
            assert public.status_code == 200 and public.json()[0]["profile_id"] == "fixture"
            assert "observations.parquet" not in public.text and "token_sha256" not in public.text
            body = {"profile_id": "fixture", "request": request.model_dump(mode="json")}
            response = await client.post(
                "/diagnose", json=body, headers=headers(key="controlled-request-0001")
            )
            assert response.status_code == 200 and response.headers["content-type"].startswith(
                "text/event-stream"
            )
            events = messages(response.text)
            assert events[0]["event"] == "accepted" and any(e["event"] == "node" for e in events)
            assert events[-1]["event"] == "awaiting_approval"
            run_id = response.headers["x-run-id"]
            assert profile.ops.work_orders() == []
            retried = await client.post(
                "/diagnose", json=body, headers=headers(key="controlled-request-0001")
            )
            assert retried.headers["x-run-id"] == run_id
            assert messages(retried.text)[-1]["data"] == events[-1]["data"]
            proposal = events[-1]["data"]["approval_request"]["proposal"]
            approval = {
                "approved": True,
                "action_id": proposal["action_id"],
                "payload_sha256": proposal["payload_sha256"],
            }
            assert (
                await client.post(f"/runs/{run_id}/approve", json=approval, headers=headers())
            ).status_code == 403
            assert (
                await client.get(f"/runs/{run_id}", headers=headers("outsider"))
            ).status_code == 404
            assert (
                await client.get(f"/runs/{run_id}", headers=headers("reviewer"))
            ).status_code == 200
            approved = await client.post(
                f"/runs/{run_id}/approve", json=approval, headers=headers("reviewer")
            )
            assert approved.status_code == 200
            final = messages(approved.text)[-1]
            assert final["event"] == "completed"
            assert final["data"]["result"]["approval"]["reviewer"] == "reviewer"
            assert final["data"]["result"]["action_outcome"]["status"] == "created"
            repeated = await client.post(
                f"/runs/{run_id}/approve", json=approval, headers=headers("reviewer")
            )
            assert repeated.json() == final["data"]
            assert len(profile.ops.work_orders()) == 1
            later = await client.get(
                f"/runs/{run_id}/events", headers=headers() | {"Last-Event-ID": events[-1]["id"]}
            )
            assert messages(later.text)[-1]["event"] == "completed"
            final_id = messages(later.text)[-1]["id"]
            assert (
                await client.get(
                    f"/runs/{run_id}/events", headers=headers() | {"Last-Event-ID": final_id}
                )
            ).status_code == 204
            assert (
                await client.get(
                    f"/runs/{run_id}/events",
                    headers=headers() | {"Last-Event-ID": f"{run_id}:999999"},
                )
            ).status_code == 400

    asyncio.run(verify())


def test_api_configuration_errors_do_not_echo_credentials(monkeypatch, capsys):
    from pvdiag.api.__main__ import main

    monkeypatch.setenv("PVDIAG_POSTGRES_DSN", "controlled-not-a-real-connection")
    monkeypatch.setenv(
        "PVDIAG_API_TOKENS",
        json.dumps([{"subject": "operator", "token": "controlled-secret-not-to-echo"}]),
    )
    with pytest.raises(SystemExit):
        main(["serve", "--profiles", "unread-fixture-path.json"])
    assert "controlled-secret-not-to-echo" not in capsys.readouterr().err


def test_approved_decision_is_durable_before_worker_start_and_recovered_after_restart(
    api_fixture, monkeypatch
):
    from fastapi import HTTPException

    async def verify():
        app = await application(api_fixture)
        dsn, schema, profile, request = api_fixture
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://fixture"
            ) as client,
        ):
            body = {"profile_id": "fixture", "request": request.model_dump(mode="json")}
            pending = await client.post(
                "/diagnose", json=body, headers=headers(key="restart-approved-decision")
            )
            run_id = pending.headers["x-run-id"]
            proposal = messages(pending.text)[-1]["data"]["approval_request"]["proposal"]
            decision = {
                "approved": True,
                "action_id": proposal["action_id"],
                "payload_sha256": proposal["payload_sha256"],
            }

            async def unavailable(*args, **kwargs):
                raise HTTPException(503, "controlled worker unavailable")

            monkeypatch.setattr(app.state.service, "spawn", unavailable)
            failed = await client.post(
                f"/runs/{run_id}/approve", json=decision, headers=headers("reviewer")
            )
            assert failed.status_code == 503
            assert profile.ops.approvals()[0].reviewer == "reviewer"
            assert profile.ops.work_orders() == []
        restarted = create_app(
            {"fixture": profile}, identities(), dsn, schema=schema, poll_interval=0.01
        )
        async with (
            restarted.router.lifespan_context(restarted),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted), base_url="http://fixture"
            ) as client,
        ):
            resumed = await client.post(f"/runs/{run_id}/resume", headers=headers())
            final = messages(resumed.text)[-1]
            assert final["data"]["result"]["action_outcome"]["status"] == "created"
            assert final["data"]["result"]["approval"]["reviewer"] == "reviewer"
            assert len(profile.ops.work_orders()) == 1

    asyncio.run(verify())


def test_capacity_rejection_does_not_register_a_second_run(api_fixture):
    async def verify():
        entered, release = asyncio.Event(), asyncio.Event()

        class WaitingRule(RuleReasoner):
            async def plan(self, state, tools, prompt):
                entered.set()
                await release.wait()
                return await super().plan(state, tools, prompt)

        _, _, profile, request = api_fixture
        profile._factory = WaitingRule
        app = await application(api_fixture, max_active=1)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://fixture"
            ) as client,
        ):
            body = {"profile_id": "fixture", "request": request.model_dump(mode="json")}
            first = asyncio.create_task(
                client.post("/diagnose", json=body, headers=headers(key="capacity-first-fixture"))
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                denied = await client.post(
                    "/diagnose", json=body, headers=headers(key="capacity-second-fixture")
                )
                assert denied.status_code == 429
                assert (
                    await app.state.service.repository.find("owner", "capacity-second-fixture")
                    is None
                )
            finally:
                release.set()
                assert (await first).status_code == 200

    asyncio.run(verify())


def test_worker_errors_are_redacted_and_restart_reads_persisted_events(api_fixture):
    async def verify():
        dsn, schema, profile, request = api_fixture

        def broken_factory():
            raise RuntimeError("controlled-private-credential-fixture")

        profile._factory = broken_factory
        app = await application(api_fixture)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://fixture"
            ) as client,
        ):
            response = await client.post(
                "/diagnose",
                json={"profile_id": "fixture", "request": request.model_dump(mode="json")},
                headers=headers(key="redaction-fixture-key"),
            )
            run_id = response.headers["x-run-id"]
            assert messages(response.text)[-1]["event"] == "error"
            assert "controlled-private-credential-fixture" not in response.text
        restarted = create_app(
            {"fixture": profile}, identities(), dsn, schema=schema, poll_interval=0.01
        )
        async with (
            restarted.router.lifespan_context(restarted),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted), base_url="http://fixture"
            ) as client,
        ):
            replay = await client.get(f"/runs/{run_id}/events", headers=headers())
            assert messages(replay.text) == messages(response.text)

    asyncio.run(verify())


def test_http_rejects_forged_identity_scope_changes_and_idempotency_conflicts(api_fixture):
    async def verify():
        app = await application(api_fixture)
        _, _, profile, request = api_fixture
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://fixture"
            ) as client,
        ):
            body = {"profile_id": "fixture", "request": request.model_dump(mode="json")}
            bad = body | {"observations": "/private/forbidden.parquet"}
            assert (
                await client.post("/diagnose", json=bad, headers=headers(key="controlled-key-0002"))
            ).status_code == 422
            bad = body | {"request": body["request"] | {"device_ids": ["inv99"]}}
            assert (
                await client.post("/diagnose", json=bad, headers=headers(key="controlled-key-0002"))
            ).status_code == 422
            response = await client.post(
                "/diagnose", json=body, headers=headers(key="controlled-key-0002")
            )
            run_id = response.headers["x-run-id"]
            proposal = messages(response.text)[-1]["data"]["approval_request"]["proposal"]
            approval = {
                "approved": False,
                "action_id": proposal["action_id"],
                "payload_sha256": proposal["payload_sha256"],
            }
            forged = await client.post(
                f"/runs/{run_id}/approve",
                json=approval | {"reviewer": "forged-secret-fixture"},
                headers=headers("reviewer"),
            )
            assert forged.status_code == 422 and "forged-secret-fixture" not in forged.text
            assert (
                await client.post(
                    f"/runs/{run_id}/approve",
                    json=approval | {"approved": "false"},
                    headers=headers("reviewer"),
                )
            ).status_code == 422
            assert (
                await client.post(
                    f"/runs/{run_id}/approve",
                    json=approval | {"payload_sha256": "0" * 64},
                    headers=headers("reviewer"),
                )
            ).status_code == 409
            changed = body | {"request": body["request"] | {"question": "再次检查光伏故障。"}}
            assert (
                await client.post(
                    "/diagnose", json=changed, headers=headers(key="controlled-key-0002")
                )
            ).status_code == 409
            rejected = await client.post(
                f"/runs/{run_id}/approve", json=approval, headers=headers("reviewer")
            )
            assert (
                messages(rejected.text)[-1]["data"]["result"]["action_outcome"]["status"]
                == "rejected"
            )
            assert profile.ops.work_orders() == []

    asyncio.run(verify())


def test_real_http_disconnect_leaves_owned_worker_running_and_events_can_resume(api_fixture):
    import uvicorn

    async def verify():
        entered, release = asyncio.Event(), asyncio.Event()

        class WaitingRule(RuleReasoner):
            async def plan(self, state, tools, prompt):
                entered.set()
                await release.wait()
                return await super().plan(state, tools, prompt)

        _, _, profile, request = api_fixture
        profile._factory = WaitingRule
        app = await application(api_fixture)
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", access_log=False, timeout_graceful_shutdown=1)
        )
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            for _ in range(500):
                if server.started:
                    break
                if task.done():
                    await task
                await asyncio.sleep(0.01)
            assert server.started
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
                body = {"profile_id": "fixture", "request": request.model_dump(mode="json")}
                async with client.stream(
                    "POST", "/diagnose", json=body, headers=headers(key="disconnect-fixture-001")
                ) as response:
                    run_id = response.headers["x-run-id"]
                    first_id = None
                    async for line in response.aiter_lines():
                        if line.startswith("id: "):
                            first_id = line[4:]
                            break
                    await asyncio.wait_for(entered.wait(), timeout=5)
                assert not release.is_set()
                state = await client.get(f"/runs/{run_id}", headers=headers())
                assert state.json()["status"] == "running"
                release.set()
                replayed = await client.get(
                    f"/runs/{run_id}/events", headers=headers() | {"Last-Event-ID": first_id}
                )
                progress = messages(replayed.text)
                assert progress and all(e["id"] != first_id for e in progress)
                assert progress[-1]["event"] == "awaiting_approval"
        finally:
            release.set()
            server.should_exit = True
            await asyncio.wait_for(task, timeout=10)
            sock.close()

    asyncio.run(verify())
