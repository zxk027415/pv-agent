"""Context-local observations across async tools, models and resumed diagnoses."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from time import time_ns
from uuid import uuid4

from pvdiag.observability.store import bounded

_SESSION = ContextVar("pvdiag_trace_session", default=None)
_SPAN = ContextVar("pvdiag_observation", default=None)


class Observation:
    def __init__(self, session=None, record=None):
        self.session, self.record = session, record

    def update(self, *, output=None, metadata=None, success=None):
        if self.record is None:
            return
        if output is not None:
            self.record["output"] = bounded(output)
        if metadata:
            self.record["metadata"].update(bounded(metadata))
        if success is not None:
            self.record["success"] = success
        self.save()

    def save(self):
        if self.session:
            self.session.store.put(
                "span", self.record["trace_id"] + ":" + self.record["span_id"], self.record
            )

    def end(self, success=True):
        if self.record:
            self.record["end_ns"] = max(self.record["start_ns"], time_ns())
            if self.record["success"] is None:
                self.record["success"] = success
            self.save()


class TraceSession:
    def __init__(self, store, run_id, request, metadata):
        self.store = store
        self.trace_id = hashlib.sha256(f"pvdiag-trace-v1\0{run_id}".encode()).hexdigest()[:32]
        root_id = hashlib.sha256(f"{self.trace_id}\0root".encode()).hexdigest()[:16]
        previous = store.get("span", self.trace_id + ":" + root_id)
        public_input = bounded(request)
        if previous and previous["input"] != public_input:
            raise ValueError("trace_request_identity_conflict")
        self.root = Observation(
            self,
            {
                "trace_id": self.trace_id,
                "span_id": root_id,
                "parent_id": None,
                "type": "agent",
                "name": "pvdiag.diagnosis",
                "input": public_input,
                "output": None,
                "metadata": bounded(metadata),
                "start_ns": previous["start_ns"] if previous else time_ns(),
                "end_ns": None,
                "success": None,
            },
        )
        self.root.save()

    def receipt(self):
        return {
            "version": "local-otlp-v1",
            "trace_id": self.trace_id,
            "root_span_id": self.root.record["span_id"],
        }


def current_session():
    return _SESSION.get()


def mark_request_sent():
    if span := _SPAN.get():
        span.update(metadata={"request_sent": True})


@contextmanager
def trace_session(store, run_id, request, metadata):
    if store is None:
        yield None
        return
    if _SESSION.get() is not None:
        raise ValueError("nested diagnosis traces are not allowed")
    session = TraceSession(store, run_id, request, metadata)
    session_token = _SESSION.set(session)
    span_token = _SPAN.set(session.root)
    try:
        yield session
    except BaseException as exc:
        session.root.update(metadata={"execution_error": type(exc).__name__}, success=False)
        raise
    finally:
        try:
            session.root.end()
        finally:
            _SPAN.reset(span_token)
            _SESSION.reset(session_token)


@contextmanager
def observe(kind, name, input_data, metadata=None):
    session = _SESSION.get()
    if session is None:
        yield Observation()
        return
    parent = _SPAN.get()
    span = Observation(
        session,
        {
            "trace_id": session.trace_id,
            "span_id": uuid4().hex[:16],
            "parent_id": parent.record["span_id"],
            "type": kind,
            "name": name,
            "input": bounded(input_data),
            "output": None,
            "metadata": bounded(metadata or {}) | {"request_sent": False},
            "start_ns": time_ns(),
            "end_ns": None,
            "success": None,
        },
    )
    span.save()
    token = _SPAN.set(span)
    try:
        yield span
    except BaseException as exc:
        span.update(metadata={"error_code": type(exc).__name__}, success=False)
        raise
    finally:
        try:
            span.end()
        finally:
            _SPAN.reset(token)
