"""Structured LangChain model adapter; constructing it performs no API calls."""

from __future__ import annotations

import asyncio
import hashlib
import json

from langchain.chat_models import init_chat_model

from pvdiag.agent.state import DiagnosisReport, ModelReview, Plan
from pvdiag.observability.tracing import mark_request_sent, observe


class StructuredOutputError(ValueError):
    """Model output failed its schema; ``details`` holds field locations, never values."""

    def __init__(self, details):
        super().__init__("invalid structured model output")
        self.details = details


def output_errors(error):
    """Field location, error type and a short message from a parse failure; no output text."""
    errors = getattr(error, "errors", None)
    if callable(errors):
        try:
            return [
                {
                    "loc": ".".join(str(part) for part in item.get("loc", ())) or "(root)",
                    "type": str(item.get("type", "")),
                    "msg": str(item.get("msg", ""))[:160],
                }
                for item in errors(include_input=False, include_url=False)[:10]
            ]
        except TypeError:
            pass
    return [{"loc": "(unknown)", "type": type(error).__name__, "msg": ""}]


class LLMReasoner:
    def __init__(
        self,
        model: str,
        *,
        provider="openai",
        base_url=None,
        temperature=0.0,
        api_key=None,
        max_model_calls=6,
        max_review_calls=3,
        call_tracker=None,
    ):
        if not 1 <= max_model_calls <= 30:
            raise ValueError("invalid model call budget")
        if not 1 <= max_review_calls <= 30:
            raise ValueError("invalid model review budget")
        if base_url and api_key is None:
            raise ValueError(
                "custom endpoints require an explicit API key; no implicit credential fallback"
            )
        self.max_model_calls = max_model_calls
        self.max_review_calls = max_review_calls
        self.call_tracker = call_tracker
        kwargs = {
            "model_provider": provider,
            "temperature": temperature,
            "timeout": 60,
            "max_retries": 0,
            "max_tokens": 4096,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.model = init_chat_model(model, **kwargs)
        self.metadata = {
            "mode": "LLM",
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "reported_models": [],
            "max_model_calls": max_model_calls,
            "max_review_calls": max_review_calls,
            "endpoint_sha256": hashlib.sha256(
                str(
                    base_url
                    or getattr(self.model, "openai_api_base", None)
                    or f"{provider}_default"
                ).encode()
            ).hexdigest(),
        }
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 0,
            "review_calls": 0,
            "usage_missing_calls": 0,
        }

    async def sync_accounting(self):
        if self.call_tracker is not None:
            try:
                self.usage, models = await self.call_tracker.snapshot()
                # Trackers without a review ledger (Ragas) report no review_calls key.
                self.usage.setdefault("review_calls", 0)
                self.metadata["reported_models"] = models
            except Exception:  # noqa: BLE001 -- storage errors may contain credential-bearing DSNs
                raise RuntimeError("model_accounting_unavailable") from None

    async def _settle(self, reservation, usage, reported):
        if self.call_tracker is not None:
            try:
                await self.call_tracker.finish(reservation, usage, reported)
                await self.sync_accounting()
            except Exception:  # noqa: BLE001 -- redact all storage exception details
                raise RuntimeError("model_accounting_unavailable") from None
        else:
            if usage:
                for key in ("input_tokens", "output_tokens"):
                    self.usage[key] += usage.get(key, 0)
            else:
                self.usage["usage_missing_calls"] += 1
            if reported and reported not in self.metadata["reported_models"]:
                self.metadata["reported_models"].append(reported)

    async def _call(self, schema, payload, prompt, purpose="diagnosis"):
        with observe(
            "generation",
            f"llm.{payload.get('task', 'structured')}",
            {"system_prompt": prompt, "payload": payload},
            {
                "model": self.metadata["model"],
                "provider": self.metadata["provider"],
                "temperature": self.metadata["temperature"],
                "max_output_tokens": 4096,
            },
        ) as observation:
            return await self._invoke(schema, payload, prompt, observation, purpose)

    async def _invoke(self, schema, payload, prompt, observation, purpose="diagnosis"):
        # Review calls draw on their own ledger: a diagnosis that legitimately
        # uses every loop must not starve the verification gate that judges it.
        if purpose == "review":
            if self.usage["review_calls"] >= self.max_review_calls:
                raise ValueError("model_review_budget_exhausted")
        elif self.usage["model_calls"] >= self.max_model_calls:
            raise ValueError("model_call_budget_exhausted")
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(serialized) > 80_000:
            raise ValueError("model context exceeds configured character budget")
        reservation = None
        if self.call_tracker is not None:
            try:
                reservation = await self.call_tracker.reserve(purpose)
            except ValueError:
                raise
            except Exception:  # noqa: BLE001 -- redact all storage exception details
                raise RuntimeError("model_accounting_unavailable") from None
        if purpose == "review":
            self.usage["review_calls"] += 1
            call_number = self.usage["review_calls"]
        else:
            self.usage["model_calls"] += 1
            call_number = self.usage["model_calls"]
        try:
            observation.update(
                metadata={"model_call_ordinal": reservation or call_number}
            )
            mark_request_sent()
            response = await asyncio.wait_for(
                self.model.with_structured_output(
                    schema, include_raw=True, method="function_calling"
                ).ainvoke([("system", prompt), ("human", serialized)]),
                timeout=65,
            )
        except Exception as error:  # noqa: BLE001 -- provider boundary strips credential-bearing error text
            # Only the exception class and HTTP status, never the message: enough to tell
            # rate limits and outages from timeouts without leaking provider text.
            status = getattr(error, "status_code", None)
            observation.update(
                metadata={
                    "provider_error_class": type(error).__name__,
                    "provider_status": status if isinstance(status, int) else None,
                }
            )
            await self._settle(reservation, None, None)
            raise RuntimeError("model_call_failed") from None
        raw = response["raw"]
        usage = raw.usage_metadata
        reported = raw.response_metadata.get("model_name") or raw.response_metadata.get("model")
        await self._settle(reservation, usage, reported)
        observation.update(
            metadata={
                "usage": usage,
                "usage_missing": usage is None,
                "reported_model": reported,
                "model_call_number": call_number,
            }
        )
        if response["parsing_error"] or response["parsed"] is None:
            details = (
                output_errors(response["parsing_error"])
                if response["parsing_error"]
                else [
                    {
                        "loc": "(root)",
                        "type": "missing",
                        "msg": "no structured output; return the report only through the provided function",
                    }
                ]
            )
            # Shape of the reply only (counts and finish reason), never its text.
            content = raw.content if isinstance(raw.content, str) else str(raw.content)
            observation.update(
                metadata={
                    "output_validation_errors": details,
                    "raw_tool_calls": len(getattr(raw, "tool_calls", None) or []),
                    "raw_content_chars": len(content),
                    "finish_reason": raw.response_metadata.get("finish_reason"),
                }
            )
            raise StructuredOutputError(details)
        observation.update(output=response["parsed"])
        return response["parsed"]

    async def plan(self, state, tools, prompt):
        return await self._call(
            Plan,
            {
                "task": "plan",
                "request": state["request"].model_dump(mode="json"),
                "scope": state["scope"].model_dump(),
                "evidence": state["evidence"],
                "critique": state["critique"],
                "verification_issues": [
                    issue.model_dump() for issue in state["verification"][-1].issues
                ]
                if state.get("verification")
                else [],
                # The planner must know when a previous verdict already passed
                # verification; an empty issue list alone does not say so.
                # No-feedback ablation states arrive without a verdict key, so
                # both fields stay None there.
                "previous_verdict": (
                    state["verdict"].model_dump(mode="json")
                    if state.get("verification") and state.get("verdict")
                    else None
                ),
                "previous_verification": (
                    {
                        "passed": state["verification"][-1].passed,
                        "report_status": state["verification"][-1].report_status,
                    }
                    if state.get("verification")
                    else None
                ),
                "tool_failures": [
                    {"name": t.name, "error": t.error} for t in state["trace"] if not t.success
                ],
                "tools": list(tools.values()),
                **({"memory": state["memory"]} if state.get("memory") is not None else {}),
            },
            prompt,
        )

    async def hypothesize(self, state, prompt):
        return await self._call(
            DiagnosisReport,
            {
                "task": "hypothesize",
                "request": state["request"].model_dump(mode="json"),
                "evidence": state["evidence"],
                # The report author must see why the previous report failed; A0-A2 states
                # arrive with these cleared, so no feedback leaks into those ablations.
                "critique": state.get("critique", []),
                "verification_issues": [
                    issue.model_dump() for issue in state["verification"][-1].issues
                ]
                if state.get("verification")
                else [],
                **({"memory": state["memory"]} if state.get("memory") is not None else {}),
            },
            prompt,
        )

    async def review(self, state, issues, prompt):
        return await self._call(
            ModelReview,
            {
                "task": "verify",
                "request": state["request"].model_dump(mode="json"),
                "report": state["verdict"].model_dump(mode="json"),
                "evidence": state["evidence"],
                "rule_issues": [issue.model_dump() for issue in issues],
            },
            prompt,
            purpose="review",
        )
