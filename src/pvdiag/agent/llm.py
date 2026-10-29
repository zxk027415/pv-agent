"""Structured LangChain model adapter; constructing it performs no API calls."""

from __future__ import annotations

import asyncio
import hashlib
import json

from langchain.chat_models import init_chat_model

from pvdiag.agent.state import DiagnosisReport, ModelReview, Plan
from pvdiag.observability.tracing import mark_request_sent, observe


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
        call_tracker=None,
    ):
        if not 1 <= max_model_calls <= 30:
            raise ValueError("invalid model call budget")
        if base_url and api_key is None:
            raise ValueError(
                "custom endpoints require an explicit API key; no implicit credential fallback"
            )
        self.max_model_calls = max_model_calls
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
            "usage_missing_calls": 0,
        }

    async def sync_accounting(self):
        if self.call_tracker is not None:
            try:
                self.usage, models = await self.call_tracker.snapshot()
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

    async def _call(self, schema, payload, prompt):
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
            return await self._invoke(schema, payload, prompt, observation)

    async def _invoke(self, schema, payload, prompt, observation):
        if self.usage["model_calls"] >= self.max_model_calls:
            raise ValueError("model_call_budget_exhausted")
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(serialized) > 80_000:
            raise ValueError("model context exceeds configured character budget")
        reservation = None
        if self.call_tracker is not None:
            try:
                reservation = await self.call_tracker.reserve()
            except ValueError:
                raise
            except Exception:  # noqa: BLE001 -- redact all storage exception details
                raise RuntimeError("model_accounting_unavailable") from None
        self.usage["model_calls"] += 1
        try:
            observation.update(
                metadata={"model_call_ordinal": reservation or self.usage["model_calls"]}
            )
            mark_request_sent()
            response = await asyncio.wait_for(
                self.model.with_structured_output(
                    schema, include_raw=True, method="function_calling"
                ).ainvoke([("system", prompt), ("human", serialized)]),
                timeout=65,
            )
        except Exception:  # noqa: BLE001 -- provider boundary strips credential-bearing error text
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
                "model_call_number": self.usage["model_calls"],
            }
        )
        if response["parsing_error"] or response["parsed"] is None:
            raise ValueError("invalid structured model output")
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
        )
