"""Structured LangChain model adapter; constructing it performs no API calls."""

from __future__ import annotations

import asyncio
import json

from langchain.chat_models import init_chat_model

from pvdiag.agent.state import DiagnosisReport, Plan


class LLMReasoner:
    def __init__(
        self, model: str, *, provider="openai", base_url=None, temperature=0.0, api_key=None
    ):
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
        }
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 0,
            "usage_missing_calls": 0,
        }

    async def _call(self, schema, payload, prompt):
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(serialized) > 80_000:
            raise ValueError("model context exceeds configured character budget")
        self.usage["model_calls"] += 1
        try:
            response = await asyncio.wait_for(
                self.model.with_structured_output(
                    schema, include_raw=True, method="function_calling"
                ).ainvoke([("system", prompt), ("human", serialized)]),
                timeout=65,
            )
        except Exception:  # noqa: BLE001 -- provider boundary strips credential-bearing error text
            self.usage["usage_missing_calls"] += 1
            raise RuntimeError("model_call_failed") from None
        raw = response["raw"]
        usage = raw.usage_metadata
        if usage:
            for key in ("input_tokens", "output_tokens"):
                self.usage[key] += usage.get(key, 0)
        else:
            self.usage["usage_missing_calls"] += 1
        reported = raw.response_metadata.get("model_name") or raw.response_metadata.get("model")
        if reported and reported not in self.metadata["reported_models"]:
            self.metadata["reported_models"].append(reported)
        if response["parsing_error"] or response["parsed"] is None:
            raise ValueError("invalid structured model output")
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
                "tool_failures": [
                    {"name": t.name, "error": t.error} for t in state["trace"] if not t.success
                ],
                "tools": list(tools.values()),
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
            },
            prompt,
        )
