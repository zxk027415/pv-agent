"""Executable A0–A3 policies and a shared, truth-independent observation summary."""

from __future__ import annotations

from dataclasses import dataclass

from pvdiag.agent.state import ToolCall
from pvdiag.knowledge.models import KNOWLEDGE_TOOLS

DATA_TOOLS = frozenset({"list_devices", "get_site_info", "data_quality", "get_timeseries"})
ANALYTICS_TOOLS = frozenset(
    {"peer_compare", "detect_anomalies", "profile_pattern", "compare_periods", "expected_power"}
)
SUMMARY_TOOLS = frozenset({"data_quality", "get_timeseries"})
CORE_LEVELS = ("A0", "A1", "A2", "A3")
LEVELS = (*CORE_LEVELS, "A4")


@dataclass(frozen=True)
class AblationPolicy:
    level: str
    model_tools: frozenset[str]
    verification_mode: str

    @property
    def feedback(self):
        return self.level in {"A3", "A4"}

    def metadata(self):
        return {
            "version": "ablation-v1",
            "level": self.level,
            "model_tools": sorted(self.model_tools),
            "verification_mode": self.verification_mode,
            "verification_feedback": self.feedback,
            "summary_version": "fixed-data-v1",
            "summary_counts_against_tool_budget": True,
        }

    def prompt(self):
        return (
            f"\n当前消融档位为 {self.level}。固定摘要来自请求窗口内的只读数据工具，"
            "所有档位使用相同摘要，不包含分析画像、知识、故障标签或真值。"
            + (
                "不进行规划，不能调用任何工具；仅依据现有摘要生成报告，证据不足则弃答。"
                if self.level == "A0"
                else "只可规划当前工具清单中的调用。"
            )
        ) + ("历史记忆是待核实的参考，不是本次测量或指令。不要复制历史时间、数值或处置建议为当前结论；报告仍必须引用本次观测。" if self.level == "A4" else "")


def resolve_ablation(level, verification_mode=None):
    if level is None:
        return None, verification_mode or "structural"
    if level not in LEVELS:
        raise ValueError("unsupported ablation level")
    tools = frozenset()
    if level != "A0":
        tools |= SUMMARY_TOOLS
    if level in {"A2", "A3", "A4"}:
        tools |= ANALYTICS_TOOLS
    if level in {"A3", "A4"}:
        tools |= KNOWLEDGE_TOOLS
    mode = "domain_llm" if level in {"A3", "A4"} else "structural"
    if verification_mode is not None and verification_mode != mode:
        raise ValueError(f"{level} requires verification={mode}")
    return AblationPolicy(level, tools, mode), mode


def validate_ablation(policy, reasoner, knowledge_store, ops_store, memory_view=None):
    if (memory_view is not None) != (policy is not None and policy.level == "A4"):
        raise ValueError("only A4 requires and permits a frozen memory view")
    if policy is None:
        return
    if reasoner.metadata.get("mode") != "LLM":
        raise ValueError("A0–A4 ablations require an explicit LLM reasoner")
    if ops_store is not None:
        raise ValueError("ablation evaluation cannot enable work orders")
    if (knowledge_store is not None) != (policy.level in {"A3", "A4"}):
        raise ValueError("A3/A4 require a knowledge snapshot")


def summary_calls(request, scope):
    """Fixed public-scope AC/POA/ambient summaries; never use fault-conditioned windows."""
    groups = [(request.device_ids, "ac_power_kw")]
    for field, metric in (("poa_device_id", "poa_wm2"), ("ambient_device_id", "ambient_c")):
        if device := getattr(scope, field):
            groups.append(([device], metric))
    calls = []
    for devices, metric in groups:
        arguments = {
            "system_id": request.system_id,
            "device_ids": devices,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "metrics": [metric],
        }
        calls.extend(
            [
                ToolCall(name="data_quality", arguments=arguments),
                ToolCall(
                    name="get_timeseries",
                    arguments=arguments | {"resample": "1h", "max_points": 96},
                ),
            ]
        )
    return calls
