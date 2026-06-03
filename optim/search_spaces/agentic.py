"""Agentic task slots for ecdsa.fail local evolution."""

from __future__ import annotations

from evomcp.pipeline.registry import DEFAULT_REGISTRY, ProgSlot, SlotKind


for slot in (
    ProgSlot(
        name="agent_task_id",
        kind=SlotKind.CATEGORICAL,
        default="reduce_peak_qubits",
        choices=("reduce_peak_qubits", "reduce_toffoli", "reroll_clean_island", "phase_garbage_guard"),
        description="Which ecdsa.fail attack/optimization hypothesis the local agent should propose.",
    ),
    ProgSlot(
        name="agent_model",
        kind=SlotKind.CATEGORICAL,
        default="haiku",
        choices=(
            "haiku",
            "sonnet",
            "default",
            "qwen/qwen3.7-max",
            "xiaomi/mimo-v2.5-pro",
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
            "tencent/hy3-preview",
            "minimax/minimax-m3",
        ),
        description="Claude/Codex alias or explicit OpenRouter model id.",
    ),
    ProgSlot(
        name="agent_model_lane",
        kind=SlotKind.CATEGORICAL,
        default="best_value",
        choices=(
            "cheap_fast",
            "cheap_explorer",
            "best_value",
            "strong",
            "max_quality",
            "long_context",
        ),
        description="OpenRouter routing prior: fast/cheap/strong/long-context scenario.",
    ),
    ProgSlot(
        name="agent_mode",
        kind=SlotKind.CATEGORICAL,
        default="plan_only",
        choices=("plan_only",),
        description="Local agent mode; patch_eval will run in isolated worktrees later.",
    ),
):
    DEFAULT_REGISTRY.register_prog(slot)
