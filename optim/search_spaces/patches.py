"""Patch-id templates for known ecdsa.fail route islands."""

from __future__ import annotations

from evomcp.pipeline.registry import DEFAULT_REGISTRY, ProgSlot, SlotKind


PATCHES: dict[str, dict[str, object]] = {
    "current_1434": {
        "DIALOG_GCD_COMPARE_BITS": 57,
        "DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS": 19,
        "DIALOG_GCD_ACTIVE_ITERATIONS": 395,
        "DIALOG_GCD_WIDTH_MARGIN": 26,
        "DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN": 7,
        "DIALOG_GCD_APPLY_CHUNKED_F_CUT": 124,
        "DIALOG_GCD_APPLY_CHUNKED_F_CUT2": 130,
        "DIALOG_REROLL": 6458,
        "DIALOG_POST_SUB_REROLL": 2553,
    },
    "b343_sm5_1434": {
        "DIALOG_GCD_COMPARE_BITS": 57,
        "DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS": 19,
        "DIALOG_GCD_ACTIVE_ITERATIONS": 395,
        "DIALOG_GCD_WIDTH_MARGIN": 26,
        "DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN": 5,
        "DIALOG_GCD_APPLY_CHUNKED_F_CUT": 124,
        "DIALOG_GCD_APPLY_CHUNKED_F_CUT2": 130,
        "DIALOG_REROLL": 1844,
        "DIALOG_POST_SUB_REROLL": 3532,
    },
}


for patch_id, env in PATCHES.items():
    DEFAULT_REGISTRY.register_patch_env(patch_id, env)

DEFAULT_REGISTRY.register_prog(
    ProgSlot(
        name="patch_id",
        kind=SlotKind.PATCH,
        default="current_1434",
        choices=tuple(PATCHES),
        description="Named route template applied before individual knob overrides.",
    )
)
