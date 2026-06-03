"""EvoX program slots for the ecdsa.fail point-addition benchmark."""

from __future__ import annotations

from evomcp.pipeline.registry import DEFAULT_REGISTRY, ProgSlot, SlotKind


def _intish(value: object) -> bool:
    try:
        int(round(float(value)))
    except (TypeError, ValueError):
        return False
    return True


for slot in (
    ProgSlot(
        name="DIALOG_GCD_COMPARE_BITS",
        kind=SlotKind.CATEGORICAL,
        default=57,
        choices=(55, 56, 57, 58, 59),
        description="Truncated branch-comparator width; lower saves Toffoli but needs clean reroll islands.",
    ),
    ProgSlot(
        name="DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS",
        kind=SlotKind.CATEGORICAL,
        default=19,
        choices=(18, 19, 20, 21),
        description="Apply-phase correction comparator width.",
    ),
    ProgSlot(
        name="DIALOG_GCD_ACTIVE_ITERATIONS",
        kind=SlotKind.CATEGORICAL,
        default=395,
        choices=(393, 394, 395, 396, 397, 398),
        description="Dialog-GCD active iteration count; shorter can save width/Toffoli but may fail support.",
    ),
    ProgSlot(
        name="DIALOG_GCD_WIDTH_MARGIN",
        kind=SlotKind.CATEGORICAL,
        default=26,
        choices=(24, 25, 26, 27, 28),
        description="Active-width safety margin for the variable-width GCD body.",
    ),
    ProgSlot(
        name="DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN",
        kind=SlotKind.CATEGORICAL,
        default=7,
        choices=(5, 6, 7, 8, 9),
        description="PA9024 compare-schedule truncation margin.",
    ),
    ProgSlot(
        name="DIALOG_GCD_APPLY_CHUNKED_F_CUT",
        kind=SlotKind.CATEGORICAL,
        default=124,
        choices=(116, 118, 120, 122, 124, 126, 128, 130),
        description="First chunk boundary for materialized apply add/sub.",
    ),
    ProgSlot(
        name="DIALOG_GCD_APPLY_CHUNKED_F_CUT2",
        kind=SlotKind.CATEGORICAL,
        default=130,
        choices=(126, 128, 130, 132, 134, 136),
        description="Second chunk boundary for 3-block materialized apply add/sub.",
    ),
    ProgSlot(
        name="DIALOG_REROLL",
        kind=SlotKind.CATEGORICAL,
        default=6458,
        choices=(0, 1, 3, 7, 17, 37, 40, 52, 118, 2553, 4959, 5983, 6458),
        description="Fiat-Shamir reroll knob; seeded with known clean-island values from memory.",
    ),
    ProgSlot(
        name="DIALOG_POST_SUB_REROLL",
        kind=SlotKind.CATEGORICAL,
        default=2553,
        choices=(0, 1, 10, 13, 28, 44, 51, 56, 118, 2553, 5983),
        description="Second Fiat-Shamir reroll knob; seeded with known clean-island values from memory.",
    ),
):
    DEFAULT_REGISTRY.register_prog(slot)
