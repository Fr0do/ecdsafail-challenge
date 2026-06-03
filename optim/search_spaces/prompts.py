"""Text slots for optional GEPA/hybrid research-loop evolution."""

from __future__ import annotations

from evomcp.pipeline.registry import DEFAULT_REGISTRY, TextSlot


DEFAULT_REGISTRY.register_text(
    TextSlot(
        name="failure_triage_prompt",
        role="ecdsa_fail_research_triage",
        seed_value=(
            "Classify a failed ecdsa.fail candidate by whether it is likely a "
            "Fiat-Shamir island miss, a width-envelope violation, a phase-cleanliness "
            "bug, or a true arithmetic rewrite bug. Prefer minimal next experiments."
        ),
        description="GEPA-mutatable triage instructions for analyzing failed trace bundles.",
        max_chars=2000,
    )
)
