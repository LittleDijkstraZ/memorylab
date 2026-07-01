from __future__ import annotations

from memory_lab.recipe import (
    MemoryLayerSpec,
    MemoryRecipe,
    MemoryRecipeOperation,
    MemorySelector,
)
from memory_lab.schema import SectionKind, TrustLevel


def one_pot_compression(
    *,
    selector: MemorySelector | None = None,
    budget_tokens: int | None = None,
) -> MemoryRecipe:
    return MemoryRecipe(
        name="one_pot_compression",
        description="Compress a selected memory scope into one auditable block.",
        selector=selector or MemorySelector(),
        renderer="compact_prompt",
        layers=(
            MemoryLayerSpec(
                id="compressed_context",
                title="Compressed Context",
                operation=MemoryRecipeOperation.COMPRESS,
                instruction=(
                    "Keep the important goals, decisions, conclusions, "
                    "observations, and open questions from the selected scope."
                ),
                budget_tokens=budget_tokens,
            ),
        ),
    )


def layered_checkpoint(
    *,
    selector: MemorySelector | None = None,
    high_level_budget: int | None = 8000,
    evidence_budget: int | None = 32000,
    recent_event_limit: int = 6,
) -> MemoryRecipe:
    source_trust = (
        TrustLevel.SOURCE_OBSERVATION.value,
        TrustLevel.EXTRACTED_EVIDENCE.value,
    )
    return MemoryRecipe(
        name="layered_checkpoint",
        description=(
            "Build a layered task-boundary checkpoint with summary, evidence, "
            "recent raw context, and deterministic unresolved markers."
        ),
        selector=selector or MemorySelector(),
        renderer="layered_prompt",
        layers=(
            MemoryLayerSpec(
                id="high_level",
                title="High Level",
                operation=MemoryRecipeOperation.COMPRESS,
                instruction=(
                    "Keep goals, decisions, conclusions, and known open "
                    "questions. Do not invent evidence."
                ),
                budget_tokens=high_level_budget,
            ),
            MemoryLayerSpec(
                id="evidence",
                title="Evidence",
                operation=MemoryRecipeOperation.RETRIEVE,
                selector=MemorySelector(trust=source_trust),
                instruction="Keep source-bearing observations.",
                budget_tokens=evidence_budget,
                section_kind=SectionKind.EVIDENCE,
                section_trust=TrustLevel.SOURCE_OBSERVATION,
                metadata={"trust_policy": "source_or_extracted_evidence"},
            ),
            MemoryLayerSpec(
                id="recent",
                title="Recent Work",
                operation=MemoryRecipeOperation.KEEP_RAW,
                selector=MemorySelector(recent_event_limit=recent_event_limit),
            ),
            MemoryLayerSpec(
                id="unresolved",
                title="Unresolved",
                operation=MemoryRecipeOperation.KEEP_RAW,
                selector=MemorySelector(metadata_exists=("missing_evidence",)),
                instruction=(
                    "Deterministically surface explicit missing evidence or "
                    "gap markers only."
                ),
                metadata={"limited": True},
            ),
        ),
    )


def retrieval_context(
    *,
    selector: MemorySelector | None = None,
    budget_tokens: int | None = None,
) -> MemoryRecipe:
    return MemoryRecipe(
        name="retrieval_context",
        description="Retrieve relevant memory without compression.",
        selector=selector or MemorySelector(),
        renderer="compact_prompt",
        layers=(
            MemoryLayerSpec(
                id="retrieved_context",
                title="Retrieved Context",
                operation=MemoryRecipeOperation.RETRIEVE,
                budget_tokens=budget_tokens,
            ),
        ),
    )


__all__ = [
    "layered_checkpoint",
    "one_pot_compression",
    "retrieval_context",
]
