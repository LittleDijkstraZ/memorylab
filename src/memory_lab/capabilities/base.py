from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memory_lab.recipe import MemorySelector
from memory_lab.schema import MemoryEvent, MemoryItem, MemoryQuery, MemoryState


@dataclass(frozen=True)
class CompressionRequest:
    layer_id: str
    events: tuple[MemoryEvent, ...]
    items: tuple[MemoryItem, ...]
    instruction: str
    budget_tokens: int | None
    query: MemoryQuery


@dataclass(frozen=True)
class RetrievalRequest:
    layer_id: str
    state: MemoryState
    selector: MemorySelector
    query: MemoryQuery
    budget_tokens: int | None


@dataclass(frozen=True)
class CapabilityResponse:
    content: str
    event_ids: tuple[str, ...] = ()
    item_ids: tuple[str, ...] = ()
    omitted_count: int = 0
    omitted_reason: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


class Compressor(Protocol):
    name: str

    def compress(self, request: CompressionRequest) -> CapabilityResponse:
        ...


class Retriever(Protocol):
    name: str

    def retrieve(self, request: RetrievalRequest) -> CapabilityResponse:
        ...


__all__ = [
    "CapabilityResponse",
    "CompressionRequest",
    "Compressor",
    "RetrievalRequest",
    "Retriever",
]
