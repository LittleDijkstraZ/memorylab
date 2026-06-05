from __future__ import annotations

from typing import Sequence

from memory_lab.models.base import BaseMemoryModel
from memory_lab.models.full_history import format_event
from memory_lab.schema import (
    ContextPacket,
    ContextSection,
    MemoryEvent,
    MemoryItem,
    MemoryItemKind,
    MemoryQuery,
    MemoryState,
    Provenance,
    SectionKind,
    TrustLevel,
    stable_id,
    unique_tuple,
)


class RollingSummaryMemory(BaseMemoryModel):
    name = "rolling_summary"

    def __init__(self, *, recent_event_limit: int = 6) -> None:
        if recent_event_limit < 1:
            raise ValueError("recent_event_limit must be at least 1")
        self.recent_event_limit = recent_event_limit

    def ingest(
        self,
        state: MemoryState,
        events: Sequence[MemoryEvent],
    ) -> MemoryState:
        state, _new_events = self._append_new_events(state, events)
        older = state.events[:-self.recent_event_limit]
        summary_items: list[MemoryItem] = []
        if older:
            content = "\n".join(format_event(event) for event in older)
            event_ids = tuple(event.id for event in older)
            summary_items.append(
                MemoryItem(
                    id=stable_id("mem", self.name, "older", *event_ids),
                    kind=MemoryItemKind.SUMMARY,
                    content=content,
                    provenance=Provenance(
                        event_ids=event_ids,
                        note="deterministic rolling summary over older events",
                    ),
                    trust=TrustLevel.MODEL_DERIVED,
                    metadata={"summary_event_count": len(older)},
                ),
            )
        recent_items = [
            MemoryItem(
                id=stable_id("mem", self.name, "recent", event.id),
                kind=MemoryItemKind.TRANSCRIPT,
                content=format_event(event),
                provenance=Provenance(event_ids=(event.id,)),
                trust=event.trust,
                task_id=event.task_id,
                worker_id=event.worker_id,
                metadata={"event_kind": event.kind.value},
            )
            for event in state.events[-self.recent_event_limit :]
        ]
        return state.replace_items(tuple(summary_items + recent_items))

    def select_context(self, state: MemoryState, query: MemoryQuery) -> ContextPacket:
        sections: list[ContextSection] = []
        summary_items = tuple(item for item in state.items if item.kind == MemoryItemKind.SUMMARY)
        recent_items = tuple(item for item in state.items if item.kind != MemoryItemKind.SUMMARY)

        if summary_items:
            sections.append(self._section("rolling_summary", "Rolling Summary", SectionKind.SUMMARY, summary_items))
        if recent_items:
            sections.append(self._section("recent_events", "Recent Events", SectionKind.MEMORY, recent_items))
        return ContextPacket(
            query=query,
            sections=tuple(sections),
            metadata={
                "model": self.name,
                "recent_event_limit": self.recent_event_limit,
            },
        )

    def _section(
        self,
        section_id: str,
        title: str,
        kind: SectionKind,
        items: Sequence[MemoryItem],
    ) -> ContextSection:
        trust = items[0].trust if items else TrustLevel.UNKNOWN
        content = "\n".join(item.content for item in items)
        item_ids = tuple(item.id for item in items)
        event_id_values = (
            event_id
            for item in items
            for event_id in item.provenance.event_ids
        )
        event_ids = unique_tuple(event_id_values)
        rows = [item.to_dict() for item in items]
        metadata = {"rows": rows}

        return ContextSection(
            id=section_id,
            title=title,
            kind=kind,
            trust=trust,
            content=content,
            item_ids=item_ids,
            event_ids=event_ids,
            metadata=metadata,
        )
