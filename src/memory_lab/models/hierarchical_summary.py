from __future__ import annotations

from collections import defaultdict
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


class HierarchicalSummaryMemory(BaseMemoryModel):
    name = "hierarchical_summary"

    def ingest(
        self,
        state: MemoryState,
        events: Sequence[MemoryEvent],
    ) -> MemoryState:
        state, _new_events = self._append_new_events(state, events)
        items = tuple(self._item_from_event(event) for event in state.events)
        return state.replace_items(items)

    def select_context(self, state: MemoryState, query: MemoryQuery) -> ContextPacket:
        grouped: dict[str, list[MemoryItem]] = defaultdict(list)
        for item in state.items:
            if query.task_id and item.task_id not in {None, query.task_id}:
                continue
            if query.worker_id and item.worker_id not in {None, query.worker_id}:
                continue
            grouped[self._group_key(item)].append(item)

        sections: list[ContextSection] = []
        for key in sorted(grouped):
            items = grouped[key]
            section_id = f"hierarchy_{key}"
            title = f"Hierarchy: {key}"
            kind = (
                SectionKind.TASK_STATE
                if key.startswith("task:")
                else SectionKind.MEMORY
            )
            content = "\n".join(item.content for item in items)
            item_ids = tuple(item.id for item in items)
            event_id_values = (
                event_id
                for item in items
                for event_id in item.provenance.event_ids
            )
            event_ids = unique_tuple(event_id_values)
            rows = [item.to_dict() for item in items]
            metadata = {"group": key, "rows": rows}

            sections.append(
                ContextSection(
                    id=section_id,
                    title=title,
                    kind=kind,
                    trust=TrustLevel.UNKNOWN,
                    content=content,
                    item_ids=item_ids,
                    event_ids=event_ids,
                    metadata=metadata,
                ),
            )
        return ContextPacket(
            query=query,
            sections=tuple(sections),
            metadata={"model": self.name, "group_count": len(sections)},
        )

    def _item_from_event(self, event: MemoryEvent) -> MemoryItem:
        return MemoryItem(
            id=stable_id("mem", self.name, event.id),
            kind=MemoryItemKind.TASK_STATE if event.task_id else MemoryItemKind.TRANSCRIPT,
            content=format_event(event),
            provenance=Provenance(event_ids=(event.id,)),
            trust=event.trust,
            task_id=event.task_id,
            worker_id=event.worker_id,
            metadata={"event_kind": event.kind.value, "group": self._event_group_key(event)},
        )

    def _group_key(self, item: MemoryItem) -> str:
        if item.worker_id:
            return f"worker:{item.worker_id}"
        if item.task_id:
            return f"task:{item.task_id}"
        return "root"

    def _event_group_key(self, event: MemoryEvent) -> str:
        if event.worker_id:
            return f"worker:{event.worker_id}"
        if event.task_id:
            return f"task:{event.task_id}"
        return "root"
