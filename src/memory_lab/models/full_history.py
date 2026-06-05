from __future__ import annotations

from typing import Iterable, Sequence

from memory_lab.models.base import BaseMemoryModel
from memory_lab.schema import (
    ContextPacket,
    ContextSection,
    EventKind,
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


TASK_EVENT_KINDS = {
    EventKind.RUN_STARTED,
    EventKind.TASK_PLAN_CREATED,
    EventKind.TASK_COMPLETED,
    EventKind.WORKER_CONTEXT_SENT,
}


def format_event(event: MemoryEvent) -> str:
    text = event.text()
    label = event.kind.value
    if event.task_id:
        label = f"{label} task={event.task_id}"
    if event.worker_id:
        label = f"{label} worker={event.worker_id}"
    if not text:
        text = "(empty)"
    return f"- [{event.id}] {label}: {text}"


def item_from_event(event: MemoryEvent) -> MemoryItem:
    if event.kind == EventKind.CONTROL_STATE:
        kind = MemoryItemKind.CONTROL
    elif event.kind in TASK_EVENT_KINDS:
        kind = MemoryItemKind.TASK_STATE
    elif event.kind == EventKind.TOOL_RESULT:
        kind = MemoryItemKind.OBSERVATION
    elif event.kind == EventKind.EVIDENCE_REVIEW:
        kind = MemoryItemKind.EVIDENCE
    elif event.kind in {EventKind.SUMMARY_CREATED, EventKind.FINAL_REPORT_CREATED}:
        kind = MemoryItemKind.SYNTHESIS
    else:
        kind = MemoryItemKind.TRANSCRIPT
    return MemoryItem(
        id=stable_id("mem", "event", event.id),
        kind=kind,
        content=event.text(),
        provenance=Provenance(
            event_ids=(event.id,),
            source_refs=event.provenance.source_refs,
            artifact_refs=event.provenance.artifact_refs,
        ),
        trust=event.trust,
        task_id=event.task_id,
        worker_id=event.worker_id,
        created_at=event.timestamp,
        metadata={
            "event_kind": event.kind.value,
            "tags": list(event.tags),
        },
    )


class FullHistoryMemory(BaseMemoryModel):
    name = "full_history"

    def ingest(
        self,
        state: MemoryState,
        events: Sequence[MemoryEvent],
    ) -> MemoryState:
        state, new_events = self._append_new_events(state, events)
        return state.with_items(tuple(item_from_event(event) for event in new_events))

    def select_context(self, state: MemoryState, query: MemoryQuery) -> ContextPacket:
        task_items = tuple(
            item
            for item in state.items
            if item.kind == MemoryItemKind.TASK_STATE and self._matches_query(item, query)
        )
        control_items = tuple(
            item
            for item in state.items
            if item.kind == MemoryItemKind.CONTROL and self._matches_query(item, query)
        )
        history_items = tuple(
            item
            for item in state.items
            if item.kind
            not in {
                MemoryItemKind.CONTROL,
                MemoryItemKind.TASK_STATE,
            }
            and self._matches_query(item, query)
        )

        sections: list[ContextSection] = []
        if task_items:
            sections.append(
                self._section_from_items(
                    "task_state",
                    "Task State",
                    SectionKind.TASK_STATE,
                    TrustLevel.SYSTEM_CONTROL,
                    task_items,
                ),
            )
        if history_items:
            sections.append(
                self._section_from_items(
                    "work_history",
                    "Work History",
                    SectionKind.MEMORY,
                    TrustLevel.UNKNOWN,
                    history_items,
                ),
            )
        if query.include_control and control_items:
            sections.append(
                self._section_from_items(
                    "control_state",
                    "Control State",
                    SectionKind.CONTROL,
                    TrustLevel.SYSTEM_CONTROL,
                    control_items[-1:],
                    omitted_count=max(0, len(control_items) - 1),
                    omitted_reason="latest-control-state-only",
                ),
            )

        return ContextPacket(
            query=query,
            sections=tuple(sections),
            metadata={
                "model": self.name,
                "event_count": len(state.events),
                "item_count": len(state.items),
            },
        )

    def _section_from_items(
        self,
        section_id: str,
        title: str,
        kind: SectionKind,
        trust: TrustLevel,
        items: Iterable[MemoryItem],
        *,
        omitted_count: int = 0,
        omitted_reason: str | None = None,
    ) -> ContextSection:
        item_tuple = tuple(items)
        event_id_values = (
            event_id
            for item in item_tuple
            for event_id in item.provenance.event_ids
        )
        event_ids = unique_tuple(event_id_values)
        item_ids = tuple(item.id for item in item_tuple)
        content = "\n".join(
            f"- [{item.id}] {item.metadata.get('event_kind', item.kind.value)}: {item.content}"
            for item in item_tuple
        )
        rows = [item.to_dict() for item in item_tuple]
        metadata = {"rows": rows}

        return ContextSection(
            id=section_id,
            title=title,
            kind=kind,
            trust=trust,
            content=content,
            item_ids=item_ids,
            event_ids=event_ids,
            omitted_count=omitted_count,
            omitted_reason=omitted_reason,
            metadata=metadata,
        )

    def _matches_query(self, item: MemoryItem, query: MemoryQuery) -> bool:
        if query.task_id and item.task_id not in {None, query.task_id}:
            return False
        if query.worker_id and item.worker_id not in {None, query.worker_id}:
            return False
        if not query.include_model_derived and item.trust not in {
            TrustLevel.SOURCE_OBSERVATION,
            TrustLevel.EXTRACTED_EVIDENCE,
            TrustLevel.SYSTEM_CONTROL,
        }:
            return False
        return True
