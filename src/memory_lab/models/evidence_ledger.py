from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from memory_lab.models.base import BaseMemoryModel
from memory_lab.schema import (
    ContextPacket,
    ContextSection,
    EvidenceStatus,
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


STATUS_ALIASES = {
    "direct": EvidenceStatus.SUPPORTED,
    "support": EvidenceStatus.SUPPORTED,
    "supported": EvidenceStatus.SUPPORTED,
    "partial": EvidenceStatus.PARTIAL,
    "partially_supported": EvidenceStatus.PARTIAL,
    "contradiction": EvidenceStatus.CONTRADICTORY,
    "contradictory": EvidenceStatus.CONTRADICTORY,
    "near_miss": EvidenceStatus.PARTIAL,
    "duplicate": EvidenceStatus.DUPLICATE,
    "irrelevant": EvidenceStatus.IRRELEVANT,
    "missing": EvidenceStatus.MISSING,
}


PRIMARY_STATUS_ORDER = (
    EvidenceStatus.SUPPORTED,
    EvidenceStatus.PARTIAL,
    EvidenceStatus.CONTRADICTORY,
    EvidenceStatus.MISSING,
    EvidenceStatus.DUPLICATE,
    EvidenceStatus.IRRELEVANT,
    EvidenceStatus.UNKNOWN,
)


def _coerce_status(value: Any, default: EvidenceStatus = EvidenceStatus.UNKNOWN) -> EvidenceStatus:
    if isinstance(value, EvidenceStatus):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return STATUS_ALIASES.get(text, EvidenceStatus(text) if text in EvidenceStatus._value2member_map_ else default)


def _as_entries(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(entry for entry in value if isinstance(entry, Mapping))
    return ()


def _first_text(entry: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _source_refs(entry: Mapping[str, Any], event: MemoryEvent) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("source_refs", "sources", "urls", "source_urls"):
        raw = entry.get(key)
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, (list, tuple)):
            values.extend(str(item) for item in raw)
    for key in ("url", "source_url", "source"):
        if entry.get(key):
            values.append(str(entry[key]))
    values.extend(event.provenance.source_refs)
    return unique_tuple(values)


def _artifact_refs(entry: Mapping[str, Any], event: MemoryEvent) -> tuple[str, ...]:
    values: list[str] = []
    raw = entry.get("artifact_refs") or entry.get("artifacts")
    if isinstance(raw, str):
        values.append(raw)
    elif isinstance(raw, (list, tuple)):
        values.extend(str(item) for item in raw)
    values.extend(event.provenance.artifact_refs)
    return unique_tuple(values)


def _claim_slot(entry: Mapping[str, Any]) -> str | None:
    return (
        entry.get("claim_slot")
        or entry.get("slot")
        or entry.get("slot_id")
        or entry.get("evidence_slot")
    )


def _confidence(entry: Mapping[str, Any]) -> float | None:
    value = entry.get("confidence")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _entry_content(entry: Mapping[str, Any]) -> str:
    return _first_text(
        entry,
        "content",
        "text",
        "summary",
        "observation",
        "claim",
        "description",
    )


class EvidenceLedgerMemory(BaseMemoryModel):
    name = "evidence_ledger"

    def ingest(
        self,
        state: MemoryState,
        events: Sequence[MemoryEvent],
    ) -> MemoryState:
        state, new_events = self._append_new_events(state, events)
        items: list[MemoryItem] = []
        for event in new_events:
            items.extend(self._items_from_event(event))
        return state.with_items(tuple(items))

    def select_context(self, state: MemoryState, query: MemoryQuery) -> ContextPacket:
        relevant = tuple(item for item in state.items if self._matches_query(item, query))
        primary = tuple(
            item
            for item in relevant
            if item.is_primary_evidence or item.kind == MemoryItemKind.MISSING_EVIDENCE
        )
        grouped: dict[EvidenceStatus, list[MemoryItem]] = defaultdict(list)
        for item in primary:
            grouped[item.evidence_status].append(item)

        sections: list[ContextSection] = []
        for status in PRIMARY_STATUS_ORDER:
            items = tuple(grouped.get(status, ()))
            if not items:
                continue
            sections.append(self._evidence_section(status, items))

        synthesis_items = tuple(
            item
            for item in relevant
            if item.kind in {MemoryItemKind.SYNTHESIS, MemoryItemKind.SUMMARY}
        )
        warnings: list[str] = []
        if synthesis_items and not query.include_synthesis:
            warnings.append(
                "model-derived summaries/final reports were excluded from primary evidence",
            )
        if query.include_synthesis and synthesis_items:
            sections.append(
                self._generic_section(
                    "synthesis_hints",
                    "Synthesis Hints",
                    SectionKind.SUMMARY,
                    TrustLevel.FINAL_SYNTHESIS,
                    synthesis_items,
                ),
            )

        return ContextPacket(
            query=query,
            sections=tuple(sections),
            warnings=tuple(warnings),
            metadata={
                "model": self.name,
                "event_count": len(state.events),
                "item_count": len(state.items),
                "primary_evidence_count": len(primary),
                "synthesis_hint_count": len(synthesis_items),
            },
        )

    def _items_from_event(self, event: MemoryEvent) -> tuple[MemoryItem, ...]:
        if event.kind == EventKind.FINAL_REPORT_CREATED:
            return (
                MemoryItem(
                    id=stable_id("mem", "final", event.id),
                    kind=MemoryItemKind.SYNTHESIS,
                    content=event.text(),
                    provenance=Provenance(
                        event_ids=(event.id,),
                        source_refs=event.provenance.source_refs,
                        artifact_refs=event.provenance.artifact_refs,
                        note="final reports are context hints, not primary evidence",
                    ),
                    trust=TrustLevel.FINAL_SYNTHESIS,
                    task_id=event.task_id,
                    worker_id=event.worker_id,
                    created_at=event.timestamp,
                    metadata={"event_kind": event.kind.value},
                ),
            )
        if event.kind == EventKind.SUMMARY_CREATED:
            return (
                MemoryItem(
                    id=stable_id("mem", "summary", event.id),
                    kind=MemoryItemKind.SUMMARY,
                    content=event.text(),
                    provenance=Provenance(event_ids=(event.id,)),
                    trust=TrustLevel.FINAL_SYNTHESIS,
                    task_id=event.task_id,
                    worker_id=event.worker_id,
                    created_at=event.timestamp,
                    metadata={"event_kind": event.kind.value},
                ),
            )

        items: list[MemoryItem] = []
        evidence_entries = self._evidence_entries_for_event(event)
        for index, entry in enumerate(evidence_entries):
            content = _entry_content(entry)
            if not content:
                continue
            status = _coerce_status(
                entry.get("status") or entry.get("relevance") or entry.get("label"),
                EvidenceStatus.SUPPORTED
                if event.kind in {EventKind.TOOL_RESULT, EventKind.EVIDENCE_REVIEW}
                else EvidenceStatus.UNKNOWN,
            )
            item_kind = (
                MemoryItemKind.MISSING_EVIDENCE
                if status == EvidenceStatus.MISSING
                else MemoryItemKind.EVIDENCE
            )
            trust = (
                TrustLevel.EXTRACTED_EVIDENCE
                if event.kind in {EventKind.EVIDENCE_REVIEW, EventKind.TASK_COMPLETED}
                else event.trust
            )
            source_refs = _source_refs(entry, event)
            artifact_refs = _artifact_refs(entry, event)
            items.append(
                MemoryItem(
                    id=stable_id("mem", event.id, index, status.value, content),
                    kind=item_kind,
                    content=content,
                    provenance=Provenance(
                        event_ids=(event.id,),
                        source_refs=source_refs,
                        artifact_refs=artifact_refs,
                    ),
                    trust=trust,
                    evidence_status=status,
                    claim_slot=_claim_slot(entry),
                    confidence=_confidence(entry),
                    task_id=event.task_id,
                    worker_id=event.worker_id,
                    created_at=event.timestamp,
                    metadata={
                        "event_kind": event.kind.value,
                        "quote": entry.get("quote"),
                        "title": entry.get("title"),
                    },
                ),
            )

        missing_entries = (
            _as_entries(event.payload.get("missing_evidence"))
            or _as_entries(event.payload.get("missing"))
            or _as_entries(event.payload.get("gaps"))
        )
        for index, entry in enumerate(missing_entries):
            content = _entry_content(entry)
            if not content:
                continue
            items.append(
                MemoryItem(
                    id=stable_id("mem", event.id, "missing", index, content),
                    kind=MemoryItemKind.MISSING_EVIDENCE,
                    content=content,
                    provenance=Provenance(event_ids=(event.id,)),
                    trust=TrustLevel.MODEL_DERIVED,
                    evidence_status=EvidenceStatus.MISSING,
                    claim_slot=_claim_slot(entry),
                    confidence=_confidence(entry),
                    task_id=event.task_id,
                    worker_id=event.worker_id,
                    created_at=event.timestamp,
                    metadata={"event_kind": event.kind.value},
                ),
            )

        if not items and event.kind == EventKind.TOOL_RESULT and event.text():
            items.append(
                MemoryItem(
                    id=stable_id("mem", "observation", event.id),
                    kind=MemoryItemKind.OBSERVATION,
                    content=event.text(),
                    provenance=Provenance(
                        event_ids=(event.id,),
                        source_refs=event.provenance.source_refs,
                        artifact_refs=event.provenance.artifact_refs,
                    ),
                    trust=event.trust,
                    evidence_status=EvidenceStatus.UNKNOWN,
                    task_id=event.task_id,
                    worker_id=event.worker_id,
                    created_at=event.timestamp,
                    metadata={"event_kind": event.kind.value},
                ),
            )
        return tuple(items)

    def _evidence_entries_for_event(self, event: MemoryEvent) -> tuple[Mapping[str, Any], ...]:
        if event.kind not in {
            EventKind.TOOL_RESULT,
            EventKind.EVIDENCE_REVIEW,
            EventKind.TASK_COMPLETED,
            EventKind.WORKER_RESULT,
        }:
            return ()
        for key in ("evidence", "observations", "items", "results"):
            entries = _as_entries(event.payload.get(key))
            if entries:
                return entries
        return ()

    def _matches_query(self, item: MemoryItem, query: MemoryQuery) -> bool:
        if query.task_id and item.task_id not in {None, query.task_id}:
            return False
        if query.worker_id and item.worker_id not in {None, query.worker_id}:
            return False
        if not query.include_model_derived and item.trust not in {
            TrustLevel.SOURCE_OBSERVATION,
            TrustLevel.EXTRACTED_EVIDENCE,
        }:
            return False
        return True

    def _evidence_section(
        self,
        status: EvidenceStatus,
        items: Sequence[MemoryItem],
    ) -> ContextSection:
        rows = [self._row(item) for item in items]
        content = "\n".join(
            "- [{status}] {slot}{content} (events: {events}; sources: {sources})".format(
                status=row["status"],
                slot=f"{row['claim_slot']}: " if row["claim_slot"] else "",
                content=row["content"],
                events=", ".join(row["event_ids"]) or "none",
                sources=", ".join(row["source_refs"]) or "none",
            )
            for row in rows
        )
        trust = (
            TrustLevel.MODEL_DERIVED
            if status == EvidenceStatus.MISSING
            else TrustLevel.EXTRACTED_EVIDENCE
        )
        section_id = f"evidence_{status.value}"
        title = f"Evidence: {status.value}"
        item_ids = tuple(item.id for item in items)
        event_id_values = (
            event_id
            for item in items
            for event_id in item.provenance.event_ids
        )
        event_ids = unique_tuple(event_id_values)
        metadata = {"rows": rows, "status": status.value}

        return ContextSection(
            id=section_id,
            title=title,
            kind=SectionKind.EVIDENCE,
            trust=trust,
            content=content,
            item_ids=item_ids,
            event_ids=event_ids,
            metadata=metadata,
        )

    def _generic_section(
        self,
        section_id: str,
        title: str,
        kind: SectionKind,
        trust: TrustLevel,
        items: Iterable[MemoryItem],
    ) -> ContextSection:
        item_tuple = tuple(items)
        content = "\n".join(f"- [{item.id}] {item.content}" for item in item_tuple)
        item_ids = tuple(item.id for item in item_tuple)
        event_id_values = (
            event_id
            for item in item_tuple
            for event_id in item.provenance.event_ids
        )
        event_ids = unique_tuple(event_id_values)
        rows = [self._row(item) for item in item_tuple]
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

    def _row(self, item: MemoryItem) -> dict[str, Any]:
        return {
            "id": item.id,
            "status": item.evidence_status.value,
            "claim_slot": item.claim_slot,
            "content": item.content,
            "confidence": item.confidence,
            "trust": item.trust.value,
            "event_ids": list(item.provenance.event_ids),
            "source_refs": list(item.provenance.source_refs),
            "artifact_refs": list(item.provenance.artifact_refs),
            "kind": item.kind.value,
        }
