from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence, TypeVar
from uuid import uuid4


JsonDict = dict[str, Any]
T = TypeVar("T")


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    TASK_PLAN_CREATED = "task_plan_created"
    REASONING_NOTE = "reasoning_note"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    EVIDENCE_REVIEW = "evidence_review"
    TASK_COMPLETED = "task_completed"
    WORKER_CONTEXT_SENT = "worker_context_sent"
    WORKER_RESULT = "worker_result"
    SUMMARY_CREATED = "summary_created"
    FINAL_REPORT_CREATED = "final_report_created"
    CONTROL_STATE = "control_state"


class TrustLevel(StrEnum):
    SOURCE_OBSERVATION = "source_observation"
    EXTRACTED_EVIDENCE = "extracted_evidence"
    MODEL_DERIVED = "model_derived"
    WORKER_SUMMARY = "worker_summary"
    FINAL_SYNTHESIS = "final_synthesis"
    SYSTEM_CONTROL = "system_control"
    UNKNOWN = "unknown"


class SectionKind(StrEnum):
    MEMORY = "memory"
    EVIDENCE = "evidence"
    CONTROL = "control"
    TASK_STATE = "task_state"
    WARNING = "warning"
    SUMMARY = "summary"


class MemoryItemKind(StrEnum):
    TRANSCRIPT = "transcript"
    OBSERVATION = "observation"
    EVIDENCE = "evidence"
    MISSING_EVIDENCE = "missing_evidence"
    SUMMARY = "summary"
    TASK_STATE = "task_state"
    CONTROL = "control"
    SYNTHESIS = "synthesis"
    OPERATION = "operation"


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONTRADICTORY = "contradictory"
    MISSING = "missing"
    DUPLICATE = "duplicate"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


class Validity(StrEnum):
    CURRENT = "current"
    INVALIDATED = "invalidated"
    STALE = "stale"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class MemoryPhase(StrEnum):
    REASONING = "reasoning"
    WORKER_HANDOFF = "worker_handoff"
    FINAL_REPORT = "final_report"
    DEBUG_REPLAY = "debug_replay"


class MemoryOperationKind(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    INVALIDATE = "invalidate"
    COMPRESS = "compress"
    IGNORE = "ignore"


SOURCE_BEARING_TRUST = {
    TrustLevel.SOURCE_OBSERVATION,
    TrustLevel.EXTRACTED_EVIDENCE,
}
MODEL_DERIVED_TRUST = {
    TrustLevel.MODEL_DERIVED,
    TrustLevel.WORKER_SUMMARY,
    TrustLevel.FINAL_SYNTHESIS,
}
CONTROL_EVENT_KINDS = {
    EventKind.CONTROL_STATE,
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def stable_id(prefix: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def compact_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _enum(enum_cls: type[StrEnum], value: StrEnum | str) -> StrEnum:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(str(value))


def _tuple_str(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


def unique_tuple(values: Iterable[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(values))


def _metadata(values: Mapping[str, Any] | None) -> JsonDict:
    return dict(values or {})


def _default_trust_for_kind(kind: EventKind) -> TrustLevel:
    if kind == EventKind.TOOL_RESULT:
        return TrustLevel.SOURCE_OBSERVATION
    if kind == EventKind.EVIDENCE_REVIEW:
        return TrustLevel.EXTRACTED_EVIDENCE
    if kind == EventKind.WORKER_RESULT:
        return TrustLevel.WORKER_SUMMARY
    if kind in {EventKind.FINAL_REPORT_CREATED, EventKind.SUMMARY_CREATED}:
        return TrustLevel.FINAL_SYNTHESIS
    if kind in {
        EventKind.CONTROL_STATE,
        EventKind.RUN_STARTED,
        EventKind.TASK_PLAN_CREATED,
        EventKind.WORKER_CONTEXT_SENT,
    }:
        return TrustLevel.SYSTEM_CONTROL
    return TrustLevel.MODEL_DERIVED


@dataclass(frozen=True)
class Provenance:
    event_ids: tuple[str, ...] = ()
    item_ids: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_ids", _tuple_str(self.event_ids))
        object.__setattr__(self, "item_ids", _tuple_str(self.item_ids))
        object.__setattr__(self, "source_refs", _tuple_str(self.source_refs))
        object.__setattr__(self, "artifact_refs", _tuple_str(self.artifact_refs))

    def merge(self, other: Provenance) -> Provenance:
        event_ids = unique_tuple((*self.event_ids, *other.event_ids))
        item_ids = unique_tuple((*self.item_ids, *other.item_ids))
        source_refs = unique_tuple((*self.source_refs, *other.source_refs))
        artifact_refs = unique_tuple((*self.artifact_refs, *other.artifact_refs))
        note = self.note or other.note

        return Provenance(
            event_ids=event_ids,
            item_ids=item_ids,
            source_refs=source_refs,
            artifact_refs=artifact_refs,
            note=note,
        )

    def to_dict(self) -> JsonDict:
        return {
            "event_ids": list(self.event_ids),
            "item_ids": list(self.item_ids),
            "source_refs": list(self.source_refs),
            "artifact_refs": list(self.artifact_refs),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Provenance:
        if not data:
            return cls()
        event_ids = tuple(data.get("event_ids", ()))
        item_ids = tuple(data.get("item_ids", ()))
        source_refs = tuple(data.get("source_refs", ()))
        artifact_refs = tuple(data.get("artifact_refs", ()))
        note = data.get("note")

        return cls(
            event_ids=event_ids,
            item_ids=item_ids,
            source_refs=source_refs,
            artifact_refs=artifact_refs,
            note=note,
        )


@dataclass(frozen=True)
class MemoryEvent:
    kind: EventKind
    content: str = ""
    id: str = field(default_factory=lambda: new_id("evt"))
    timestamp: str | None = None
    payload: JsonDict = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)
    trust: TrustLevel | None = None
    task_id: str | None = None
    worker_id: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = _enum(EventKind, self.kind)
        trust = _enum(TrustLevel, self.trust) if self.trust else _default_trust_for_kind(kind)
        provenance = (
            self.provenance
            if isinstance(self.provenance, Provenance)
            else Provenance.from_dict(self.provenance)
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "trust", trust)
        object.__setattr__(self, "payload", _metadata(self.payload))
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "tags", _tuple_str(self.tags))

    @property
    def is_control(self) -> bool:
        return self.kind in CONTROL_EVENT_KINDS or self.trust == TrustLevel.SYSTEM_CONTROL

    @property
    def is_source_bearing(self) -> bool:
        return self.trust in SOURCE_BEARING_TRUST

    @property
    def is_model_derived(self) -> bool:
        return self.trust in MODEL_DERIVED_TRUST

    def text(self) -> str:
        if self.content:
            return self.content
        if self.payload:
            return compact_json(self.payload)
        return ""

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
            "provenance": self.provenance.to_dict(),
            "trust": self.trust.value,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryEvent:
        return cls(
            id=str(data["id"]),
            kind=EventKind(data["kind"]),
            content=str(data.get("content") or ""),
            timestamp=data.get("timestamp"),
            payload=dict(data.get("payload") or {}),
            provenance=Provenance.from_dict(data.get("provenance")),
            trust=TrustLevel(data["trust"]) if data.get("trust") else None,
            task_id=data.get("task_id"),
            worker_id=data.get("worker_id"),
            tags=tuple(data.get("tags") or ()),
        )


@dataclass(frozen=True)
class MemoryItem:
    kind: MemoryItemKind
    content: str
    provenance: Provenance
    id: str = field(default_factory=lambda: new_id("mem"))
    trust: TrustLevel = TrustLevel.UNKNOWN
    validity: Validity = Validity.CURRENT
    evidence_status: EvidenceStatus = EvidenceStatus.UNKNOWN
    claim_slot: str | None = None
    confidence: float | None = None
    task_id: str | None = None
    worker_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(MemoryItemKind, self.kind))
        object.__setattr__(self, "trust", _enum(TrustLevel, self.trust))
        object.__setattr__(self, "validity", _enum(Validity, self.validity))
        object.__setattr__(
            self,
            "evidence_status",
            _enum(EvidenceStatus, self.evidence_status),
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, Provenance)
            else Provenance.from_dict(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def is_primary_evidence(self) -> bool:
        return (
            self.kind in {MemoryItemKind.OBSERVATION, MemoryItemKind.EVIDENCE}
            and self.trust in SOURCE_BEARING_TRUST
            and self.validity == Validity.CURRENT
            and self.evidence_status
            not in {EvidenceStatus.MISSING, EvidenceStatus.DUPLICATE, EvidenceStatus.IRRELEVANT}
            and bool(self.provenance.event_ids)
        )

    @property
    def is_control(self) -> bool:
        return self.kind == MemoryItemKind.CONTROL or self.trust == TrustLevel.SYSTEM_CONTROL

    def with_updates(self, **updates: Any) -> MemoryItem:
        data = self.to_dict()
        data.update(updates)
        if "provenance" in data and isinstance(data["provenance"], Provenance):
            data["provenance"] = data["provenance"].to_dict()
        return MemoryItem.from_dict(data)

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "content": self.content,
            "provenance": self.provenance.to_dict(),
            "trust": self.trust.value,
            "validity": self.validity.value,
            "evidence_status": self.evidence_status.value,
            "claim_slot": self.claim_slot,
            "confidence": self.confidence,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryItem:
        return cls(
            id=str(data["id"]),
            kind=MemoryItemKind(data["kind"]),
            content=str(data.get("content") or ""),
            provenance=Provenance.from_dict(data.get("provenance")),
            trust=TrustLevel(data.get("trust", TrustLevel.UNKNOWN.value)),
            validity=Validity(data.get("validity", Validity.CURRENT.value)),
            evidence_status=EvidenceStatus(
                data.get("evidence_status", EvidenceStatus.UNKNOWN.value),
            ),
            claim_slot=data.get("claim_slot"),
            confidence=data.get("confidence"),
            task_id=data.get("task_id"),
            worker_id=data.get("worker_id"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MemoryState:
    model_name: str
    events: tuple[MemoryEvent, ...] = ()
    items: tuple[MemoryItem, ...] = ()
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "events",
            tuple(
                event if isinstance(event, MemoryEvent) else MemoryEvent.from_dict(event)
                for event in self.events
            ),
        )
        object.__setattr__(
            self,
            "items",
            tuple(
                item if isinstance(item, MemoryItem) else MemoryItem.from_dict(item)
                for item in self.items
            ),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def with_events(self, events: Sequence[MemoryEvent]) -> MemoryState:
        existing = {event.id for event in self.events}
        merged = (*self.events, *(event for event in events if event.id not in existing))
        return MemoryState(self.model_name, events=merged, items=self.items, metadata=self.metadata)

    def with_items(self, items: Sequence[MemoryItem]) -> MemoryState:
        existing = {item.id for item in self.items}
        merged = (*self.items, *(item for item in items if item.id not in existing))
        return MemoryState(self.model_name, events=self.events, items=merged, metadata=self.metadata)

    def replace_items(self, items: Sequence[MemoryItem]) -> MemoryState:
        return MemoryState(self.model_name, events=self.events, items=tuple(items), metadata=self.metadata)

    def with_metadata(self, **metadata: Any) -> MemoryState:
        merged = {**self.metadata, **metadata}
        return MemoryState(self.model_name, events=self.events, items=self.items, metadata=merged)

    def to_dict(self) -> JsonDict:
        return {
            "model_name": self.model_name,
            "events": [event.to_dict() for event in self.events],
            "items": [item.to_dict() for item in self.items],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryState:
        return cls(
            model_name=str(data["model_name"]),
            events=tuple(MemoryEvent.from_dict(event) for event in data.get("events", ())),
            items=tuple(MemoryItem.from_dict(item) for item in data.get("items", ())),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MemoryQuery:
    phase: MemoryPhase = MemoryPhase.REASONING
    objective: str = ""
    task_id: str | None = None
    worker_id: str | None = None
    budget_tokens: int | None = None
    include_control: bool = True
    include_synthesis: bool = False
    include_model_derived: bool = True
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _enum(MemoryPhase, self.phase))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> JsonDict:
        return {
            "phase": self.phase.value,
            "objective": self.objective,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "budget_tokens": self.budget_tokens,
            "include_control": self.include_control,
            "include_synthesis": self.include_synthesis,
            "include_model_derived": self.include_model_derived,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryQuery:
        return cls(
            phase=MemoryPhase(data.get("phase", MemoryPhase.REASONING.value)),
            objective=str(data.get("objective") or ""),
            task_id=data.get("task_id"),
            worker_id=data.get("worker_id"),
            budget_tokens=data.get("budget_tokens"),
            include_control=bool(data.get("include_control", True)),
            include_synthesis=bool(data.get("include_synthesis", False)),
            include_model_derived=bool(data.get("include_model_derived", True)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ContextSection:
    id: str
    title: str
    kind: SectionKind
    content: str
    trust: TrustLevel = TrustLevel.UNKNOWN
    item_ids: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    token_estimate: int | None = None
    omitted_count: int = 0
    omitted_reason: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(SectionKind, self.kind))
        object.__setattr__(self, "trust", _enum(TrustLevel, self.trust))
        object.__setattr__(self, "item_ids", _tuple_str(self.item_ids))
        object.__setattr__(self, "event_ids", _tuple_str(self.event_ids))
        object.__setattr__(self, "warnings", _tuple_str(self.warnings))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if self.token_estimate is None:
            object.__setattr__(self, "token_estimate", estimate_tokens(self.content))

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind.value,
            "content": self.content,
            "trust": self.trust.value,
            "item_ids": list(self.item_ids),
            "event_ids": list(self.event_ids),
            "token_estimate": self.token_estimate,
            "omitted_count": self.omitted_count,
            "omitted_reason": self.omitted_reason,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextSection:
        section_id = str(data["id"])
        title = str(data.get("title") or "")
        kind = SectionKind(data["kind"])
        content = str(data.get("content") or "")
        trust = TrustLevel(data.get("trust", TrustLevel.UNKNOWN.value))
        item_ids = tuple(data.get("item_ids") or ())
        event_ids = tuple(data.get("event_ids") or ())
        token_estimate = data.get("token_estimate")
        omitted_count = int(data.get("omitted_count") or 0)
        omitted_reason = data.get("omitted_reason")
        warnings = tuple(data.get("warnings") or ())
        metadata = dict(data.get("metadata") or {})

        return cls(
            id=section_id,
            title=title,
            kind=kind,
            content=content,
            trust=trust,
            item_ids=item_ids,
            event_ids=event_ids,
            token_estimate=token_estimate,
            omitted_count=omitted_count,
            omitted_reason=omitted_reason,
            warnings=warnings,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ContextPacket:
    query: MemoryQuery
    sections: tuple[ContextSection, ...]
    id: str = field(default_factory=lambda: new_id("pkt"))
    warnings: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        query = (
            self.query
            if isinstance(self.query, MemoryQuery)
            else MemoryQuery.from_dict(self.query)
        )
        object.__setattr__(self, "query", query)
        object.__setattr__(
            self,
            "sections",
            tuple(
                section
                if isinstance(section, ContextSection)
                else ContextSection.from_dict(section)
                for section in self.sections
            ),
        )
        object.__setattr__(self, "warnings", _tuple_str(self.warnings))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def token_estimate(self) -> int:
        return sum(section.token_estimate or 0 for section in self.sections)

    @property
    def event_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for section in self.sections:
            ids.extend(section.event_ids)
        return unique_tuple(ids)

    @property
    def item_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for section in self.sections:
            ids.extend(section.item_ids)
        return unique_tuple(ids)

    def section(self, section_id: str) -> ContextSection | None:
        return next((section for section in self.sections if section.id == section_id), None)

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "query": self.query.to_dict(),
            "sections": [section.to_dict() for section in self.sections],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextPacket:
        return cls(
            id=str(data["id"]),
            query=MemoryQuery.from_dict(data["query"]),
            sections=tuple(ContextSection.from_dict(section) for section in data.get("sections", ())),
            warnings=tuple(data.get("warnings") or ()),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MemoryOperation:
    kind: MemoryOperationKind
    provenance: Provenance
    id: str = field(default_factory=lambda: new_id("op"))
    item_id: str | None = None
    item: MemoryItem | None = None
    content: str | None = None
    reason: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(MemoryOperationKind, self.kind))
        provenance = (
            self.provenance
            if isinstance(self.provenance, Provenance)
            else Provenance.from_dict(self.provenance)
        )
        item = self.item if self.item is None or isinstance(self.item, MemoryItem) else MemoryItem.from_dict(self.item)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "item", item)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "provenance": self.provenance.to_dict(),
            "item_id": self.item_id,
            "item": self.item.to_dict() if self.item else None,
            "content": self.content,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryOperation:
        return cls(
            id=str(data["id"]),
            kind=MemoryOperationKind(data["kind"]),
            provenance=Provenance.from_dict(data.get("provenance")),
            item_id=data.get("item_id"),
            item=MemoryItem.from_dict(data["item"]) if data.get("item") else None,
            content=data.get("content"),
            reason=data.get("reason"),
            metadata=dict(data.get("metadata") or {}),
        )
