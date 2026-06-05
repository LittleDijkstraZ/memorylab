from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from memory_lab.models import (
    EvidenceLedgerMemory,
    FullHistoryMemory,
    HierarchicalSummaryMemory,
    LLMManagedMemory,
    MemoryModel,
    RollingSummaryMemory,
)
from memory_lab.renderers import (
    CompactPromptRenderer,
    ContextRenderer,
    EvidenceTableRenderer,
    RenderedContext,
    ResearchBriefRenderer,
)
from memory_lab.schema import (
    EventKind,
    MemoryEvent,
    MemoryPhase,
    MemoryQuery,
    MemoryState,
    Provenance,
    TrustLevel,
    new_id,
)


MODEL_REGISTRY = {
    "full_history": FullHistoryMemory,
    "evidence_ledger": EvidenceLedgerMemory,
    "rolling_summary": RollingSummaryMemory,
    "hierarchical_summary": HierarchicalSummaryMemory,
    "llm_managed": LLMManagedMemory,
}


RENDERER_REGISTRY = {
    "research_brief": ResearchBriefRenderer,
    "compact_prompt": CompactPromptRenderer,
    "evidence_table": EvidenceTableRenderer,
}


DEFAULT_RENDERERS = {
    "full_history": "research_brief",
    "evidence_ledger": "evidence_table",
    "rolling_summary": "compact_prompt",
    "hierarchical_summary": "compact_prompt",
    "llm_managed": "research_brief",
}


EVENT_KIND_ALIASES = {
    "run": EventKind.RUN_STARTED,
    "start": EventKind.RUN_STARTED,
    "run_started": EventKind.RUN_STARTED,
    "task_plan": EventKind.TASK_PLAN_CREATED,
    "task_plan_created": EventKind.TASK_PLAN_CREATED,
    "note": EventKind.REASONING_NOTE,
    "record": EventKind.REASONING_NOTE,
    "reasoning": EventKind.REASONING_NOTE,
    "reasoning_note": EventKind.REASONING_NOTE,
    "tool_call": EventKind.TOOL_CALL,
    "tool_result": EventKind.TOOL_RESULT,
    "observation": EventKind.TOOL_RESULT,
    "evidence": EventKind.TOOL_RESULT,
    "missing": EventKind.EVIDENCE_REVIEW,
    "missing_evidence": EventKind.EVIDENCE_REVIEW,
    "evidence_review": EventKind.EVIDENCE_REVIEW,
    "task_completed": EventKind.TASK_COMPLETED,
    "worker_context_sent": EventKind.WORKER_CONTEXT_SENT,
    "worker_result": EventKind.WORKER_RESULT,
    "summary": EventKind.SUMMARY_CREATED,
    "summary_created": EventKind.SUMMARY_CREATED,
    "final_report": EventKind.FINAL_REPORT_CREATED,
    "final_report_created": EventKind.FINAL_REPORT_CREATED,
    "control": EventKind.CONTROL_STATE,
    "control_state": EventKind.CONTROL_STATE,
}


class Memory:
    """Stateful beginner-friendly wrapper around a memory model."""

    def __init__(
        self,
        model: str | MemoryModel = "full_history",
        *,
        renderer: str | ContextRenderer | None = None,
        state: MemoryState | None = None,
    ) -> None:
        self.model = self._resolve_model(model)
        self.renderer = self._resolve_renderer(renderer, self.model.name)
        self.state = state or self.model.initial_state()

    def update(
        self,
        data: str | Mapping[str, Any] | MemoryEvent | Iterable[str | Mapping[str, Any] | MemoryEvent],
        *,
        kind: str | EventKind | None = None,
        **fields: Any,
    ) -> Memory:
        events = self._normalize_events(data, kind=kind, fields=fields)
        self.state = self.model.ingest(self.state, events)
        return self

    def read(
        self,
        objective: str = "",
        *,
        phase: str | MemoryPhase = MemoryPhase.REASONING,
        task_id: str | None = None,
        worker_id: str | None = None,
        budget_tokens: int | None = None,
        include_control: bool = True,
        include_synthesis: bool = False,
        include_model_derived: bool = True,
        renderer: str | ContextRenderer | None = None,
    ) -> RenderedContext:
        query = MemoryQuery(
            phase=phase,
            objective=objective,
            task_id=task_id,
            worker_id=worker_id,
            budget_tokens=budget_tokens,
            include_control=include_control,
            include_synthesis=include_synthesis,
            include_model_derived=include_model_derived,
        )
        packet = self.model.select_context(self.state, query)
        active_renderer = self._resolve_renderer(renderer, self.model.name) if renderer else self.renderer
        return active_renderer.render(packet, budget_tokens=budget_tokens)

    def packet(
        self,
        objective: str = "",
        *,
        phase: str | MemoryPhase = MemoryPhase.REASONING,
        task_id: str | None = None,
        worker_id: str | None = None,
        budget_tokens: int | None = None,
        include_control: bool = True,
        include_synthesis: bool = False,
        include_model_derived: bool = True,
    ):
        query = MemoryQuery(
            phase=phase,
            objective=objective,
            task_id=task_id,
            worker_id=worker_id,
            budget_tokens=budget_tokens,
            include_control=include_control,
            include_synthesis=include_synthesis,
            include_model_derived=include_model_derived,
        )
        return self.model.select_context(self.state, query)

    def _normalize_events(
        self,
        data: str | Mapping[str, Any] | MemoryEvent | Iterable[str | Mapping[str, Any] | MemoryEvent],
        *,
        kind: str | EventKind | None,
        fields: Mapping[str, Any],
    ) -> tuple[MemoryEvent, ...]:
        if isinstance(data, MemoryEvent):
            return (data,)
        if isinstance(data, str):
            merged = {"content": data, **fields}
            if kind is not None:
                merged["kind"] = kind
            return (self._event_from_mapping(merged),)
        if isinstance(data, Mapping):
            merged = {**data, **fields}
            if kind is not None:
                merged["kind"] = kind
            return (self._event_from_mapping(merged),)

        events = []
        for entry in data:
            if isinstance(entry, MemoryEvent):
                events.append(entry)
            elif isinstance(entry, str):
                entry_kind = kind or self._default_kind()
                events.append(self._event_from_mapping({"kind": entry_kind, "content": entry}))
            elif isinstance(entry, Mapping):
                merged = {**entry}
                if kind is not None and "kind" not in merged:
                    merged["kind"] = kind
                events.append(self._event_from_mapping(merged))
            else:
                raise TypeError(f"Unsupported memory update entry: {type(entry)!r}")
        return tuple(events)

    def _event_from_mapping(self, data: Mapping[str, Any]) -> MemoryEvent:
        raw_kind = data.get("kind", self._default_kind())
        kind_name = str(raw_kind.value if isinstance(raw_kind, EventKind) else raw_kind)
        if kind_name in {"evidence", "missing", "missing_evidence"}:
            return self._simple_evidence_event(kind_name, data)

        event_kind = self._event_kind(kind_name)
        content = str(data.get("content") or data.get("text") or "")
        payload = self._payload(data)
        source_refs = self._refs(data, "source", "source_ref", "source_refs", "url")
        artifact_refs = self._refs(data, "artifact", "artifact_ref", "artifact_refs")
        provenance = Provenance(source_refs=source_refs, artifact_refs=artifact_refs)
        trust = self._trust(data)

        return MemoryEvent(
            id=data.get("id") or new_id("evt"),
            kind=event_kind,
            content=content,
            timestamp=data.get("timestamp"),
            payload=payload,
            provenance=provenance,
            trust=trust,
            task_id=data.get("task_id"),
            worker_id=data.get("worker_id"),
            tags=tuple(data.get("tags") or ()),
        )

    def _simple_evidence_event(
        self,
        kind_name: str,
        data: Mapping[str, Any],
    ) -> MemoryEvent:
        content = str(data.get("content") or data.get("text") or "")
        entry = {
            "claim_slot": data.get("slot") or data.get("claim_slot"),
            "content": content,
            "status": data.get("status", "missing" if kind_name.startswith("missing") else "supported"),
            "confidence": data.get("confidence"),
        }
        source_refs = self._refs(data, "source", "source_ref", "source_refs", "url")
        artifact_refs = self._refs(data, "artifact", "artifact_ref", "artifact_refs")
        if source_refs:
            entry["source_refs"] = list(source_refs)
        if artifact_refs:
            entry["artifact_refs"] = list(artifact_refs)

        if kind_name.startswith("missing"):
            event_kind = EventKind.EVIDENCE_REVIEW
            payload = {"missing_evidence": [entry]}
            trust = TrustLevel.MODEL_DERIVED
        else:
            event_kind = EventKind.TOOL_RESULT
            payload = {"evidence": [entry]}
            trust = TrustLevel.SOURCE_OBSERVATION

        return MemoryEvent(
            id=data.get("id") or new_id("evt"),
            kind=event_kind,
            content="",
            timestamp=data.get("timestamp"),
            payload=payload,
            provenance=Provenance(source_refs=source_refs, artifact_refs=artifact_refs),
            trust=trust,
            task_id=data.get("task_id"),
            worker_id=data.get("worker_id"),
            tags=tuple(data.get("tags") or ()),
        )

    def _resolve_model(self, model: str | MemoryModel) -> MemoryModel:
        if not isinstance(model, str):
            return model
        try:
            model_cls = MODEL_REGISTRY[model]
        except KeyError as exc:
            known = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown memory model {model!r}. Known models: {known}") from exc
        return model_cls()

    def _resolve_renderer(
        self,
        renderer: str | ContextRenderer | None,
        model_name: str,
    ) -> ContextRenderer:
        if renderer is None:
            renderer = DEFAULT_RENDERERS.get(model_name, "research_brief")
        if not isinstance(renderer, str):
            return renderer
        try:
            renderer_cls = RENDERER_REGISTRY[renderer]
        except KeyError as exc:
            known = ", ".join(sorted(RENDERER_REGISTRY))
            raise ValueError(f"Unknown renderer {renderer!r}. Known renderers: {known}") from exc
        return renderer_cls()

    def _event_kind(self, raw_kind: str) -> EventKind:
        if raw_kind in EVENT_KIND_ALIASES:
            return EVENT_KIND_ALIASES[raw_kind]
        try:
            return EventKind(raw_kind)
        except ValueError as exc:
            known = ", ".join(sorted(EVENT_KIND_ALIASES))
            raise ValueError(f"Unknown memory update kind {raw_kind!r}. Known kinds: {known}") from exc

    def _default_kind(self) -> str:
        if self.model.name == "evidence_ledger":
            return "observation"
        return "record"

    def _payload(self, data: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(data.get("payload") or {})
        metadata = data.get("metadata")
        if isinstance(metadata, Mapping):
            existing_metadata = dict(payload.get("metadata") or {})
            payload["metadata"] = {**existing_metadata, **metadata}

        reserved_keys = {
            "id",
            "kind",
            "content",
            "text",
            "timestamp",
            "payload",
            "metadata",
            "source",
            "source_ref",
            "source_refs",
            "url",
            "artifact",
            "artifact_ref",
            "artifact_refs",
            "trust",
            "task_id",
            "worker_id",
            "tags",
        }
        extra = {
            key: value
            for key, value in data.items()
            if key not in reserved_keys
        }
        if extra:
            payload["fields"] = {**dict(payload.get("fields") or {}), **extra}
        return payload

    def _trust(self, data: Mapping[str, Any]) -> TrustLevel | None:
        raw_trust = data.get("trust")
        if raw_trust is None:
            return None
        if isinstance(raw_trust, TrustLevel):
            return raw_trust
        return TrustLevel(str(raw_trust))

    def _refs(self, data: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
        refs: list[str] = []
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                refs.append(value)
            elif isinstance(value, Sequence):
                refs.extend(str(item) for item in value)
            else:
                refs.append(str(value))
        return tuple(dict.fromkeys(refs))


__all__ = ["Memory"]
