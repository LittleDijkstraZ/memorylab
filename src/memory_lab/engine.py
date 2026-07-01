from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from memory_lab.capabilities.base import (
    CapabilityResponse,
    CompressionRequest,
    Compressor,
    RetrievalRequest,
    Retriever,
)
from memory_lab.capabilities.deterministic import DeterministicCompressor
from memory_lab.capabilities.retrieval import KeywordRetriever
from memory_lab.recipe import (
    CapabilityTrace,
    MemoryDeriveResult,
    MemoryLayerSpec,
    MemoryRecipe,
    MemoryRecipeOperation,
    MemorySelector,
)
from memory_lab.schema import (
    ContextPacket,
    ContextSection,
    EventKind,
    MemoryEvent,
    MemoryItem,
    MemoryItemKind,
    MemoryQuery,
    MemoryState,
    MODEL_DERIVED_TRUST,
    SectionKind,
    TrustLevel,
    estimate_tokens,
    unique_tuple,
)


class MissingCapabilityError(ValueError):
    """Raised when a recipe requires unavailable capabilities."""


@dataclass(frozen=True)
class RecipeCapabilities:
    compressor: Compressor | None = None
    retriever: Retriever | None = None
    allow_deterministic_fallback: bool = True

    @classmethod
    def deterministic(cls) -> RecipeCapabilities:
        return cls(
            compressor=DeterministicCompressor(),
            retriever=KeywordRetriever(),
        )

    def supports(self, capability: str) -> bool:
        normalized = str(capability)
        if normalized == MemoryRecipeOperation.KEEP_RAW.value:
            return True
        if normalized == MemoryRecipeOperation.COMPRESS.value:
            return self.compressor is not None
        if normalized == MemoryRecipeOperation.RETRIEVE.value:
            return self.retriever is not None
        return False


class RecipeEngine:
    """Derive context packets from an existing MemoryState using a recipe."""

    def derive(
        self,
        state: MemoryState,
        query: MemoryQuery,
        recipe: MemoryRecipe,
        *,
        capabilities: RecipeCapabilities | None = None,
    ) -> MemoryDeriveResult:
        capabilities = capabilities or RecipeCapabilities.deterministic()
        self._require_capabilities(recipe, capabilities)

        query_state = self._query_state(state, query)
        base_state = self._selected_state(query_state, recipe.selector)
        sections: list[ContextSection] = []
        warnings: list[str] = []
        traces: list[CapabilityTrace] = []

        for layer in recipe.layers:
            layer_state = self._selected_state(base_state, layer.selector)
            response, layer_warnings, capability_name = self._derive_layer(
                layer,
                layer_state,
                query,
                capabilities,
            )
            layer_warnings = (
                *self._optional_capability_warnings(layer, capabilities),
                *layer_warnings,
            )
            warnings.extend(layer_warnings)
            section = self._section_from_response(layer, response)
            sections.append(section)
            traces.append(
                CapabilityTrace(
                    layer_id=layer.id,
                    operation=layer.operation,
                    capability=capability_name,
                    warnings=layer_warnings,
                    metadata=dict(response.metadata),
                ),
            )

        pre_budget_event_ids = _section_event_ids(sections)
        pre_budget_item_ids = _section_item_ids(sections)
        sections, budget_warnings, omitted_sections = self._apply_packet_budget(
            sections,
            query.budget_tokens,
        )
        included_event_ids = _section_event_ids(sections)
        included_item_ids = _section_item_ids(sections)
        omitted_event_ids = _difference(pre_budget_event_ids, included_event_ids)
        omitted_item_ids = _difference(pre_budget_item_ids, included_item_ids)
        warnings.extend(budget_warnings)
        packet = ContextPacket(
            query=query,
            sections=tuple(sections),
            warnings=tuple(warnings),
            metadata={
                "recipe": recipe.name,
                "recipe_schema_version": recipe.schema_version,
                "renderer": recipe.renderer,
                "omitted_section_ids": omitted_sections,
                "omitted_event_ids": list(omitted_event_ids),
                "omitted_item_ids": list(omitted_item_ids),
            },
        )
        return MemoryDeriveResult(
            packet=packet,
            traces=tuple(traces),
            recipe_name=recipe.name,
            recipe_schema_version=recipe.schema_version,
            selected_event_ids=included_event_ids,
            selected_item_ids=included_item_ids,
            warnings=tuple(warnings),
            metadata={
                "model_name": state.model_name,
                "layer_count": len(recipe.layers),
                "pre_budget_selected_event_ids": list(pre_budget_event_ids),
                "pre_budget_selected_item_ids": list(pre_budget_item_ids),
                "omitted_event_ids": list(omitted_event_ids),
                "omitted_item_ids": list(omitted_item_ids),
            },
        )

    def _require_capabilities(
        self,
        recipe: MemoryRecipe,
        capabilities: RecipeCapabilities,
    ) -> None:
        missing = tuple(
            capability
            for capability in (
                *recipe.requires_capabilities,
                *(
                    capability
                    for layer in recipe.layers
                    for capability in layer.requires_capabilities
                ),
            )
            if not capabilities.supports(capability)
        )
        if missing:
            missing_text = ", ".join(missing)
            raise MissingCapabilityError(
                f"Recipe {recipe.name!r} requires unavailable "
                f"capabilities: {missing_text}",
            )

    def _optional_capability_warnings(
        self,
        layer: MemoryLayerSpec,
        capabilities: RecipeCapabilities,
    ) -> tuple[str, ...]:
        return tuple(
            (
                f"optional capability {capability!r} unavailable for "
                f"layer {layer.id!r}; deterministic fallback may be used"
            )
            for capability in layer.optional_capabilities
            if not capabilities.supports(capability)
        )

    def _derive_layer(
        self,
        layer: MemoryLayerSpec,
        state: MemoryState,
        query: MemoryQuery,
        capabilities: RecipeCapabilities,
    ) -> tuple[CapabilityResponse, tuple[str, ...], str]:
        if layer.operation == MemoryRecipeOperation.KEEP_RAW:
            return self._keep_raw(layer, state), (), "keep_raw"
        if layer.operation == MemoryRecipeOperation.COMPRESS:
            compressor = capabilities.compressor
            warnings: tuple[str, ...] = ()
            capability_name = getattr(compressor, "name", "compress")
            if compressor is None:
                if not capabilities.allow_deterministic_fallback:
                    raise MissingCapabilityError(
                        "compress capability unavailable and fallback disabled",
                    )
                compressor = DeterministicCompressor()
                capability_name = compressor.name
                warnings = (
                    "optional compress capability missing; used deterministic fallback",
                )
            request = CompressionRequest(
                layer_id=layer.id,
                events=state.events,
                items=state.items,
                instruction=layer.instruction,
                budget_tokens=layer.budget_tokens,
                query=query,
            )
            try:
                response = compressor.compress(request)
            except Exception as exc:
                if not capabilities.allow_deterministic_fallback:
                    raise
                fallback = DeterministicCompressor()
                fallback_response = fallback.compress(request)
                fallback_warning = (
                    f"compress capability {capability_name!r} failed; "
                    "used deterministic fallback: "
                    f"{type(exc).__name__}: {exc}",
                )
                response = replace(
                    fallback_response,
                    warnings=(*fallback_response.warnings, fallback_warning),
                    metadata={
                        **fallback_response.metadata,
                        "fallback_from": capability_name,
                        "fallback_error_type": type(exc).__name__,
                        "fallback_error": str(exc),
                    },
                )
                return response, (*warnings, *response.warnings), fallback.name
            return response, (*warnings, *response.warnings), capability_name
        if layer.operation == MemoryRecipeOperation.RETRIEVE:
            retriever = capabilities.retriever
            warnings = ()
            capability_name = getattr(retriever, "name", "retrieve")
            if retriever is None:
                if not capabilities.allow_deterministic_fallback:
                    raise MissingCapabilityError(
                        "retrieve capability unavailable and fallback disabled",
                    )
                retriever = KeywordRetriever()
                capability_name = retriever.name
                warnings = (
                    "optional retrieve capability missing; used deterministic fallback",
                )
            response = retriever.retrieve(
                RetrievalRequest(
                    layer_id=layer.id,
                    state=state,
                    selector=layer.selector,
                    query=query,
                    budget_tokens=layer.budget_tokens,
                ),
            )
            return response, (*warnings, *response.warnings), capability_name
        raise ValueError(f"Unsupported recipe operation {layer.operation!r}")

    def _keep_raw(
        self,
        layer: MemoryLayerSpec,
        state: MemoryState,
    ) -> CapabilityResponse:
        lines = [_event_line(event) for event in state.events]
        if not lines:
            lines = [_item_line(item) for item in state.items]
        content = "\n".join(lines) or "(no selected memory content)"
        return CapabilityResponse(
            content=content,
            event_ids=unique_tuple(event.id for event in state.events),
            item_ids=unique_tuple(item.id for item in state.items),
            metadata={
                "method": "keep_raw",
                "instruction": layer.instruction,
                "token_estimate": estimate_tokens(content),
            },
        )

    def _section_from_response(
        self,
        layer: MemoryLayerSpec,
        response: CapabilityResponse,
    ) -> ContextSection:
        return ContextSection(
            id=layer.id,
            title=layer.title,
            kind=self._section_kind(layer),
            trust=self._section_trust(layer),
            content=response.content,
            item_ids=response.item_ids if layer.include_provenance else (),
            event_ids=response.event_ids if layer.include_provenance else (),
            omitted_count=response.omitted_count,
            omitted_reason=response.omitted_reason,
            warnings=response.warnings,
            metadata={
                **response.metadata,
                "operation": layer.operation.value,
                "layer_metadata": dict(layer.metadata),
            },
        )

    def _apply_packet_budget(
        self,
        sections: list[ContextSection],
        budget_tokens: int | None,
    ) -> tuple[list[ContextSection], tuple[str, ...], list[str]]:
        if budget_tokens is None:
            return sections, (), []
        kept: list[ContextSection] = []
        warnings: list[str] = []
        omitted: list[str] = []
        token_total = 0
        for section in sections:
            section_tokens = section.token_estimate or 0
            if kept and token_total + section_tokens > budget_tokens:
                omitted.append(section.id)
                warnings.append(
                    f"section {section.id} omitted by packet budget {budget_tokens}",
                )
                continue
            if not kept and section_tokens > budget_tokens:
                warnings.append(
                    f"section {section.id} exceeds packet budget {budget_tokens}",
                )
            kept.append(section)
            token_total += section_tokens
        return kept, tuple(warnings), omitted

    def _selected_state(
        self,
        state: MemoryState,
        selector: MemorySelector,
    ) -> MemoryState:
        events = self._select_events(state.events, selector)
        items = self._select_items(state.items, selector)
        event_id_filter = set(selector.event_ids)
        item_id_filter = set(selector.item_ids)
        if item_id_filter and not event_id_filter:
            item_event_ids = {
                event_id
                for item in items
                for event_id in item.provenance.event_ids
            }
            events = tuple(event for event in events if event.id in item_event_ids)
        if event_id_filter and not item_id_filter:
            items = tuple(
                item
                for item in items
                if set(item.provenance.event_ids) & event_id_filter
            )
        return MemoryState(
            model_name=state.model_name,
            events=events,
            items=items,
            metadata=state.metadata,
        )

    def _query_state(
        self,
        state: MemoryState,
        query: MemoryQuery,
    ) -> MemoryState:
        events = tuple(
            event
            for event in state.events
            if self._matches_query_event(event, query)
        )
        items = tuple(
            item
            for item in state.items
            if self._matches_query_item(item, query)
        )
        return MemoryState(
            model_name=state.model_name,
            events=events,
            items=items,
            metadata=state.metadata,
        )

    def _select_events(
        self,
        events: tuple[MemoryEvent, ...],
        selector: MemorySelector,
    ) -> tuple[MemoryEvent, ...]:
        selected = tuple(event for event in events if self._matches_event(event, selector))
        if selector.recent_event_limit is not None:
            selected = selected[-selector.recent_event_limit :]
        return selected

    def _select_items(
        self,
        items: tuple[MemoryItem, ...],
        selector: MemorySelector,
    ) -> tuple[MemoryItem, ...]:
        return tuple(item for item in items if self._matches_item(item, selector))

    def _matches_event(self, event: MemoryEvent, selector: MemorySelector) -> bool:
        if selector.event_ids and event.id not in selector.event_ids:
            return False
        if selector.task_id and event.task_id != selector.task_id:
            return False
        if selector.worker_id and event.worker_id != selector.worker_id:
            return False
        if selector.kinds and event.kind.value not in selector.kinds:
            return False
        if selector.trust and event.trust.value not in selector.trust:
            return False
        if selector.tags and not set(selector.tags).issubset(set(event.tags)):
            return False
        return self._matches_metadata(event.to_dict(), event.payload, selector)

    def _matches_item(self, item: MemoryItem, selector: MemorySelector) -> bool:
        if selector.item_ids and item.id not in selector.item_ids:
            return False
        if selector.task_id and item.task_id != selector.task_id:
            return False
        if selector.worker_id and item.worker_id != selector.worker_id:
            return False
        if selector.kinds and item.kind.value not in selector.kinds:
            return False
        if selector.trust and item.trust.value not in selector.trust:
            return False
        item_tags = tuple(str(tag) for tag in item.metadata.get("tags", ()))
        if selector.tags and not set(selector.tags).issubset(set(item_tags)):
            return False
        return self._matches_metadata(item.to_dict(), item.metadata, selector)

    def _matches_metadata(
        self,
        top_level: Mapping[str, Any],
        metadata: Mapping[str, Any],
        selector: MemorySelector,
    ) -> bool:
        combined = self._combined_metadata(top_level, metadata)
        for key in selector.metadata_exists:
            if not _exists(combined.get(key)):
                return False
        for key, expected in selector.metadata_equals.items():
            if combined.get(key) != expected:
                return False
        for key, expected in selector.metadata_contains.items():
            if not _contains(combined.get(key), expected):
                return False
        return True

    def _combined_metadata(
        self,
        top_level: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        combined = dict(top_level)
        combined.update(metadata)
        payload_metadata = metadata.get("metadata")
        if isinstance(payload_metadata, Mapping):
            combined.update(payload_metadata)
        fields = metadata.get("fields")
        if isinstance(fields, Mapping):
            combined.update(fields)
        return combined

    def _matches_query_event(self, event: MemoryEvent, query: MemoryQuery) -> bool:
        if query.task_id and event.task_id not in {None, query.task_id}:
            return False
        if query.worker_id and event.worker_id not in {None, query.worker_id}:
            return False
        if not query.include_control and event.is_control:
            return False
        if not query.include_model_derived and event.is_model_derived:
            return False
        if not query.include_synthesis and event.kind in {
            EventKind.SUMMARY_CREATED,
            EventKind.FINAL_REPORT_CREATED,
        }:
            return False
        return True

    def _matches_query_item(self, item: MemoryItem, query: MemoryQuery) -> bool:
        if query.task_id and item.task_id not in {None, query.task_id}:
            return False
        if query.worker_id and item.worker_id not in {None, query.worker_id}:
            return False
        if not query.include_control and item.is_control:
            return False
        if not query.include_model_derived and item.trust in MODEL_DERIVED_TRUST:
            return False
        if not query.include_synthesis and item.kind == MemoryItemKind.SYNTHESIS:
            return False
        return True

    def _section_kind(self, layer: MemoryLayerSpec) -> SectionKind:
        if layer.section_kind is not None:
            return layer.section_kind
        if layer.operation == MemoryRecipeOperation.COMPRESS:
            return SectionKind.SUMMARY
        return SectionKind.MEMORY

    def _section_trust(self, layer: MemoryLayerSpec) -> TrustLevel:
        if layer.section_trust is not None:
            return layer.section_trust
        if layer.operation == MemoryRecipeOperation.COMPRESS:
            return TrustLevel.MODEL_DERIVED
        return TrustLevel.UNKNOWN


def _contains(value: Any, expected: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return str(expected) in value
    if isinstance(value, Mapping):
        return str(expected) in {str(key) for key in value}
    if isinstance(value, (list, tuple, set)):
        return expected in value or str(expected) in {str(item) for item in value}
    return value == expected


def _exists(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return False
    return True


def _section_event_ids(sections: list[ContextSection]) -> tuple[str, ...]:
    return unique_tuple(
        event_id
        for section in sections
        for event_id in section.event_ids
    )


def _section_item_ids(sections: list[ContextSection]) -> tuple[str, ...]:
    return unique_tuple(
        item_id
        for section in sections
        for item_id in section.item_ids
    )


def _difference(values: tuple[str, ...], included: tuple[str, ...]) -> tuple[str, ...]:
    included_set = set(included)
    return tuple(value for value in values if value not in included_set)


def _event_line(event: MemoryEvent) -> str:
    text = event.text()
    if not text:
        text = "(empty)"
    prefix = event.kind.value
    if event.task_id:
        prefix = f"{prefix} task={event.task_id}"
    if event.worker_id:
        prefix = f"{prefix} worker={event.worker_id}"
    return f"- [{event.id}] {prefix}: {text}"


def _item_line(item: MemoryItem) -> str:
    prefix = item.kind.value
    if item.task_id:
        prefix = f"{prefix} task={item.task_id}"
    if item.worker_id:
        prefix = f"{prefix} worker={item.worker_id}"
    return f"- [{item.id}] {prefix}: {item.content}"


__all__ = [
    "MissingCapabilityError",
    "RecipeCapabilities",
    "RecipeEngine",
]
