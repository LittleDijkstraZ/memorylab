from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence

from memory_lab.schema import (
    ContextPacket,
    SectionKind,
    TrustLevel,
    _metadata,
    _tuple_str,
)


RECIPE_SCHEMA_VERSION = 1


class MemoryRecipeOperation(StrEnum):
    KEEP_RAW = "keep_raw"
    COMPRESS = "compress"
    RETRIEVE = "retrieve"


@dataclass(frozen=True)
class MemorySelector:
    event_ids: tuple[str, ...] = ()
    item_ids: tuple[str, ...] = ()
    task_id: str | None = None
    worker_id: str | None = None
    kinds: tuple[str, ...] = ()
    trust: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    recent_event_limit: int | None = None
    metadata_exists: tuple[str, ...] = ()
    metadata_equals: dict[str, Any] = field(default_factory=dict)
    metadata_contains: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_ids", _tuple_str(self.event_ids))
        object.__setattr__(self, "item_ids", _tuple_str(self.item_ids))
        object.__setattr__(self, "kinds", _tuple_str(self.kinds))
        object.__setattr__(self, "trust", _tuple_str(self.trust))
        object.__setattr__(self, "tags", _tuple_str(self.tags))
        object.__setattr__(self, "metadata_exists", _tuple_str(self.metadata_exists))
        object.__setattr__(self, "metadata_equals", _metadata(self.metadata_equals))
        object.__setattr__(
            self,
            "metadata_contains",
            _metadata(self.metadata_contains),
        )
        if self.recent_event_limit is not None and self.recent_event_limit < 1:
            raise ValueError("recent_event_limit must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_ids": list(self.event_ids),
            "item_ids": list(self.item_ids),
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "kinds": list(self.kinds),
            "trust": list(self.trust),
            "tags": list(self.tags),
            "recent_event_limit": self.recent_event_limit,
            "metadata_exists": list(self.metadata_exists),
            "metadata_equals": dict(self.metadata_equals),
            "metadata_contains": dict(self.metadata_contains),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> MemorySelector:
        if not data:
            return cls()
        return cls(
            event_ids=tuple(data.get("event_ids") or ()),
            item_ids=tuple(data.get("item_ids") or ()),
            task_id=data.get("task_id"),
            worker_id=data.get("worker_id"),
            kinds=tuple(data.get("kinds") or ()),
            trust=tuple(data.get("trust") or ()),
            tags=tuple(data.get("tags") or ()),
            recent_event_limit=data.get("recent_event_limit"),
            metadata_exists=tuple(data.get("metadata_exists") or ()),
            metadata_equals=dict(data.get("metadata_equals") or {}),
            metadata_contains=dict(data.get("metadata_contains") or {}),
        )


@dataclass(frozen=True)
class MemoryLayerSpec:
    id: str
    title: str
    operation: MemoryRecipeOperation
    selector: MemorySelector = field(default_factory=MemorySelector)
    instruction: str = ""
    budget_tokens: int | None = None
    include_provenance: bool = True
    section_kind: SectionKind | None = None
    section_trust: TrustLevel | None = None
    requires_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("MemoryLayerSpec.id must be non-empty")
        if not self.title:
            object.__setattr__(self, "title", self.id.replace("_", " ").title())
        operation = (
            self.operation
            if isinstance(self.operation, MemoryRecipeOperation)
            else MemoryRecipeOperation(str(self.operation))
        )
        selector = (
            self.selector
            if isinstance(self.selector, MemorySelector)
            else MemorySelector.from_dict(self.selector)
        )
        section_kind = (
            self.section_kind
            if self.section_kind is None or isinstance(self.section_kind, SectionKind)
            else SectionKind(str(self.section_kind))
        )
        section_trust = (
            self.section_trust
            if self.section_trust is None or isinstance(self.section_trust, TrustLevel)
            else TrustLevel(str(self.section_trust))
        )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "selector", selector)
        object.__setattr__(self, "section_kind", section_kind)
        object.__setattr__(self, "section_trust", section_trust)
        object.__setattr__(
            self,
            "requires_capabilities",
            _tuple_str(self.requires_capabilities),
        )
        object.__setattr__(
            self,
            "optional_capabilities",
            _tuple_str(self.optional_capabilities),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "operation": self.operation.value,
            "selector": self.selector.to_dict(),
            "instruction": self.instruction,
            "budget_tokens": self.budget_tokens,
            "include_provenance": self.include_provenance,
            "section_kind": self.section_kind.value if self.section_kind else None,
            "section_trust": self.section_trust.value if self.section_trust else None,
            "requires_capabilities": list(self.requires_capabilities),
            "optional_capabilities": list(self.optional_capabilities),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryLayerSpec:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or data["id"]),
            operation=MemoryRecipeOperation(data["operation"]),
            selector=MemorySelector.from_dict(data.get("selector")),
            instruction=str(data.get("instruction") or ""),
            budget_tokens=data.get("budget_tokens"),
            include_provenance=bool(data.get("include_provenance", True)),
            section_kind=(
                SectionKind(data["section_kind"])
                if data.get("section_kind")
                else None
            ),
            section_trust=(
                TrustLevel(data["section_trust"])
                if data.get("section_trust")
                else None
            ),
            requires_capabilities=tuple(data.get("requires_capabilities") or ()),
            optional_capabilities=tuple(data.get("optional_capabilities") or ()),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MemoryRecipe:
    name: str
    layers: tuple[MemoryLayerSpec, ...]
    description: str = ""
    selector: MemorySelector = field(default_factory=MemorySelector)
    renderer: str | None = None
    requires_capabilities: tuple[str, ...] = ()
    schema_version: int = RECIPE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != RECIPE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported recipe schema version {self.schema_version}; "
                f"expected {RECIPE_SCHEMA_VERSION}",
            )
        if not self.name:
            raise ValueError("MemoryRecipe.name must be non-empty")
        layers = tuple(
            layer if isinstance(layer, MemoryLayerSpec) else MemoryLayerSpec.from_dict(layer)
            for layer in self.layers
        )
        if not layers:
            raise ValueError("MemoryRecipe.layers must not be empty")
        selector = (
            self.selector
            if isinstance(self.selector, MemorySelector)
            else MemorySelector.from_dict(self.selector)
        )
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "selector", selector)
        object.__setattr__(
            self,
            "requires_capabilities",
            _tuple_str(self.requires_capabilities),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "selector": self.selector.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
            "renderer": self.renderer,
            "requires_capabilities": list(self.requires_capabilities),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryRecipe:
        return cls(
            schema_version=int(
                data.get("schema_version", RECIPE_SCHEMA_VERSION),
            ),
            name=str(data["name"]),
            description=str(data.get("description") or ""),
            selector=MemorySelector.from_dict(data.get("selector")),
            layers=tuple(
                MemoryLayerSpec.from_dict(layer)
                for layer in data.get("layers", ())
            ),
            renderer=data.get("renderer"),
            requires_capabilities=tuple(data.get("requires_capabilities") or ()),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CapabilityTrace:
    layer_id: str
    operation: MemoryRecipeOperation
    capability: str
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = (
            self.operation
            if isinstance(self.operation, MemoryRecipeOperation)
            else MemoryRecipeOperation(str(self.operation))
        )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "warnings", _tuple_str(self.warnings))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "operation": self.operation.value,
            "capability": self.capability,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityTrace:
        return cls(
            layer_id=str(data["layer_id"]),
            operation=MemoryRecipeOperation(data["operation"]),
            capability=str(data.get("capability") or ""),
            warnings=tuple(data.get("warnings") or ()),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MemoryDeriveResult:
    packet: ContextPacket
    traces: tuple[CapabilityTrace, ...] = ()
    recipe_name: str = ""
    recipe_schema_version: int = RECIPE_SCHEMA_VERSION
    selected_event_ids: tuple[str, ...] = ()
    selected_item_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "traces",
            tuple(
                trace
                if isinstance(trace, CapabilityTrace)
                else CapabilityTrace.from_dict(trace)
                for trace in self.traces
            ),
        )
        object.__setattr__(
            self,
            "selected_event_ids",
            _tuple_str(self.selected_event_ids),
        )
        object.__setattr__(
            self,
            "selected_item_ids",
            _tuple_str(self.selected_item_ids),
        )
        object.__setattr__(self, "warnings", _tuple_str(self.warnings))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet": self.packet.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "recipe_name": self.recipe_name,
            "recipe_schema_version": self.recipe_schema_version,
            "selected_event_ids": list(self.selected_event_ids),
            "selected_item_ids": list(self.selected_item_ids),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryDeriveResult:
        return cls(
            packet=ContextPacket.from_dict(data["packet"]),
            traces=tuple(
                CapabilityTrace.from_dict(trace)
                for trace in data.get("traces", ())
            ),
            recipe_name=str(data.get("recipe_name") or ""),
            recipe_schema_version=int(
                data.get("recipe_schema_version", RECIPE_SCHEMA_VERSION),
            ),
            selected_event_ids=tuple(data.get("selected_event_ids") or ()),
            selected_item_ids=tuple(data.get("selected_item_ids") or ()),
            warnings=tuple(data.get("warnings") or ()),
            metadata=dict(data.get("metadata") or {}),
        )


def trust_values(values: Sequence[str | TrustLevel]) -> tuple[str, ...]:
    return tuple(
        value.value if isinstance(value, TrustLevel) else str(value)
        for value in values
    )


__all__ = [
    "CapabilityTrace",
    "MemoryDeriveResult",
    "MemoryLayerSpec",
    "MemoryRecipe",
    "MemoryRecipeOperation",
    "MemorySelector",
    "RECIPE_SCHEMA_VERSION",
    "trust_values",
]
