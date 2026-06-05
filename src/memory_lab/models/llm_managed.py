from __future__ import annotations

from typing import Callable, Sequence

from memory_lab.models.base import BaseMemoryModel
from memory_lab.models.full_history import FullHistoryMemory
from memory_lab.schema import (
    ContextPacket,
    MemoryEvent,
    MemoryItem,
    MemoryOperation,
    MemoryOperationKind,
    MemoryQuery,
    MemoryState,
    Validity,
)


MemoryManager = Callable[
    [MemoryState, Sequence[MemoryEvent]],
    Sequence[MemoryOperation],
]


class LLMManagedMemory(BaseMemoryModel):
    """Applies explicit memory operations from an injected manager.

    The manager can be a real LLM policy later, but tests should use a fake
    manager that returns deterministic `MemoryOperation` objects.
    """

    name = "llm_managed"

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self.manager = manager or self._noop_manager
        self._fallback = FullHistoryMemory()

    def ingest(
        self,
        state: MemoryState,
        events: Sequence[MemoryEvent],
    ) -> MemoryState:
        state, new_events = self._append_new_events(state, events)
        operations = tuple(self.manager(state, new_events))
        items = list(state.items)
        for operation in operations:
            items = self._apply_operation(items, operation)
        return MemoryState(
            model_name=self.name,
            events=state.events,
            items=tuple(items),
            metadata={
                **state.metadata,
                "operation_count": state.metadata.get("operation_count", 0)
                + len(operations),
                "last_operations": [operation.to_dict() for operation in operations],
            },
        )

    def select_context(self, state: MemoryState, query: MemoryQuery) -> ContextPacket:
        fallback_state = MemoryState(
            model_name=self._fallback.name,
            events=state.events,
            items=state.items,
            metadata=state.metadata,
        )
        packet = self._fallback.select_context(fallback_state, query)
        return ContextPacket(
            id=packet.id,
            query=packet.query,
            sections=packet.sections,
            warnings=(
                *packet.warnings,
                "llm_managed memory uses explicit operations; verify operation logs before trusting edits",
            ),
            metadata={**packet.metadata, "model": self.name},
        )

    def _apply_operation(
        self,
        items: list[MemoryItem],
        operation: MemoryOperation,
    ) -> list[MemoryItem]:
        if operation.kind == MemoryOperationKind.IGNORE:
            return items
        if operation.kind in {MemoryOperationKind.CREATE, MemoryOperationKind.COMPRESS}:
            if operation.item is None:
                return items
            return [*items, operation.item]
        if operation.item_id is None:
            return items
        if operation.kind == MemoryOperationKind.UPDATE:
            return [
                item.with_updates(content=operation.content)
                if item.id == operation.item_id and operation.content is not None
                else item
                for item in items
            ]
        if operation.kind == MemoryOperationKind.INVALIDATE:
            return [
                item.with_updates(
                    validity=Validity.INVALIDATED.value,
                    metadata={
                        **item.metadata,
                        "invalidated_reason": operation.reason,
                        "invalidated_by_operation": operation.id,
                    },
                )
                if item.id == operation.item_id
                else item
                for item in items
            ]
        return items

    def _noop_manager(
        self,
        _state: MemoryState,
        _events: Sequence[MemoryEvent],
    ) -> Sequence[MemoryOperation]:
        return ()
