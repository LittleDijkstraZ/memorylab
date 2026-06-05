from __future__ import annotations

from typing import Protocol, Sequence

from memory_lab.schema import ContextPacket, MemoryEvent, MemoryQuery, MemoryState


class MemoryModel(Protocol):
    name: str

    def initial_state(self) -> MemoryState:
        ...

    def ingest(
        self,
        state: MemoryState,
        events: Sequence[MemoryEvent],
    ) -> MemoryState:
        ...

    def select_context(self, state: MemoryState, query: MemoryQuery) -> ContextPacket:
        ...


class BaseMemoryModel:
    name = "base"

    def initial_state(self) -> MemoryState:
        return MemoryState(model_name=self.name)

    def ingest(
        self,
        state: MemoryState,
        events: Sequence[MemoryEvent],
    ) -> MemoryState:
        raise NotImplementedError

    def select_context(self, state: MemoryState, query: MemoryQuery) -> ContextPacket:
        raise NotImplementedError

    def _append_new_events(
        self,
        state: MemoryState,
        events: Sequence[MemoryEvent],
    ) -> tuple[MemoryState, tuple[MemoryEvent, ...]]:
        existing = {event.id for event in state.events}
        new_events = tuple(event for event in events if event.id not in existing)
        return state.with_events(new_events), new_events
