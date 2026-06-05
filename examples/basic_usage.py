from memory_lab import (
    EventKind,
    FullHistoryMemory,
    MemoryEvent,
    MemoryQuery,
    ResearchBriefRenderer,
)


def main() -> None:
    model = FullHistoryMemory()
    state = model.initial_state()
    state = model.ingest(
        state,
        (
            MemoryEvent(
                id="evt_run",
                kind=EventKind.RUN_STARTED,
                content="Verify whether the claim is supported.",
            ),
            MemoryEvent(
                id="evt_reasoning",
                kind=EventKind.REASONING_NOTE,
                content="I should start with primary sources.",
            ),
            MemoryEvent(
                id="evt_control",
                kind=EventKind.CONTROL_STATE,
                content="Call a tool before final synthesis.",
            ),
        ),
    )
    packet = model.select_context(state, MemoryQuery(objective="verify claim"))
    rendered = ResearchBriefRenderer().render(packet)
    print(rendered.text)


if __name__ == "__main__":
    main()
