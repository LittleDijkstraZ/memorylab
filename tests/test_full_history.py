from memory_lab import (
    CompactPromptRenderer,
    EventKind,
    FullHistoryMemory,
    MemoryEvent,
    MemoryQuery,
    ResearchBriefRenderer,
    SectionKind,
)


def test_full_history_keeps_control_state_separate_from_work_history():
    model = FullHistoryMemory()
    events = (
        MemoryEvent(id="evt_run", kind=EventKind.RUN_STARTED, content="Verify claim."),
        MemoryEvent(
            id="evt_reason",
            kind=EventKind.REASONING_NOTE,
            content="I should search for primary sources.",
        ),
        MemoryEvent(
            id="evt_tool",
            kind=EventKind.TOOL_RESULT,
            content="Primary source found.",
            task_id="root",
        ),
        MemoryEvent(
            id="evt_control",
            kind=EventKind.CONTROL_STATE,
            content="Must call a tool or generate response.",
        ),
    )
    state = model.ingest(model.initial_state(), events)
    packet = model.select_context(state, MemoryQuery(objective="verify claim"))

    assert packet.section("task_state") is not None
    work_history = packet.section("work_history")
    control = packet.section("control_state")
    assert work_history is not None
    assert control is not None
    assert work_history.kind == SectionKind.MEMORY
    assert control.kind == SectionKind.CONTROL
    assert "evt_reason" in work_history.event_ids
    assert "evt_tool" in work_history.event_ids
    assert "evt_control" not in work_history.event_ids
    assert control.event_ids == ("evt_control",)


def test_full_history_renderers_report_budget_omissions():
    model = FullHistoryMemory()
    events = tuple(
        MemoryEvent(
            id=f"evt_{index}",
            kind=EventKind.REASONING_NOTE,
            content="long reasoning note " * 20,
        )
        for index in range(3)
    )
    state = model.ingest(model.initial_state(), events)
    packet = model.select_context(state, MemoryQuery(objective="compact"))

    rendered = ResearchBriefRenderer().render(packet, budget_tokens=10)

    assert rendered.text
    assert rendered.warnings
    assert "truncated" in " ".join(rendered.warnings)

    compact = CompactPromptRenderer().render(packet)
    assert "## Work History" in compact.text
