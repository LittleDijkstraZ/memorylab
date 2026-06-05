from pathlib import Path

from memory_lab import (
    EventKind,
    FullHistoryMemory,
    HierarchicalSummaryMemory,
    LLMManagedMemory,
    MemoryEvent,
    MemoryItem,
    MemoryItemKind,
    MemoryOperation,
    MemoryOperationKind,
    MemoryQuery,
    Provenance,
    RollingSummaryMemory,
    TrustLevel,
    Validity,
    compare_packet,
)


def test_rolling_summary_keeps_recent_events_and_deterministic_summary():
    model = RollingSummaryMemory(recent_event_limit=2)
    events = tuple(
        MemoryEvent(
            id=f"evt_{index}",
            kind=EventKind.REASONING_NOTE,
            content=f"note {index}",
        )
        for index in range(5)
    )
    state = model.ingest(model.initial_state(), events)
    packet = model.select_context(state, MemoryQuery())

    assert packet.section("rolling_summary") is not None
    assert packet.section("recent_events") is not None
    assert packet.section("rolling_summary").event_ids == ("evt_0", "evt_1", "evt_2")
    assert packet.section("recent_events").event_ids == ("evt_3", "evt_4")


def test_hierarchical_summary_groups_by_task_and_worker():
    model = HierarchicalSummaryMemory()
    state = model.ingest(
        model.initial_state(),
        (
            MemoryEvent(id="evt_root", kind=EventKind.RUN_STARTED, content="root"),
            MemoryEvent(
                id="evt_task",
                kind=EventKind.TOOL_RESULT,
                content="task result",
                task_id="task-a",
            ),
            MemoryEvent(
                id="evt_worker",
                kind=EventKind.WORKER_RESULT,
                content="worker result",
                task_id="task-a",
                worker_id="worker-1",
            ),
        ),
    )
    packet = model.select_context(state, MemoryQuery())

    assert packet.section("hierarchy_root") is not None
    assert packet.section("hierarchy_task:task-a") is not None
    assert packet.section("hierarchy_worker:worker-1") is not None


def test_llm_managed_memory_uses_injected_operations_without_live_model():
    created = MemoryItem(
        id="mem_note",
        kind=MemoryItemKind.SUMMARY,
        content="Manager-created note.",
        provenance=Provenance(event_ids=("evt_1",)),
        trust=TrustLevel.MODEL_DERIVED,
    )

    def fake_manager(_state, _events):
        return (
            MemoryOperation(
                id="op_create",
                kind=MemoryOperationKind.CREATE,
                provenance=Provenance(event_ids=("evt_1",)),
                item=created,
            ),
            MemoryOperation(
                id="op_invalidate",
                kind=MemoryOperationKind.INVALIDATE,
                provenance=Provenance(event_ids=("evt_2",)),
                item_id="mem_note",
                reason="superseded",
            ),
        )

    model = LLMManagedMemory(manager=fake_manager)
    state = model.ingest(
        model.initial_state(),
        (
            MemoryEvent(id="evt_1", kind=EventKind.REASONING_NOTE, content="first"),
            MemoryEvent(id="evt_2", kind=EventKind.REASONING_NOTE, content="second"),
        ),
    )

    assert state.items[0].id == "mem_note"
    assert state.items[0].validity == Validity.INVALIDATED
    assert state.metadata["operation_count"] == 2
    packet = model.select_context(state, MemoryQuery())
    assert "verify operation logs" in " ".join(packet.warnings)


def test_packet_comparison_reports_coverage_and_budget():
    model = FullHistoryMemory()
    state = model.ingest(
        model.initial_state(),
        (
            MemoryEvent(id="evt_a", kind=EventKind.REASONING_NOTE, content="alpha"),
            MemoryEvent(id="evt_b", kind=EventKind.REASONING_NOTE, content="beta"),
        ),
    )
    packet = model.select_context(state, MemoryQuery())
    comparison = compare_packet(
        packet,
        expected_event_ids=("evt_a", "evt_b", "evt_c"),
        budget_tokens=1,
    )

    assert comparison.event_coverage == 2 / 3
    assert comparison.missing_event_ids == ("evt_c",)
    assert comparison.over_budget


def test_memory_lab_core_does_not_import_jay2_symbols():
    root = Path(__file__).resolve().parents[1]
    python_files = [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]
    forbidden = ("import jay2", "from jay2")
    offenders = []
    for path in python_files:
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in forbidden):
            offenders.append(path.relative_to(root).as_posix())

    assert offenders == []
