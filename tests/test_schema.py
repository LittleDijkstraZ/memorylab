from memory_lab import (
    ContextPacket,
    ContextSection,
    EventKind,
    EvidenceStatus,
    MemoryEvent,
    MemoryItem,
    MemoryItemKind,
    MemoryPhase,
    MemoryQuery,
    MemoryState,
    Provenance,
    SectionKind,
    TrustLevel,
)


def test_memory_schema_round_trips_with_enums_and_provenance():
    event = MemoryEvent(
        id="evt_tool",
        kind=EventKind.TOOL_RESULT,
        content="A source says the benchmark improved.",
        payload={"url": "https://example.test/paper"},
        provenance=Provenance(source_refs=("https://example.test/paper",)),
        task_id="task-1",
    )
    item = MemoryItem(
        id="mem_evidence",
        kind=MemoryItemKind.EVIDENCE,
        content="Benchmark improved on the held-out split.",
        provenance=Provenance(event_ids=(event.id,), source_refs=("paper:1",)),
        trust=TrustLevel.EXTRACTED_EVIDENCE,
        evidence_status=EvidenceStatus.SUPPORTED,
        claim_slot="benchmark",
        confidence=0.8,
    )
    query = MemoryQuery(phase=MemoryPhase.REASONING, objective="verify claim")
    section = ContextSection(
        id="evidence_supported",
        title="Evidence: supported",
        kind=SectionKind.EVIDENCE,
        trust=TrustLevel.EXTRACTED_EVIDENCE,
        content=item.content,
        item_ids=(item.id,),
        event_ids=(event.id,),
    )
    packet = ContextPacket(query=query, sections=(section,), id="pkt_1")
    state = MemoryState(model_name="test", events=(event,), items=(item,))

    assert MemoryEvent.from_dict(event.to_dict()) == event
    assert MemoryItem.from_dict(item.to_dict()) == item
    assert MemoryState.from_dict(state.to_dict()) == state
    assert ContextPacket.from_dict(packet.to_dict()) == packet

    restored = MemoryEvent.from_dict(event.to_dict())
    assert restored.trust == TrustLevel.SOURCE_OBSERVATION
    assert restored.is_source_bearing
    assert item.is_primary_evidence


def test_control_event_is_not_source_bearing():
    event = MemoryEvent(
        id="evt_control",
        kind=EventKind.CONTROL_STATE,
        content="Call a tool before final answer.",
    )

    assert event.is_control
    assert not event.is_source_bearing
    assert event.trust == TrustLevel.SYSTEM_CONTROL
