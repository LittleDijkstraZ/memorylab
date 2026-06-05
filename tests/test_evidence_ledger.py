from memory_lab import (
    EvidenceLedgerMemory,
    EvidenceStatus,
    EvidenceTableRenderer,
    EventKind,
    MemoryEvent,
    MemoryQuery,
    TrustLevel,
)


def test_evidence_ledger_extracts_structured_evidence_and_missing_gaps():
    model = EvidenceLedgerMemory()
    events = (
        MemoryEvent(
            id="evt_tool",
            kind=EventKind.TOOL_RESULT,
            payload={
                "evidence": [
                    {
                        "claim_slot": "benchmark",
                        "content": "The paper reports a 12% improvement.",
                        "status": "supported",
                        "url": "https://example.test/paper",
                        "confidence": 0.9,
                    },
                ],
            },
        ),
        MemoryEvent(
            id="evt_review",
            kind=EventKind.EVIDENCE_REVIEW,
            payload={
                "evidence": [
                    {
                        "claim_slot": "deployment",
                        "content": "Deployment evidence refers to a different model.",
                        "status": "contradictory",
                        "source_refs": ["review:deployment"],
                    },
                ],
                "missing_evidence": [
                    {
                        "claim_slot": "safety",
                        "content": "Need independent safety evaluation.",
                    },
                ],
            },
        ),
    )

    state = model.ingest(model.initial_state(), events)
    packet = model.select_context(state, MemoryQuery(objective="verify evidence"))

    assert packet.section("evidence_supported") is not None
    assert packet.section("evidence_contradictory") is not None
    assert packet.section("evidence_missing") is not None
    assert set(packet.event_ids) == {"evt_tool", "evt_review"}
    supported = packet.section("evidence_supported")
    assert supported.metadata["rows"][0]["claim_slot"] == "benchmark"
    assert supported.metadata["rows"][0]["status"] == EvidenceStatus.SUPPORTED.value


def test_evidence_ledger_excludes_final_report_as_primary_evidence_by_default():
    model = EvidenceLedgerMemory()
    state = model.ingest(
        model.initial_state(),
        (
            MemoryEvent(
                id="evt_source",
                kind=EventKind.TOOL_RESULT,
                content="Raw source observation.",
            ),
            MemoryEvent(
                id="evt_final",
                kind=EventKind.FINAL_REPORT_CREATED,
                content="The final report says the claim is true.",
            ),
        ),
    )

    packet = model.select_context(state, MemoryQuery())

    assert "evt_source" in packet.event_ids
    assert "evt_final" not in packet.event_ids
    assert packet.warnings == (
        "model-derived summaries/final reports were excluded from primary evidence",
    )

    with_hints = model.select_context(state, MemoryQuery(include_synthesis=True))
    assert "evt_final" in with_hints.event_ids
    assert with_hints.section("synthesis_hints") is not None


def test_evidence_table_renderer_preserves_sources_and_trust_labels():
    model = EvidenceLedgerMemory()
    state = model.ingest(
        model.initial_state(),
        (
            MemoryEvent(
                id="evt_review",
                kind=EventKind.EVIDENCE_REVIEW,
                payload={
                    "evidence": [
                        {
                            "claim_slot": "slot-a",
                            "content": "Quoted evidence.",
                            "status": "direct",
                            "url": "https://example.test/source",
                        },
                    ],
                },
            ),
        ),
    )
    packet = model.select_context(state, MemoryQuery())
    rendered = EvidenceTableRenderer().render(packet)

    assert "| status | slot | trust | confidence | content | events | sources |" in rendered.text
    assert "slot-a" in rendered.text
    assert TrustLevel.EXTRACTED_EVIDENCE.value in rendered.text
    assert "https://example.test/source" in rendered.text
