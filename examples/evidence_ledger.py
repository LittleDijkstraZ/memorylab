from memory_lab import (
    EventKind,
    EvidenceLedgerMemory,
    EvidenceTableRenderer,
    MemoryEvent,
    MemoryQuery,
)


def main() -> None:
    model = EvidenceLedgerMemory()
    state = model.initial_state()
    state = model.ingest(
        state,
        (
            MemoryEvent(
                id="evt_source",
                kind=EventKind.TOOL_RESULT,
                payload={
                    "evidence": [
                        {
                            "claim_slot": "benchmark",
                            "content": "The source reports a 12% improvement.",
                            "status": "supported",
                            "url": "https://example.test/paper",
                            "confidence": 0.9,
                        },
                    ],
                },
            ),
            MemoryEvent(
                id="evt_gap",
                kind=EventKind.EVIDENCE_REVIEW,
                payload={
                    "missing_evidence": [
                        {
                            "claim_slot": "safety",
                            "content": "Need an independent safety evaluation.",
                        },
                    ],
                },
            ),
        ),
    )
    packet = model.select_context(state, MemoryQuery(objective="audit evidence"))
    rendered = EvidenceTableRenderer().render(packet)
    print(rendered.text)


if __name__ == "__main__":
    main()
