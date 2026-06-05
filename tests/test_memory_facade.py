from memory_lab import Memory, MemoryEvent, EventKind


def test_memory_facade_updates_and_reads_full_history_text():
    memory = Memory("full_history")
    memory.update("Verify whether the claim is supported.", kind="run_started")
    memory.update("I should start with primary sources.", kind="note")

    rendered = memory.read(objective="verify claim")

    assert "Verify whether the claim is supported." in rendered.text
    assert "I should start with primary sources." in rendered.text


def test_memory_facade_accepts_batch_dict_updates_for_evidence_ledger():
    memory = Memory("evidence_ledger")
    memory.update(
        [
            {
                "kind": "evidence",
                "slot": "benchmark",
                "content": "The source reports a 12% improvement.",
                "source": "https://example.test/paper",
                "status": "supported",
                "confidence": 0.9,
            },
            {
                "kind": "missing",
                "slot": "safety",
                "content": "Need independent safety evaluation.",
            },
        ],
    )

    rendered = memory.read(objective="audit evidence")

    assert "benchmark" in rendered.text
    assert "https://example.test/paper" in rendered.text
    assert "safety" in rendered.text


def test_memory_facade_keeps_advanced_memory_event_path():
    memory = Memory("full_history")
    memory.update(
        MemoryEvent(
            id="evt_control",
            kind=EventKind.CONTROL_STATE,
            content="Call a tool before final synthesis.",
        ),
    )

    packet = memory.packet()

    assert packet.section("control_state") is not None
