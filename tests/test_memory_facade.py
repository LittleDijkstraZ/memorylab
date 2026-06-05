from memory_lab import Memory, MemoryEvent, EventKind


def test_memory_facade_updates_and_reads_full_history_text():
    memory = Memory()
    memory.update("Verify whether the claim is supported.")
    memory.update("I should start with primary sources.")

    rendered = memory.read(objective="verify claim")

    assert "Verify whether the claim is supported." in rendered.text
    assert "I should start with primary sources." in rendered.text


def test_memory_facade_accepts_generic_dict_updates():
    memory = Memory()
    memory.update(
        {
            "content": "The user is comparing memory strategies.",
            "metadata": {"project": "memorylab"},
        },
    )

    packet = memory.packet()
    work_history = packet.section("work_history")

    assert work_history is not None
    assert work_history.metadata["rows"][0]["metadata"]["event_kind"] == "reasoning_note"


def test_memory_facade_keeps_evidence_ledger_generic_by_default():
    memory = Memory("evidence_ledger")
    memory.update(
        [
            "A source reports a 12% improvement on the benchmark.",
            "No independent safety evaluation has been found yet.",
        ],
    )

    rendered = memory.read(objective="audit evidence")

    assert "12% improvement" in rendered.text
    assert "safety evaluation" in rendered.text


def test_memory_facade_still_accepts_structured_evidence_for_adapters():
    memory = Memory("evidence_ledger")
    memory.update(
        {
            "kind": "evidence",
            "slot": "benchmark",
            "content": "The source reports a 12% improvement.",
            "source": "https://example.test/paper",
            "status": "supported",
        },
    )

    rendered = memory.read()

    assert "benchmark" in rendered.text
    assert "https://example.test/paper" in rendered.text


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
