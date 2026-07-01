import pytest

from memory_lab import (
    EventKind,
    CapabilityResponse,
    EvidenceStatus,
    InjectedEmbeddingRetriever,
    InjectedLLMCompressor,
    LayeredPromptRenderer,
    Memory,
    MemoryEvent,
    MemoryItem,
    MemoryItemKind,
    MemoryLayerSpec,
    MemoryQuery,
    MemoryRecipe,
    MemoryRecipeOperation,
    MemorySelector,
    MemoryState,
    MissingCapabilityError,
    Provenance,
    RecipeCapabilities,
    RecipeEngine,
    SectionKind,
    TrustLevel,
    layered_checkpoint,
    one_pot_compression,
    retrieval_context,
)


def test_recipe_schema_round_trips_with_layer_capabilities():
    recipe = MemoryRecipe(
        name="unit_recipe",
        requires_capabilities=("keep_raw",),
        layers=(
            MemoryLayerSpec(
                id="summary",
                title="Summary",
                operation=MemoryRecipeOperation.COMPRESS,
                selector=MemorySelector(task_id="task-a"),
                requires_capabilities=("compress",),
                optional_capabilities=("retrieve",),
                budget_tokens=50,
                section_kind=SectionKind.SUMMARY,
                section_trust=TrustLevel.MODEL_DERIVED,
            ),
        ),
    )

    restored = MemoryRecipe.from_dict(recipe.to_dict())

    assert restored == recipe
    assert restored.layers[0].requires_capabilities == ("compress",)
    assert restored.layers[0].optional_capabilities == ("retrieve",)
    assert restored.layers[0].section_kind == SectionKind.SUMMARY
    assert restored.layers[0].section_trust == TrustLevel.MODEL_DERIVED


def test_derive_is_read_only_and_preserves_event_provenance():
    memory = Memory("full_history")
    memory.update(
        [
            MemoryEvent(
                id="evt_a",
                kind=EventKind.REASONING_NOTE,
                content="Alpha decision for the investigation.",
                task_id="task-a",
            ),
            MemoryEvent(
                id="evt_b",
                kind=EventKind.TOOL_RESULT,
                content="Primary source says beta.",
                task_id="task-a",
            ),
        ],
    )
    before = memory.state

    result = memory.derive(
        one_pot_compression(selector=MemorySelector(task_id="task-a")),
        objective="alpha beta",
    )

    assert memory.state == before
    assert result.packet.section("compressed_context") is not None
    assert result.selected_event_ids == ("evt_a", "evt_b")
    assert result.packet.section("compressed_context").event_ids == (
        "evt_a",
        "evt_b",
    )
    assert result.traces[0].capability == "deterministic_compressor"


def test_required_capability_missing_fails_closed():
    recipe = MemoryRecipe(
        name="needs_compress",
        layers=(
            MemoryLayerSpec(
                id="summary",
                title="Summary",
                operation=MemoryRecipeOperation.COMPRESS,
                requires_capabilities=("compress",),
            ),
        ),
    )
    state = Memory("full_history").state
    capabilities = RecipeCapabilities(compressor=None, retriever=None)

    with pytest.raises(MissingCapabilityError):
        RecipeEngine().derive(state, MemoryQuery(), recipe, capabilities=capabilities)


def test_missing_optional_operation_fails_when_fallback_disabled():
    recipe = MemoryRecipe(
        name="compress_without_fallback",
        layers=(
            MemoryLayerSpec(
                id="summary",
                title="Summary",
                operation=MemoryRecipeOperation.COMPRESS,
            ),
        ),
    )
    state = Memory("full_history").state
    capabilities = RecipeCapabilities(
        compressor=None,
        retriever=None,
        allow_deterministic_fallback=False,
    )

    with pytest.raises(MissingCapabilityError):
        RecipeEngine().derive(state, MemoryQuery(), recipe, capabilities=capabilities)


def test_optional_capability_missing_uses_deterministic_fallback_warning():
    memory = Memory("full_history")
    memory.update("Need a concise summary.")
    recipe = MemoryRecipe(
        name="optional_compress",
        layers=(
            MemoryLayerSpec(
                id="summary",
                title="Summary",
                operation=MemoryRecipeOperation.COMPRESS,
                optional_capabilities=("compress",),
            ),
        ),
    )
    capabilities = RecipeCapabilities(compressor=None, retriever=None)

    result = memory.derive(recipe, capabilities=capabilities)

    assert result.packet.section("summary") is not None
    assert "deterministic fallback" in " ".join(result.warnings)
    assert result.traces[0].capability == "deterministic_compressor"


def test_injected_llm_compressor_drives_compression_layer_with_provenance():
    memory = Memory("full_history")
    memory.update(
        [
            MemoryEvent(
                id="evt_goal",
                kind=EventKind.REASONING_NOTE,
                content="Need to summarize calibration results.",
            ),
            MemoryEvent(
                id="evt_source",
                kind=EventKind.TOOL_RESULT,
                content="Source says calibration improved.",
            ),
        ],
    )

    def fake_llm(request):
        assert request.layer_id == "compressed_context"
        assert tuple(event.id for event in request.events) == (
            "evt_goal",
            "evt_source",
        )
        return "LLM summary: calibration improved."

    result = memory.derive(
        one_pot_compression(),
        capabilities=RecipeCapabilities(
            compressor=InjectedLLMCompressor(fake_llm),
        ),
    )

    section = result.packet.section("compressed_context")
    assert section.content == "LLM summary: calibration improved."
    assert section.kind == SectionKind.SUMMARY
    assert section.trust == TrustLevel.MODEL_DERIVED
    assert section.event_ids == ("evt_goal", "evt_source")
    assert result.selected_event_ids == ("evt_goal", "evt_source")
    assert result.traces[0].capability == "injected_llm_compressor"
    assert result.traces[0].metadata["method"] == "injected_llm_compressor"


def test_injected_llm_response_gets_default_provenance_and_method_metadata():
    memory = Memory("full_history")
    memory.update(
        MemoryEvent(
            id="evt_input",
            kind=EventKind.REASONING_NOTE,
            content="Input event for an injected compressor.",
        ),
    )

    def fake_llm(_request):
        return CapabilityResponse(
            content="Provider-shaped summary.",
            metadata={"provider": "fake"},
        )

    result = memory.derive(
        one_pot_compression(),
        capabilities=RecipeCapabilities(
            compressor=InjectedLLMCompressor(fake_llm),
        ),
    )

    section = result.packet.section("compressed_context")
    assert section.event_ids == ("evt_input",)
    assert section.metadata["provider"] == "fake"
    assert section.metadata["method"] == "injected_llm_compressor"
    assert result.traces[0].metadata["method"] == "injected_llm_compressor"


def test_failed_injected_llm_compression_falls_back_deterministically():
    memory = Memory("full_history")
    memory.update(
        MemoryEvent(
            id="evt_keep",
            kind=EventKind.REASONING_NOTE,
            content="Fallback must preserve this original event.",
        ),
    )

    def broken_llm(_request):
        raise RuntimeError("provider offline")

    result = memory.derive(
        one_pot_compression(),
        capabilities=RecipeCapabilities(
            compressor=InjectedLLMCompressor(broken_llm),
        ),
    )

    section = result.packet.section("compressed_context")
    assert "Fallback must preserve this original event." in section.content
    assert section.event_ids == ("evt_keep",)
    assert section.trust == TrustLevel.MODEL_DERIVED
    assert result.traces[0].capability == "deterministic_compressor"
    assert result.traces[0].metadata["fallback_from"] == "injected_llm_compressor"
    assert result.traces[0].metadata["fallback_error_type"] == "RuntimeError"
    assert "provider offline" in result.traces[0].metadata["fallback_error"]
    assert "used deterministic fallback" in " ".join(result.warnings)


def test_failed_injected_llm_compression_propagates_when_fallback_disabled():
    memory = Memory("full_history")
    memory.update("This should not be silently summarized.")

    def broken_llm(_request):
        raise RuntimeError("provider offline")

    with pytest.raises(RuntimeError, match="provider offline"):
        memory.derive(
            one_pot_compression(),
            capabilities=RecipeCapabilities(
                compressor=InjectedLLMCompressor(broken_llm),
                allow_deterministic_fallback=False,
            ),
        )


def test_selector_event_ids_limit_items_by_provenance():
    memory = Memory("full_history")
    memory.update(
        [
            MemoryEvent(
                id="evt_a",
                kind=EventKind.TOOL_RESULT,
                content="alpha",
            ),
            MemoryEvent(
                id="evt_b",
                kind=EventKind.TOOL_RESULT,
                content="beta",
            ),
        ],
    )
    item_id = next(
        item.id
        for item in memory.state.items
        if item.provenance.event_ids == ("evt_b",)
    )
    recipe = MemoryRecipe(
        name="event_scope",
        layers=(
            MemoryLayerSpec(
                id="raw",
                title="Raw",
                operation=MemoryRecipeOperation.KEEP_RAW,
                selector=MemorySelector(event_ids=("evt_b",)),
            ),
        ),
    )

    result = memory.derive(recipe)

    section = result.packet.section("raw")
    assert section.event_ids == ("evt_b",)
    assert section.item_ids == (item_id,)
    assert "evt_a" not in section.content


def test_selector_item_ids_limit_events_by_provenance():
    memory = Memory("full_history")
    memory.update(
        [
            MemoryEvent(
                id="evt_a",
                kind=EventKind.TOOL_RESULT,
                content="alpha",
            ),
            MemoryEvent(
                id="evt_b",
                kind=EventKind.TOOL_RESULT,
                content="beta",
            ),
        ],
    )
    item_id = next(
        item.id
        for item in memory.state.items
        if item.provenance.event_ids == ("evt_b",)
    )
    recipe = MemoryRecipe(
        name="item_scope",
        layers=(
            MemoryLayerSpec(
                id="raw",
                title="Raw",
                operation=MemoryRecipeOperation.KEEP_RAW,
                selector=MemorySelector(item_ids=(item_id,)),
            ),
        ),
    )

    result = memory.derive(recipe)

    section = result.packet.section("raw")
    assert section.event_ids == ("evt_b",)
    assert section.item_ids == (item_id,)
    assert "evt_a" not in section.content


def test_derive_task_id_query_scopes_base_state():
    memory = Memory("full_history")
    memory.update(
        [
            MemoryEvent(
                id="evt_a",
                kind=EventKind.REASONING_NOTE,
                content="task a only",
                task_id="task-a",
            ),
            MemoryEvent(
                id="evt_b",
                kind=EventKind.REASONING_NOTE,
                content="task b only",
                task_id="task-b",
            ),
        ],
    )

    result = memory.derive(one_pot_compression(), task_id="task-a")

    section = result.packet.section("compressed_context")
    assert section.event_ids == ("evt_a",)
    assert "task b only" not in section.content


def test_retrieval_context_filters_by_query_and_trust():
    memory = Memory("full_history")
    memory.update(
        [
            MemoryEvent(
                id="evt_source",
                kind=EventKind.TOOL_RESULT,
                content="The source discusses calibration.",
                trust=TrustLevel.SOURCE_OBSERVATION,
            ),
            MemoryEvent(
                id="evt_note",
                kind=EventKind.REASONING_NOTE,
                content="Unrelated planning note.",
            ),
        ],
    )

    result = memory.derive(
        retrieval_context(
            selector=MemorySelector(trust=(TrustLevel.SOURCE_OBSERVATION.value,)),
        ),
        objective="calibration",
    )

    section = result.packet.section("retrieved_context")
    assert section is not None
    assert section.event_ids == ("evt_source",)
    assert "evt_note" not in section.content


def test_keyword_retrieval_returns_relevant_items_and_events():
    state = MemoryState(
        model_name="custom",
        events=(
            MemoryEvent(
                id="evt_shared",
                kind=EventKind.TOOL_RESULT,
                content="Raw calibration benchmark transcript.",
                trust=TrustLevel.SOURCE_OBSERVATION,
            ),
            MemoryEvent(
                id="evt_independent",
                kind=EventKind.REASONING_NOTE,
                content="Calibration benchmark follow-up question.",
            ),
        ),
        items=(
            MemoryItem(
                id="mem_curated",
                kind=MemoryItemKind.EVIDENCE,
                content="Curated calibration benchmark evidence.",
                provenance=Provenance(event_ids=("evt_shared",)),
                trust=TrustLevel.EXTRACTED_EVIDENCE,
                evidence_status=EvidenceStatus.SUPPORTED,
            ),
        ),
    )
    recipe = retrieval_context()

    result = RecipeEngine().derive(
        state,
        MemoryQuery(objective="calibration benchmark"),
        recipe,
    )

    section = result.packet.section("retrieved_context")
    assert section.item_ids == ("mem_curated",)
    assert section.event_ids == ("evt_shared", "evt_independent")
    assert "mem_curated" in section.content
    assert "evt_independent" in section.content
    assert "evt_shared] tool_result" not in section.content


def test_injected_embedding_retriever_can_drive_retrieval_layer():
    memory = Memory("full_history")
    memory.update(
        MemoryEvent(
            id="evt_vector",
            kind=EventKind.TOOL_RESULT,
            content="Vector backend selected this result.",
        ),
    )

    def fake_retriever(request):
        event = request.state.events[0]
        return CapabilityResponse(
            content=f"embedding result: {event.content}",
            event_ids=(event.id,),
            metadata={"method": "fake_embedding"},
        )

    result = memory.derive(
        retrieval_context(),
        capabilities=RecipeCapabilities(
            retriever=InjectedEmbeddingRetriever(fake_retriever),
        ),
    )

    section = result.packet.section("retrieved_context")
    assert section.event_ids == ("evt_vector",)
    assert "embedding result" in section.content
    assert result.traces[0].capability == "injected_embedding_retriever"


def test_layered_prompt_renderer_reports_section_metadata():
    memory = Memory("full_history")
    memory.update("A compact note for layered rendering.")
    result = memory.derive(one_pot_compression())

    rendered = LayeredPromptRenderer().render(result.packet)

    assert "## Compressed Context" in rendered.text
    assert "summary" in rendered.text


def test_layered_checkpoint_produces_expected_sections():
    memory = Memory("full_history")
    memory.update(
        [
            MemoryEvent(
                id="evt_goal",
                kind=EventKind.REASONING_NOTE,
                content="Need to verify calibration.",
            ),
            MemoryEvent(
                id="evt_evidence",
                kind=EventKind.TOOL_RESULT,
                content="Source reports calibration measurements.",
                trust=TrustLevel.SOURCE_OBSERVATION,
            ),
            MemoryEvent(
                id="evt_recent",
                kind=EventKind.REASONING_NOTE,
                content="Recent note.",
            ),
            MemoryEvent(
                id="evt_missing",
                kind=EventKind.EVIDENCE_REVIEW,
                payload={"missing_evidence": ["independent replication"]},
            ),
        ],
    )

    result = memory.derive(
        layered_checkpoint(recent_event_limit=2),
        objective="calibration evidence",
    )

    assert result.packet.section("high_level") is not None
    evidence = result.packet.section("evidence")
    assert evidence.kind == SectionKind.EVIDENCE
    assert evidence.trust == TrustLevel.SOURCE_OBSERVATION
    assert evidence.event_ids == ("evt_evidence",)
    rendered = memory.render_packet(result.packet, renderer="layered_prompt")
    assert "## Evidence [evidence; source_observation]" in rendered.text
    assert result.packet.section("recent").event_ids == (
        "evt_recent",
        "evt_missing",
    )
    assert result.packet.section("unresolved").event_ids == ("evt_missing",)


def test_packet_budget_reports_only_included_selected_ids():
    memory = Memory("full_history")
    memory.update(
        [
            MemoryEvent(
                id="evt_a",
                kind=EventKind.REASONING_NOTE,
                content="alpha " * 20,
            ),
            MemoryEvent(
                id="evt_b",
                kind=EventKind.REASONING_NOTE,
                content="beta " * 20,
            ),
        ],
    )
    recipe = MemoryRecipe(
        name="budget_scope",
        layers=(
            MemoryLayerSpec(
                id="first",
                title="First",
                operation=MemoryRecipeOperation.KEEP_RAW,
                selector=MemorySelector(event_ids=("evt_a",)),
            ),
            MemoryLayerSpec(
                id="second",
                title="Second",
                operation=MemoryRecipeOperation.KEEP_RAW,
                selector=MemorySelector(event_ids=("evt_b",)),
            ),
        ),
    )

    result = memory.derive(recipe, budget_tokens=15)

    assert result.packet.section("first") is not None
    assert result.packet.section("second") is None
    assert result.selected_event_ids == ("evt_a",)
    assert result.metadata["omitted_event_ids"] == ["evt_b"]
