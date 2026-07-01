from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from memory_lab.capabilities.base import CapabilityResponse, CompressionRequest
from memory_lab.schema import estimate_tokens, unique_tuple


LLMCompressorFn = Callable[[CompressionRequest], str | CapabilityResponse]


class InjectedLLMCompressor:
    """Adapter for caller-provided LLM compression policies.

    The portable core owns no provider client. Tests should pass a fake callable;
    applications can inject a real LLM-backed callable later.
    """

    name = "injected_llm_compressor"

    def __init__(self, fn: LLMCompressorFn) -> None:
        self.fn = fn

    def compress(self, request: CompressionRequest) -> CapabilityResponse:
        result = self.fn(request)
        if isinstance(result, CapabilityResponse):
            metadata = dict(result.metadata)
            metadata.setdefault("method", self.name)
            metadata.setdefault("token_estimate", estimate_tokens(result.content))
            return replace(
                result,
                event_ids=result.event_ids
                or unique_tuple(event.id for event in request.events),
                item_ids=result.item_ids
                or unique_tuple(item.id for item in request.items),
                metadata=metadata,
            )
        content = str(result)
        return CapabilityResponse(
            content=content,
            event_ids=unique_tuple(event.id for event in request.events),
            item_ids=unique_tuple(item.id for item in request.items),
            metadata={
                "method": self.name,
                "token_estimate": estimate_tokens(content),
            },
        )


__all__ = ["InjectedLLMCompressor", "LLMCompressorFn"]
