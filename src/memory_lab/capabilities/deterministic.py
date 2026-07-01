from __future__ import annotations

from memory_lab.capabilities.base import CapabilityResponse, CompressionRequest
from memory_lab.schema import compact_json, estimate_tokens, unique_tuple


class DeterministicCompressor:
    """Bounded deterministic compaction for tests and fallback paths."""

    name = "deterministic_compressor"

    def compress(self, request: CompressionRequest) -> CapabilityResponse:
        lines = _request_lines(request)
        event_ids = unique_tuple(event.id for event in request.events)
        item_ids = unique_tuple(item.id for item in request.items)
        content = "\n".join(lines)
        omitted_count = 0
        omitted_reason = None
        if request.budget_tokens is not None:
            max_chars = max(0, request.budget_tokens * 4)
            if len(content) > max_chars:
                content = content[:max_chars].rstrip()
                omitted_count = max(0, len(lines) - _line_count(content))
                omitted_reason = "deterministic-compressor-budget"
                content += "\n[truncated by deterministic compressor budget]"
        if not content:
            content = "(no selected memory content)"
        return CapabilityResponse(
            content=content,
            event_ids=event_ids,
            item_ids=item_ids,
            omitted_count=omitted_count,
            omitted_reason=omitted_reason,
            metadata={
                "method": self.name,
                "instruction": request.instruction,
                "token_estimate": estimate_tokens(content),
            },
        )


def _request_lines(request: CompressionRequest) -> list[str]:
    if request.events:
        return [_event_line(event) for event in request.events]
    return [_item_line(item) for item in request.items]


def _event_line(event) -> str:
    text = event.text()
    if not text:
        text = compact_json(event.payload)
    prefix = event.kind.value
    if event.task_id:
        prefix = f"{prefix} task={event.task_id}"
    if event.worker_id:
        prefix = f"{prefix} worker={event.worker_id}"
    return f"- [{event.id}] {prefix}: {text}"


def _item_line(item) -> str:
    prefix = item.kind.value
    if item.task_id:
        prefix = f"{prefix} task={item.task_id}"
    if item.worker_id:
        prefix = f"{prefix} worker={item.worker_id}"
    return f"- [{item.id}] {prefix}: {item.content}"


def _line_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


__all__ = ["DeterministicCompressor"]
