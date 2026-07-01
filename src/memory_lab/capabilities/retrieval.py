from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from memory_lab.capabilities.base import CapabilityResponse, RetrievalRequest
from memory_lab.schema import (
    MemoryItemKind,
    TrustLevel,
    compact_json,
    estimate_tokens,
    unique_tuple,
)


class KeywordRetriever:
    """Small deterministic retriever based on query-term overlap."""

    name = "keyword_retriever"

    def retrieve(self, request: RetrievalRequest) -> CapabilityResponse:
        query_terms = _terms(request.query.objective)
        candidates = self._candidates(request, query_terms)

        lines: list[str] = []
        event_ids: list[str] = []
        item_ids: list[str] = []
        covered_event_ids: set[str] = set()
        omitted_count = 0
        token_total = 0

        for candidate in candidates:
            if _is_duplicate(candidate, covered_event_ids):
                continue
            if _would_exceed(
                request.budget_tokens,
                token_total,
                candidate.token_count,
                lines,
            ):
                omitted_count += 1
                continue
            token_total += candidate.token_count
            lines.append(candidate.line)
            event_ids.extend(candidate.event_ids)
            covered_event_ids.update(candidate.event_ids)
            if candidate.item_id is not None:
                item_ids.append(candidate.item_id)

        content = "\n".join(lines) or "(no retrieved memory content)"
        return CapabilityResponse(
            content=content,
            event_ids=unique_tuple(event_ids),
            item_ids=unique_tuple(item_ids),
            omitted_count=omitted_count,
            omitted_reason="retrieval-budget" if omitted_count else None,
            metadata={
                "method": self.name,
                "query_terms": sorted(query_terms),
                "candidate_count": len(candidates),
                "token_estimate": estimate_tokens(content),
            },
        )

    def _candidates(
        self,
        request: RetrievalRequest,
        query_terms: set[str],
    ) -> list["_Candidate"]:
        candidates: list[_Candidate] = []
        for order, event in enumerate(request.state.events):
            text = _event_text(event)
            score = _score(text, query_terms)
            if query_terms and score <= 0:
                continue
            line = _event_line(event)
            candidates.append(
                _Candidate(
                    score=score,
                    priority=0,
                    order=order,
                    line=line,
                    token_count=estimate_tokens(line),
                    event_ids=(event.id,),
                ),
            )

        item_offset = len(candidates)
        for order, item in enumerate(request.state.items, start=item_offset):
            score = _score(item.content, query_terms)
            if query_terms and score <= 0:
                continue
            line = _item_line(item)
            candidates.append(
                _Candidate(
                    score=score,
                    priority=_item_priority(item),
                    order=order,
                    line=line,
                    token_count=estimate_tokens(line),
                    event_ids=item.provenance.event_ids,
                    item_id=item.id,
                ),
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                -candidate.priority,
                candidate.order,
            ),
        )
        return candidates


@dataclass(frozen=True)
class _Candidate:
    score: int
    priority: int
    order: int
    line: str
    token_count: int
    event_ids: tuple[str, ...]
    item_id: str | None = None


EmbeddingRetrieverFn = Callable[[RetrievalRequest], CapabilityResponse]


class InjectedEmbeddingRetriever:
    """Adapter for caller-provided embedding/vector retrieval policies."""

    name = "injected_embedding_retriever"

    def __init__(self, fn: EmbeddingRetrieverFn) -> None:
        self.fn = fn

    def retrieve(self, request: RetrievalRequest) -> CapabilityResponse:
        return self.fn(request)


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[A-Za-z0-9_]+", text.lower())
        if len(term) > 2
    }


def _score(text: str, query_terms: set[str]) -> int:
    if not query_terms:
        return 1
    terms = _terms(text)
    return len(terms & query_terms)


def _event_text(event) -> str:
    text = event.text()
    if text:
        return text
    if event.payload:
        return compact_json(event.payload)
    return ""


def _event_line(event) -> str:
    prefix = event.kind.value
    if event.task_id:
        prefix = f"{prefix} task={event.task_id}"
    return f"- [{event.id}] {prefix}: {_event_text(event)}"


def _item_line(item) -> str:
    prefix = item.kind.value
    if item.task_id:
        prefix = f"{prefix} task={item.task_id}"
    return f"- [{item.id}] {prefix}: {item.content}"


def _is_duplicate(candidate: _Candidate, covered_event_ids: set[str]) -> bool:
    if not candidate.event_ids:
        return False
    return set(candidate.event_ids).issubset(covered_event_ids)


def _item_priority(item) -> int:
    if getattr(item, "is_primary_evidence", False):
        return 3
    if item.kind == MemoryItemKind.EVIDENCE:
        return 2
    if item.trust in {TrustLevel.SOURCE_OBSERVATION, TrustLevel.EXTRACTED_EVIDENCE}:
        return 1
    return 0


def _would_exceed(
    budget_tokens: int | None,
    token_total: int,
    line_tokens: int,
    lines: list[str],
) -> bool:
    return (
        budget_tokens is not None
        and bool(lines)
        and token_total + line_tokens > budget_tokens
    )


__all__ = [
    "EmbeddingRetrieverFn",
    "InjectedEmbeddingRetriever",
    "KeywordRetriever",
]
