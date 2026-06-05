from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memory_lab.schema import ContextPacket, ContextSection, estimate_tokens


@dataclass(frozen=True)
class RenderedContext:
    text: str
    packet_id: str
    token_estimate: int
    omitted_section_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


class ContextRenderer(Protocol):
    name: str

    def render(
        self,
        packet: ContextPacket,
        *,
        budget_tokens: int | None = None,
    ) -> RenderedContext:
        ...


class BaseTextRenderer:
    name = "base_text"

    def render(
        self,
        packet: ContextPacket,
        *,
        budget_tokens: int | None = None,
    ) -> RenderedContext:
        pieces: list[str] = []
        omitted: list[str] = []
        warnings: list[str] = list(packet.warnings)
        token_total = 0
        for section in packet.sections:
            piece = self.render_section(section)
            piece_tokens = estimate_tokens(piece)
            if (
                budget_tokens is not None
                and pieces
                and token_total + piece_tokens > budget_tokens
            ):
                omitted.append(section.id)
                warnings.append(
                    f"section {section.id} omitted by renderer budget {budget_tokens}",
                )
                continue
            if budget_tokens is not None and not pieces and piece_tokens > budget_tokens:
                keep_chars = max(0, budget_tokens * 4)
                piece = piece[:keep_chars].rstrip() + "\n[truncated by renderer budget]"
                piece_tokens = estimate_tokens(piece)
                warnings.append(
                    f"section {section.id} truncated by renderer budget {budget_tokens}",
                )
            pieces.append(piece)
            token_total += piece_tokens
        text = self.join_sections(pieces)
        return RenderedContext(
            text=text,
            packet_id=packet.id,
            token_estimate=estimate_tokens(text),
            omitted_section_ids=tuple(omitted),
            warnings=tuple(warnings),
            metadata={"renderer": self.name, "budget_tokens": budget_tokens},
        )

    def join_sections(self, pieces: list[str]) -> str:
        return "\n\n".join(piece for piece in pieces if piece)

    def render_section(self, section: ContextSection) -> str:
        raise NotImplementedError
