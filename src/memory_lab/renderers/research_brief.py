from __future__ import annotations

from memory_lab.renderers.base import BaseTextRenderer
from memory_lab.schema import ContextSection


class ResearchBriefRenderer(BaseTextRenderer):
    name = "research_brief"

    def render_section(self, section: ContextSection) -> str:
        attrs = (
            f'id="{section.id}" '
            f'kind="{section.kind.value}" '
            f'trust="{section.trust.value}" '
            f'tokens="{section.token_estimate}"'
        )
        warnings = ""
        if section.warnings:
            warnings = "\n<warnings>\n" + "\n".join(section.warnings) + "\n</warnings>"
        return (
            f"<section {attrs}>\n"
            f"<title>{section.title}</title>\n"
            f"{section.content}"
            f"{warnings}\n"
            f"</section>"
        )
