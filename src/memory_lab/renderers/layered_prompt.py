from __future__ import annotations

from memory_lab.renderers.base import BaseTextRenderer
from memory_lab.schema import ContextSection


class LayeredPromptRenderer(BaseTextRenderer):
    name = "layered_prompt"

    def render_section(self, section: ContextSection) -> str:
        header = f"## {section.title} [{section.kind.value}; {section.trust.value}]"
        details = []
        if section.omitted_count:
            details.append(f"{section.omitted_count} omitted")
            if section.omitted_reason:
                details.append(section.omitted_reason)
        if section.warnings:
            details.append("warnings: " + "; ".join(section.warnings))
        if details:
            header += " (" + "; ".join(details) + ")"
        return f"{header}\n{section.content}"


__all__ = ["LayeredPromptRenderer"]
