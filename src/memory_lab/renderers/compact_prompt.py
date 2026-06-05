from __future__ import annotations

from memory_lab.renderers.base import BaseTextRenderer
from memory_lab.schema import ContextSection


class CompactPromptRenderer(BaseTextRenderer):
    name = "compact_prompt"

    def render_section(self, section: ContextSection) -> str:
        header = f"## {section.title} [{section.kind.value}; {section.trust.value}]"
        if section.omitted_count:
            header += f" ({section.omitted_count} omitted: {section.omitted_reason})"
        return f"{header}\n{section.content}"
