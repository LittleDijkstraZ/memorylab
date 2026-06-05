from memory_lab.renderers.base import BaseTextRenderer, ContextRenderer, RenderedContext
from memory_lab.renderers.compact_prompt import CompactPromptRenderer
from memory_lab.renderers.evidence_table import EvidenceTableRenderer
from memory_lab.renderers.research_brief import ResearchBriefRenderer
from memory_lab.registry import register_renderer


register_renderer(ResearchBriefRenderer, replace=True)
register_renderer(CompactPromptRenderer, replace=True)
register_renderer(EvidenceTableRenderer, replace=True)

__all__ = [
    "BaseTextRenderer",
    "CompactPromptRenderer",
    "ContextRenderer",
    "EvidenceTableRenderer",
    "RenderedContext",
    "ResearchBriefRenderer",
]
