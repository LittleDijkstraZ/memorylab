from memory_lab.models.base import BaseMemoryModel, MemoryModel
from memory_lab.models.evidence_ledger import EvidenceLedgerMemory
from memory_lab.models.full_history import FullHistoryMemory
from memory_lab.models.hierarchical_summary import HierarchicalSummaryMemory
from memory_lab.models.llm_managed import LLMManagedMemory, MemoryManager
from memory_lab.models.rolling_summary import RollingSummaryMemory
from memory_lab.registry import register_model


register_model(FullHistoryMemory, default_renderer="research_brief", replace=True)
register_model(EvidenceLedgerMemory, default_renderer="evidence_table", replace=True)
register_model(RollingSummaryMemory, default_renderer="compact_prompt", replace=True)
register_model(HierarchicalSummaryMemory, default_renderer="compact_prompt", replace=True)
register_model(LLMManagedMemory, default_renderer="research_brief", replace=True)

__all__ = [
    "BaseMemoryModel",
    "EvidenceLedgerMemory",
    "FullHistoryMemory",
    "HierarchicalSummaryMemory",
    "LLMManagedMemory",
    "MemoryManager",
    "MemoryModel",
    "RollingSummaryMemory",
]
