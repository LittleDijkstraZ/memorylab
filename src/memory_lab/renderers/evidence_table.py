from __future__ import annotations

from typing import Any, Mapping

from memory_lab.renderers.base import BaseTextRenderer
from memory_lab.schema import ContextSection


class EvidenceTableRenderer(BaseTextRenderer):
    name = "evidence_table"

    def render_section(self, section: ContextSection) -> str:
        rows = section.metadata.get("rows")
        if not isinstance(rows, list) or not rows:
            return f"## {section.title}\n{section.content}"
        lines = [
            f"## {section.title}",
            "| status | slot | trust | confidence | content | events | sources |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            if isinstance(row, Mapping):
                lines.append(self._row(row))
        return "\n".join(lines)

    def _row(self, row: Mapping[str, Any]) -> str:
        status = self._cell(row.get("status"))
        slot = self._cell(row.get("claim_slot"))
        trust = self._cell(row.get("trust"))
        confidence = self._cell(row.get("confidence"))
        content = self._cell(row.get("content"))
        events = self._cell(", ".join(str(value) for value in row.get("event_ids", ())))
        sources = self._cell(", ".join(str(value) for value in row.get("source_refs", ())))
        return f"| {status} | {slot} | {trust} | {confidence} | {content} | {events} | {sources} |"

    def _cell(self, value: object) -> str:
        if value is None:
            return ""
        text = str(value).replace("|", "\\|").replace("\n", " ")
        return text
