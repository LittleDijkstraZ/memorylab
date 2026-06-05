from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from memory_lab.schema import ContextPacket, unique_tuple


@dataclass(frozen=True)
class PacketComparison:
    expected_event_ids: tuple[str, ...]
    covered_event_ids: tuple[str, ...]
    missing_event_ids: tuple[str, ...]
    unexpected_event_ids: tuple[str, ...]
    expected_item_ids: tuple[str, ...]
    covered_item_ids: tuple[str, ...]
    missing_item_ids: tuple[str, ...]
    expected_claim_slots: tuple[str, ...]
    covered_claim_slots: tuple[str, ...]
    missing_claim_slots: tuple[str, ...]
    token_estimate: int
    over_budget: bool
    warnings: tuple[str, ...]

    @property
    def event_coverage(self) -> float:
        if not self.expected_event_ids:
            return 1.0
        return len(self.covered_event_ids) / len(self.expected_event_ids)

    @property
    def item_coverage(self) -> float:
        if not self.expected_item_ids:
            return 1.0
        return len(self.covered_item_ids) / len(self.expected_item_ids)

    @property
    def claim_slot_coverage(self) -> float:
        if not self.expected_claim_slots:
            return 1.0
        return len(self.covered_claim_slots) / len(self.expected_claim_slots)


def compare_packet(
    packet: ContextPacket,
    *,
    expected_event_ids: Iterable[str] = (),
    expected_item_ids: Iterable[str] = (),
    expected_claim_slots: Iterable[str] = (),
    budget_tokens: int | None = None,
) -> PacketComparison:
    expected_events = unique_tuple(str(value) for value in expected_event_ids)
    expected_items = unique_tuple(str(value) for value in expected_item_ids)
    expected_slots = unique_tuple(str(value) for value in expected_claim_slots)

    packet_events = packet.event_ids
    packet_items = packet.item_ids
    packet_slot_values = (
        str(row.get("claim_slot"))
        for section in packet.sections
        for row in section.metadata.get("rows", ())
        if isinstance(row, dict) and row.get("claim_slot")
    )
    packet_slots = unique_tuple(packet_slot_values)
    covered_events = tuple(event_id for event_id in expected_events if event_id in packet_events)
    covered_items = tuple(item_id for item_id in expected_items if item_id in packet_items)
    covered_slots = tuple(slot for slot in expected_slots if slot in packet_slots)
    missing_event_ids = tuple(
        event_id for event_id in expected_events if event_id not in packet_events
    )
    unexpected_event_ids = tuple(
        event_id for event_id in packet_events if event_id not in expected_events
    )
    missing_item_ids = tuple(
        item_id for item_id in expected_items if item_id not in packet_items
    )
    missing_claim_slots = tuple(slot for slot in expected_slots if slot not in packet_slots)
    over_budget = budget_tokens is not None and packet.token_estimate > budget_tokens
    warnings = list(packet.warnings)
    if over_budget:
        warnings.append(
            f"packet token estimate {packet.token_estimate} exceeds budget {budget_tokens}",
        )
    return PacketComparison(
        expected_event_ids=expected_events,
        covered_event_ids=covered_events,
        missing_event_ids=missing_event_ids,
        unexpected_event_ids=unexpected_event_ids,
        expected_item_ids=expected_items,
        covered_item_ids=covered_items,
        missing_item_ids=missing_item_ids,
        expected_claim_slots=expected_slots,
        covered_claim_slots=covered_slots,
        missing_claim_slots=missing_claim_slots,
        token_estimate=packet.token_estimate,
        over_budget=over_budget,
        warnings=tuple(warnings),
    )
