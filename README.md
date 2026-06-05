# Memory Lab

Memory Lab is a small Python library for experimenting with agent memory
strategies. It treats memory as a swappable model over events:

```text
MemoryEvent -> MemoryModel.ingest(...) -> MemoryState
MemoryState + MemoryQuery -> ContextPacket
ContextPacket + ContextRenderer -> model-ready text
```

The goal is to make memory strategies comparable without forcing a vector
database, graph database, hosted memory service, or one specific agent runtime.

## Install

```bash
git clone git@github.com:LittleDijkstraZ/memorylab.git
cd memorylab
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Memory Lab currently has no runtime dependencies outside the Python standard
library. Python 3.11 or newer is required.

## Quick Example

```python
from memory_lab import (
    EventKind,
    FullHistoryMemory,
    MemoryEvent,
    MemoryQuery,
    ResearchBriefRenderer,
)

model = FullHistoryMemory()
state = model.initial_state()
state = model.ingest(
    state,
    (
        MemoryEvent(
            id="evt_run",
            kind=EventKind.RUN_STARTED,
            content="Verify whether the claim is supported.",
        ),
        MemoryEvent(
            id="evt_reasoning",
            kind=EventKind.REASONING_NOTE,
            content="I should start with primary sources.",
        ),
    ),
)

packet = model.select_context(state, MemoryQuery(objective="verify claim"))
rendered = ResearchBriefRenderer().render(packet)
print(rendered.text)
```

## What Is Included

- `MemoryEvent`, `MemoryItem`, `MemoryState`, `MemoryQuery`, and
  `ContextPacket` schemas.
- Trust and provenance labels for source observations, model-derived summaries,
  worker summaries, final synthesis, and control state.
- Memory models:
  - `FullHistoryMemory`
  - `EvidenceLedgerMemory`
  - `RollingSummaryMemory`
  - `HierarchicalSummaryMemory`
  - `LLMManagedMemory` with an injected operation manager
- Renderers:
  - `ResearchBriefRenderer`
  - `CompactPromptRenderer`
  - `EvidenceTableRenderer`
- Packet comparison helpers for coverage and token-budget checks.

## Evidence Ledger Example

```python
from memory_lab import (
    EventKind,
    EvidenceLedgerMemory,
    EvidenceTableRenderer,
    MemoryEvent,
    MemoryQuery,
)

model = EvidenceLedgerMemory()
state = model.initial_state()
state = model.ingest(
    state,
    (
        MemoryEvent(
            id="evt_source",
            kind=EventKind.TOOL_RESULT,
            payload={
                "evidence": [
                    {
                        "claim_slot": "benchmark",
                        "content": "The source reports a 12% improvement.",
                        "status": "supported",
                        "url": "https://example.test/paper",
                    },
                ],
            },
        ),
    ),
)

packet = model.select_context(state, MemoryQuery(objective="audit evidence"))
print(EvidenceTableRenderer().render(packet).text)
```

More runnable examples are in `examples/`.

## Design Notes

Memory Lab keeps durable memory separate from control context. Tool-use
protocols, budget hints, and current-turn instructions can be rendered into a
context packet, but they should not become source evidence.

The evidence ledger is rule-first. Final reports and model summaries are useful
context hints, but they are not treated as primary evidence unless they cite
source-bearing events.
