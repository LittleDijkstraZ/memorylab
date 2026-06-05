# Memory Lab

Memory Lab is a small Python library for agent memory. It gives you two
everyday operations:

```python
memory.update(...)
memory.read(...)
```

Under the hood, you can swap different memory strategies such as full history,
rolling summaries, task hierarchies, and evidence ledgers.

## Install

```bash
git clone git@github.com:LittleDijkstraZ/memorylab.git
cd memorylab
python -m pip install -e .
```

For local development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Memory Lab has no runtime dependencies outside the Python standard library.
Python 3.11 or newer is required.

## Quick Start

```python
from memory_lab import Memory

memory = Memory("full_history")

memory.update("Verify whether the claim is supported.", kind="run_started")
memory.update("I should start with primary sources.", kind="note")

context = memory.read(objective="verify claim")
print(context.text)
```

That is the main idea: write with `update`, read with `read`.

## Evidence Memory

For research workflows, use the evidence ledger:

```python
from memory_lab import Memory

memory = Memory("evidence_ledger")

memory.update(
    {
        "kind": "evidence",
        "slot": "benchmark",
        "content": "The source reports a 12% improvement.",
        "source": "https://example.test/paper",
        "status": "supported",
    }
)

memory.update(
    {
        "kind": "missing",
        "slot": "safety",
        "content": "Need an independent safety evaluation.",
    }
)

print(memory.read(objective="audit evidence").text)
```

## Available Memory Models

```python
Memory("full_history")
Memory("evidence_ledger")
Memory("rolling_summary")
Memory("hierarchical_summary")
Memory("llm_managed")
```

The default renderer is chosen for the model. You can override it:

```python
memory.read(objective="audit evidence", renderer="compact_prompt")
```

## Examples

Runnable examples are in `examples/`:

```bash
python examples/basic_usage.py
python examples/evidence_ledger.py
```

## Advanced API

The simple `Memory` object is a wrapper around the lower-level pieces:

```text
MemoryEvent -> MemoryModel.ingest(...) -> MemoryState
MemoryState + MemoryQuery -> ContextPacket
ContextPacket + ContextRenderer -> text
```

Use the lower-level schema/model APIs when you are building adapters, replay
harnesses, or comparing memory strategies directly.
