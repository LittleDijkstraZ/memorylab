# Memory Lab

Memory Lab is a small Python library for agent memory. It gives you two
everyday operations:

```python
memory.update(...)
memory.read(...)
```

The default API is intentionally plain: write text or dictionaries in, read
context out. You can swap the memory strategy later.

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

memory = Memory()

memory.update("The user prefers concise research notes.")
memory.update("The current task is a literature review.")

context = memory.read()
print(context.text)
```

You can also pass dictionaries when your app has extra metadata:

```python
memory.update({
    "content": "The user is comparing memory strategies.",
    "metadata": {"project": "memorylab"},
})
```

That is the main idea: `update` writes memory, `read` returns usable context.

## Try Another Strategy

Different memory strategies can organize the same updates differently:

```python
from memory_lab import Memory

memory = Memory("rolling_summary")

memory.update("The user prefers concise research notes.")
memory.update("The current task is a literature review.")
memory.update("The next step is to compare memory strategies.")

print(memory.read().text)
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
memory.read(renderer="compact_prompt")
```

## Examples

Runnable examples are in `examples/`:

```bash
python examples/basic_usage.py
python examples/strategy_usage.py
```

## Advanced API

The simple `Memory` object is a wrapper around lower-level pieces:

```text
MemoryEvent -> MemoryModel.ingest(...) -> MemoryState
MemoryState + MemoryQuery -> ContextPacket
ContextPacket + ContextRenderer -> text
```

Adapters can pass explicit event kinds, provenance, and other structured
fields. Most users can start with `Memory.update()` and `Memory.read()`.
