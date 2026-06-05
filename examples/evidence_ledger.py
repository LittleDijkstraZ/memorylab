from memory_lab import Memory


def main() -> None:
    memory = Memory("evidence_ledger")
    memory.update(
        {
            "kind": "evidence",
            "slot": "benchmark",
            "content": "The source reports a 12% improvement.",
            "source": "https://example.test/paper",
            "status": "supported",
            "confidence": 0.9,
        },
    )
    memory.update(
        {
            "kind": "missing",
            "slot": "safety",
            "content": "Need an independent safety evaluation.",
        },
    )
    print(memory.read(objective="audit evidence").text)


if __name__ == "__main__":
    main()
