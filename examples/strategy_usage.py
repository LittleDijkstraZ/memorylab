from memory_lab import Memory


def main() -> None:
    memory = Memory("rolling_summary")
    memory.update("The user prefers concise research notes.")
    memory.update("The current task is a literature review.")
    memory.update("The next step is to compare memory strategies.")
    print(memory.read().text)


if __name__ == "__main__":
    main()
