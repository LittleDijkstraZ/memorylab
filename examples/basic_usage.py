from memory_lab import Memory


def main() -> None:
    memory = Memory("full_history")
    memory.update("Verify whether the claim is supported.", kind="run_started")
    memory.update("I should start with primary sources.", kind="note")
    print(memory.read(objective="verify claim").text)


if __name__ == "__main__":
    main()
