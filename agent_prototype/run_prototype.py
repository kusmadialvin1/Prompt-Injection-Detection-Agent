from __future__ import annotations

from pathlib import Path

from agent import MockLLM, OllamaLLM, SecureAgent
from detector import PIGuardDetector
from security_logger import SecurityLogger


def main() -> None:
    detector = PIGuardDetector()
    if OllamaLLM.available("qwen3:4b-nothink"):
        llm = OllamaLLM()
    else:
        llm = MockLLM()
        print("Ollama model unavailable; using mock LLM.")
    logger = SecurityLogger(Path(__file__).resolve().parent / "logs" / "security_events.jsonl")
    agent = SecureAgent(detector, logger, llm)

    examples = [
        ("normal user input", agent.run("Please calculate 2 + 2.", "calculator", "2 + 2")),
        (
            "malicious tool output",
            agent.run(
                "Search for prompt-injection information.",
                "mock_search",
                "prompt injection",
                malicious_tool=True,
            ),
        ),
    ]
    for name, result in examples:
        print(f"\n[{name}] action={result.action} llm={result.llm_name}")
        print(result.response)


if __name__ == "__main__":
    main()
