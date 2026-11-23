from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent import SecureAgent
from detector import PIGuardDetector
from security_logger import SecurityLogger


ROOT = Path(__file__).resolve().parent
QUERY = "The Temple of Heaven in Beijing history"
EXPECTED_TOOL_OUTPUT = (
    f"Search result for '{QUERY}': the topic is documented."
)


class DeterministicPlanner:
    """Controlled planner for testing accepted tool output and context reuse."""

    name = "deterministic-benign-test-planner"

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        return self.generate_structured(prompt)

    def generate_structured(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            return json.dumps(
                {
                    "action": "tool",
                    "tool_name": "mock_search",
                    "arguments": {
                        "query": QUERY,
                        "malicious": False,
                    },
                }
            )
        if self.calls == 2:
            return json.dumps(
                {
                    "action": "answer",
                    "answer": "The search result was accepted as benign.",
                }
            )
        raise AssertionError("The planner exceeded the expected two calls.")


def main() -> None:
    output_dir = ROOT / "logs" / f"benign_tool_test_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output_dir.mkdir(parents=True, exist_ok=False)
    security_log = output_dir / "security_events.jsonl"

    planner = DeterministicPlanner()
    agent = SecureAgent(
        PIGuardDetector(),
        SecurityLogger(security_log),
        planner,
    )
    result = agent.run_autonomous("Search for information about the Temple of Heaven.")

    events = [
        json.loads(line)
        for line in security_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selection = next(event for event in events if event["stage"] == "tool_selection")
    execution = next(event for event in events if event["stage"] == "tool_execution")
    detection = next(event for event in events if event["stage"] == "tool_output")
    final_response = [
        event for event in events
        if event["stage"] == "llm_response" and event["action"] == "returned"
    ]
    context_received = (
        len(planner.prompts) >= 2
        and EXPECTED_TOOL_OUTPUT in planner.prompts[1]
        and "UNTRUSTED TOOL OUTPUT" in planner.prompts[1]
    )

    report = {
        "status": "passed" if (
            result.action == "allowed"
            and execution["text_preview"] == EXPECTED_TOOL_OUTPUT
            and detection["predicted_label"] == 0
            and detection["action"] == "allowed"
            and context_received
            and planner.calls == 2
            and len(final_response) == 1
        ) else "failed",
        "planner": planner.name,
        "tool_selected": selection,
        "actual_tool_output": execution["text_preview"],
        "expected_tool_output": EXPECTED_TOOL_OUTPUT,
        "piguard_predicted_label": detection["predicted_label"],
        "security_action": detection["action"],
        "benign_output_entered_next_llm_context": context_received,
        "planner_called_again_after_acceptance": planner.calls >= 2,
        "another_ollama_request_occurred": False,
        "planner_calls": planner.calls,
        "agent_action": result.action,
        "agent_response": result.response,
        "security_log": str(security_log),
        "note": (
            "This is a deterministic security-path test, not evidence of "
            "Ollama behavior."
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"RESULT_DIR={output_dir}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
