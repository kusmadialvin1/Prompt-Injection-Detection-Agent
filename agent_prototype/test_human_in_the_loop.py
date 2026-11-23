from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent import SecureAgent
from detector import PIGuardDetector
from security_logger import SecurityLogger


ROOT = Path(__file__).resolve().parent
DOCUMENTS = ROOT / "documents"


class DeterministicDocumentPlanner:
    name = "deterministic-human-confirmation-planner"

    def __init__(self, filename: str) -> None:
        self.filename = filename
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
                    "tool_name": "read_document",
                    "arguments": {"filename": self.filename},
                }
            )
        if self.calls == 2:
            return json.dumps(
                {
                    "action": "answer",
                    "answer": "The user-approved document was processed.",
                }
            )
        raise AssertionError("The planner was called more than twice.")


def run_case(choice: str) -> dict:
    filename = "malicious_document.txt"
    timestamp = datetime.now()
    case_name = "block" if choice == "B" else "continue"
    output_dir = (
        ROOT
        / "logs"
        / f"human_confirmation_test_{case_name}_{timestamp:%Y%m%d_%H%M%S_%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    security_log = output_dir / "security_events.jsonl"
    planner = DeterministicDocumentPlanner(filename)

    agent = SecureAgent(
        PIGuardDetector(),
        SecurityLogger(security_log),
        planner,
        confirmation_input=lambda prompt: choice,
    )
    result = agent.run_autonomous("Read this document and summarize it.")

    events = [
        json.loads(line)
        for line in security_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    detection = next(event for event in events if event["stage"] == "tool_output")
    confirmation = next(
        event for event in events if event["stage"] == "user_security_confirmation"
    )
    document_text = (DOCUMENTS / filename).read_text(encoding="utf-8")
    document_entered_context = (
        len(planner.prompts) >= 2
        and any(document_text in prompt for prompt in planner.prompts[1:])
    )

    if choice == "B":
        passed = (
            detection["predicted_label"] == 1
            and detection["action"] == "blocked"
            and confirmation["decision"] == "block"
            and confirmation["content_sent_to_llm"] is False
            and not document_entered_context
            and planner.calls == 1
            and result.action == "blocked"
        )
    else:
        passed = (
            detection["predicted_label"] == 1
            and confirmation["decision"] == "continue"
            and confirmation["content_sent_to_llm"] is True
            and document_entered_context
            and planner.calls == 2
            and result.action == "allowed"
        )

    report = {
        "status": "passed" if passed else "failed",
        "choice": choice,
        "document": filename,
        "piguard_predicted_label": detection["predicted_label"],
        "detector_action": detection["action"],
        "warning_generated": True,
        "user_decision": confirmation["decision"],
        "malicious_document_entered_next_llm_context": document_entered_context,
        "planner_calls": planner.calls,
        "agent_action": result.action,
        "agent_response": result.response,
        "security_log": str(security_log),
        "note": (
            "This is a deterministic human-confirmation boundary test, "
            "not evidence of Ollama behavior."
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"RESULT_DIR={output_dir}")
    return report


def main() -> None:
    cases = [run_case("B"), run_case("C")]
    if any(case["status"] != "passed" for case in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
