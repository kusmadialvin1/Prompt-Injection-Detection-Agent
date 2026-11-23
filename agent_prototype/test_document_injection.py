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
    """Controlled planner for testing normal and malicious document handling."""

    name = "deterministic-document-test-planner"

    def __init__(self, filename: str, malicious: bool) -> None:
        self.filename = filename
        self.malicious = malicious
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
        if self.calls == 2 and not self.malicious:
            return json.dumps(
                {
                    "action": "answer",
                    "answer": "A summary of the accepted document.",
                }
            )
        raise AssertionError(
            "The planner was called after the document security boundary should have blocked."
        )


def run_case(filename: str, malicious: bool) -> dict:
    timestamp = datetime.now()
    case_name = "malicious_document" if malicious else "normal_document"
    output_dir = (
        ROOT
        / "logs"
        / f"document_injection_test_{case_name}_{timestamp:%Y%m%d_%H%M%S_%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    security_log = output_dir / "security_events.jsonl"

    planner = DeterministicDocumentPlanner(filename, malicious)
    agent = SecureAgent(
        PIGuardDetector(),
        SecurityLogger(security_log),
        planner,
        confirmation_input=lambda prompt: "B" if malicious else "C",
    )
    result = agent.run_autonomous("Read this document and summarize it.")

    events = [
        json.loads(line)
        for line in security_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selection = next(event for event in events if event["stage"] == "tool_selection")
    execution = next(event for event in events if event["stage"] == "tool_execution")
    detection = next(event for event in events if event["stage"] == "tool_output")
    document_text = (DOCUMENTS / filename).read_text(encoding="utf-8")

    if malicious:
        document_entered_context = (
            len(planner.prompts) >= 2
            and any(document_text in prompt for prompt in planner.prompts[1:])
        )
        another_planner_call = planner.calls >= 2
        passed = (
            result.action == "blocked"
            and detection["predicted_label"] == 1
            and detection["action"] == "blocked"
            and not document_entered_context
            and not another_planner_call
            and planner.calls == 1
        )
    else:
        document_entered_context = (
            len(planner.prompts) >= 2
            and document_text in planner.prompts[1]
            and "UNTRUSTED TOOL OUTPUT" in planner.prompts[1]
        )
        another_planner_call = planner.calls >= 2
        passed = (
            result.action == "allowed"
            and detection["predicted_label"] == 0
            and detection["action"] == "allowed"
            and document_entered_context
            and another_planner_call
            and planner.calls == 2
        )

    report = {
        "status": "passed" if passed else "failed",
        "document": filename,
        "planner": planner.name,
        "tool_selected": selection,
        "actual_tool_output": execution["text_preview"],
        "piguard_predicted_label": detection["predicted_label"],
        "security_action": detection["action"],
        "document_entered_next_llm_context": document_entered_context,
        "planner_called_again": another_planner_call,
        "planner_calls": planner.calls,
        "agent_action": result.action,
        "agent_response": result.response,
        "security_log": str(security_log),
        "note": (
            "This is a deterministic security-boundary test, not evidence of "
            "Ollama behavior."
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
    cases = [
        run_case("normal_document.txt", malicious=False),
        run_case("malicious_document.txt", malicious=True),
    ]
    if any(case["status"] != "passed" for case in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
