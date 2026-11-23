"""Offline V2 tool-output gate regression using the real detector and a stub LLM.

Run directly. Creates a fresh, separate log directory; never calls Ollama.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from detector import PIGuardDetector
from evaluate_prototype_v2 import V2Agent, V2_ANSWER_INSTRUCTIONS, V2_TOOL_OUTPUT_MARKER, load_cases
from mock_search_v2 import MockSearchV2
from security_logger import SecurityLogger


ROOT = Path(__file__).resolve().parent
ORIGINAL_CASES = ROOT / "test_cases.json"


class RecordingLLM:
    name = "recording-stub-no-ollama"

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "STUB_RESPONSE; not evidence of LLM safety"


def main() -> None:
    original = json.loads(ORIGINAL_CASES.read_text(encoding="utf-8"))
    malicious = [case for case in original if case["id"].startswith("tool-injection-")]
    benign = load_cases()
    if (len(malicious) != 10 or len(benign) != 10
            or {case["id"] for case in malicious} != {f"tool-injection-{n:02d}" for n in range(1, 11)}):
        raise ValueError("Expected the ten original malicious cases and ten V2 normal cases.")

    output_dir = ROOT / "logs" / f"security_regression_v2_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output_dir.mkdir(parents=True, exist_ok=False)
    detector = PIGuardDetector()
    search = MockSearchV2()
    records = []
    violations = []

    for case in benign + malicious:
        attack = case in malicious
        case_id = case["id"]
        request = case["text"]
        query = case["tool_input"] if attack else case["query"]
        payload = case["malicious_payload"] if attack else None
        expected_output = payload if attack else search.run(query).output
        llm = RecordingLLM()
        security_log = output_dir / f"{case_id}_security_events.jsonl"
        agent = V2Agent(detector, SecurityLogger(security_log), llm)
        result = agent.run(request, tool="mock_search", tool_input=query,
                           malicious_tool=attack, malicious_payload=payload)
        events = [json.loads(line) for line in security_log.read_text(encoding="utf-8").splitlines()]
        tool_events = [event for event in events if event["stage"] == "tool_output"]
        tool_event = tool_events[0] if len(tool_events) == 1 else None
        prompt = llm.prompts[0] if len(llm.prompts) == 1 else None
        actual_label = result.tool_detection.label if result.tool_detection is not None else None
        actual_classification = {0: "benign", 1: "injection"}.get(actual_label, "not_evaluated")
        action = tool_event["action"] if tool_event else "not_evaluated"
        record = {
            "case_id": case_id,
            "expected_classification": "injection" if attack else "benign",
            "actual_classification": actual_classification,
            "detector_confidence": result.tool_detection.confidence if result.tool_detection else None,
            "tool_output_action": action,
            "llm_received_prompt": prompt is not None,
            "tool_output_in_prompt": prompt is not None and expected_output in prompt,
            "untrusted_marker_present": prompt is not None and V2_TOOL_OUTPUT_MARKER in prompt,
            "v2_safety_instructions_present": prompt is not None and V2_ANSWER_INSTRUCTIONS in prompt,
            "detector_false_negative": attack and actual_label == 0,
            "security_log": str(security_log.relative_to(output_dir)),
        }
        records.append(record)

        if (tool_event is None or result.tool_output != expected_output
                or tool_event["predicted_label"] != actual_label
                or action != ("blocked" if actual_label == 1 else "allowed")
                or result.action != action):
            violations.append(f"{case_id}: tool execution, detector decision, or log mismatch")
        if action == "blocked" and llm.prompts:
            violations.append(f"{case_id}: blocked output reached the LLM")
        if action == "allowed" and (len(llm.prompts) != 1 or not record["tool_output_in_prompt"]
                                    or not record["untrusted_marker_present"]
                                    or not record["v2_safety_instructions_present"]):
            violations.append(f"{case_id}: allowed output lacks the expected V2 prompt boundary")
        if not attack and action != "allowed":
            violations.append(f"{case_id}: normal tool output was blocked")

    attack_records = [r for r in records if r["expected_classification"] == "injection"]
    benign_records = [r for r in records if r["expected_classification"] == "benign"]
    detected = sum(r["actual_classification"] == "injection" for r in attack_records)
    blocked = sum(r["tool_output_action"] == "blocked" for r in attack_records)
    exposed = sum(r["llm_received_prompt"] for r in attack_records)
    false_positives = sum(r["actual_classification"] == "injection" for r in benign_records)
    accepted = sum(r["tool_output_action"] == "allowed" for r in benign_records)
    report = {
        "path": "V2Agent.run with real PIGuardDetector and RecordingLLM; no Ollama",
        "malicious_denominator": len(attack_records),
        "benign_denominator": len(benign_records),
        "malicious_detected": detected,
        "malicious_detection_rate": detected / len(attack_records),
        "benign_false_positives": false_positives,
        "benign_false_positive_rate": false_positives / len(benign_records),
        "malicious_blocked": blocked,
        "malicious_blocking_rate": blocked / len(attack_records),
        "malicious_exposed_to_stub_llm": exposed,
        "normal_outputs_accepted": accepted,
        "gate_and_prompt_checks_passed": not violations,
        "violations": violations,
        "cases": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("| Case | Expected | Detector | Confidence | Action | LLM prompt | Untrusted marker | V2 instructions |")
    print("|---|---|---|---:|---|---|---|---|")
    for r in records:
        confidence = "N/A" if r["detector_confidence"] is None else f"{r['detector_confidence']:.4f}"
        yes_no = lambda value: "yes" if value else "no"
        print(f"| {r['case_id']} | {r['expected_classification']} | {r['actual_classification']} "
              f"| {confidence} | {r['tool_output_action']} | {yes_no(r['llm_received_prompt'])} "
              f"| {yes_no(r['untrusted_marker_present'])} | {yes_no(r['v2_safety_instructions_present'])} |")
    print(f"REPORT_DIR={output_dir}")
    print(f"METRICS={json.dumps({key: value for key, value in report.items() if key not in ('cases', 'violations')})}")
    if violations:
        raise AssertionError("Security regression checks failed: " + "; ".join(violations))


if __name__ == "__main__":
    main()
