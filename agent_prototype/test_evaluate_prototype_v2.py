"""Offline validation of V2 record and reviewer isolation."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agent import SecureAgent
from evaluate_prototype_v2 import (
    V2Agent, V2_ANSWER_INSTRUCTIONS, V2_TOOL_OUTPUT_MARKER,
    collection_status, load_cases, review_records,
)
from mock_search_v2 import MockSearchV2


def fixtures():
    cases, search = load_cases(), MockSearchV2()
    records, reviews = [], {}
    for case in cases:
        case_id, query = case["id"], case["query"]
        output = search.run(query).output
        action, response = "allowed", "A useful answer."
        if case_id == "tool-normal-09":
            action, response = "blocked", "Blocked tool output."
        elif case_id == "tool-normal-10":
            action, response = "error", "LLM error: Ollama returned no text response."
        records.append({"id": case_id, "case_id": case_id, "tool_name": "mock_search",
                        "result_id": search.results[query]["id"], "request": case["text"],
                        "query": query, "task_criteria": case["task_criteria"],
                        "tool_output": output, "expected_tool_output": output,
                        "action": action, "status": collection_status(action, response),
                        "response": response, "llm": "test-llm"})
        reviews[case_id] = {"task_score": 0, "grounding_score": "N/A" if action != "allowed" else 0,
                            "rationale": "Constructed example", "supporting_quote": ""}
    reviews["tool-normal-01"].update(task_score=2, grounding_score=2,
                                      supporting_quote="Defense in depth uses multiple independent protective layers")
    reviews["tool-normal-02"].update(task_score=2, grounding_score=0)
    reviews["tool-normal-03"].update(task_score=1, grounding_score=1,
                                      supporting_quote="Access control decides which identities")
    return records, reviews


class V2Checks(unittest.TestCase):
    @staticmethod
    def agent_dependencies():
        detector = Mock()
        detector.detect.return_value = SimpleNamespace(
            is_injection=False, label=0, confidence=0.0, error=None)
        logger = Mock()
        llm = Mock()
        llm.name = "offline-test-llm"
        llm.generate.return_value = "Offline answer"
        return detector, logger, llm

    def test_v2_prompt_contains_request_result_and_answer_guidance_only(self):
        detector, logger, llm = self.agent_dependencies()
        agent = V2Agent(detector, logger, llm)
        search = MockSearchV2()
        for case in load_cases():
            with self.subTest(case_id=case["id"]):
                llm.generate.reset_mock()
                result = agent.run(case["text"], tool="mock_search", tool_input=case["query"])
                expected_output = search.run(case["query"]).output
                expected_prompt = (
                    f"{V2_ANSWER_INSTRUCTIONS}\n\n"
                    f"User request:\n{case['text']}\n\n"
                    f"{V2_TOOL_OUTPUT_MARKER}{expected_output}"
                )
                llm.generate.assert_called_once_with(expected_prompt)
                self.assertEqual(result.tool_output, expected_output)
                self.assertNotIn(case["task_criteria"], expected_prompt)

    def test_legacy_prompt_construction_is_unchanged(self):
        detector, logger, llm = self.agent_dependencies()
        agent = SecureAgent(detector, logger, llm)
        case = load_cases()[0]
        result = agent.run(case["text"], tool="mock_search", tool_input=case["query"])
        llm.generate.assert_called_once_with(
            f"User request:\n{case['text']}\n\n"
            f"{V2_TOOL_OUTPUT_MARKER}{result.tool_output}"
        )
        self.assertNotIn(V2_ANSWER_INSTRUCTIONS, llm.generate.call_args.args[0])

    def test_result_mapping_and_repeatability(self):
        search = MockSearchV2()
        for case in load_cases():
            output = search.run(case["query"]).output
            self.assertEqual(output, search.run(case["query"]).output)
            self.assertIn("Result ID:", output)
            self.assertIn("Content:", output)

    def test_valid_reviews_preserve_evidence_and_denominator(self):
        records, reviews = fixtures()
        before = copy.deepcopy(records)
        result = review_records(records, reviews)
        self.assertEqual(records, before)
        self.assertEqual(result["v2_denominator"], 10)
        self.assertEqual(result["original_keyword_result"]["completed"], 3)
        self.assertEqual(result["semantic_completion"], 2)
        self.assertEqual(result["grounded_completion"], 1)
        self.assertEqual(result["partial_completion"], 1)
        self.assertEqual(result["blocked"], 1)
        self.assertEqual(result["model_response_failure"], 1)
        self.assertEqual(result["cases"][8]["status"], "blocked")
        self.assertEqual(result["cases"][9]["status"], "empty_response")

    def test_rejects_overwrite_attempts_and_extra_fields(self):
        for field in ("status", "response", "tool_output", "case_id", "tool_name", "result_id", "extra"):
            with self.subTest(field=field):
                records, reviews = fixtures()
                reviews["tool-normal-01"][field] = "forged"
                with self.assertRaises(ValueError):
                    review_records(records, reviews)

    def test_rejects_invalid_case_and_result_ids(self):
        for field in ("case_id", "id", "result_id", "tool_output"):
            with self.subTest(field=field):
                records, reviews = fixtures()
                records[0][field] = "forged"
                with self.assertRaises(ValueError):
                    review_records(records, reviews)
        records, reviews = fixtures()
        reviews["forged"] = reviews.pop("tool-normal-01")
        with self.assertRaises(ValueError):
            review_records(records, reviews)

    def test_blocked_and_empty_cannot_receive_scores(self):
        for case_id in ("tool-normal-09", "tool-normal-10"):
            with self.subTest(case_id=case_id):
                records, reviews = fixtures()
                reviews[case_id]["task_score"] = 2
                with self.assertRaises(ValueError):
                    review_records(records, reviews)


if __name__ == "__main__":
    unittest.main()
