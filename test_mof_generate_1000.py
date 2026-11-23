import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("mof_generate_1000.py")
SPEC = importlib.util.spec_from_file_location("mof_generate_1000", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Generate1000Tests(unittest.TestCase):
    def candidate_file(self, directory):
        path = directory / "candidates.jsonl"
        rows = [
            {"decoded_text": "Alpha"},
            {"decoded_text": "Beta"},
            {"decoded_text": "Gamma"},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_combination_sizes_and_exact_matching(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            candidates = self.candidate_file(directory)
            for tokens in (["Alpha"], ["Alpha", "Beta"], ["Alpha", "Beta", "Gamma"]):
                checks, reasons = MODULE.deterministic_checks(
                    tokens,
                    "The " + " ".join(tokens) + " discussion continued calmly today.",
                )
                self.assertIn(len(tokens), (1, 2, 3))
                self.assertTrue(checks["exact_case_sensitive_tokens"])
                self.assertFalse(reasons)
            self.assertEqual(MODULE.read_candidates(candidates)[0], ["Alpha", "Beta", "Gamma"])

    def test_malformed_json_and_invalid_verdict(self):
        self.assertIsNone(MODULE.parse_json("{not json"))
        valid, reason = MODULE.valid_judge_verdict({"benign": True})
        self.assertFalse(valid)
        self.assertIn("missing", reason)

    def test_target_stop_duplicate_and_resume(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            candidates = self.candidate_file(directory)
            calls = {"count": 0}

            def fake_chat(prompt, schema, max_tokens):
                calls["count"] += 1
                if "sentence" in schema["properties"]:
                    return {
                        "raw": '{"sentence":"The Alpha Beta Gamma study continued today."}',
                        "answer": '{"sentence":"The Alpha Beta Gamma study continued today."}',
                        "done_reason": "stop",
                    }
                return {
                    "raw": '{"benign":true,"contains_candidate":true,"accept":true,"reason":"clear"}',
                    "answer": '{"benign":true,"contains_candidate":true,"accept":true,"reason":"clear"}',
                    "done_reason": "stop",
                }

            original_chat = MODULE.chat
            MODULE.chat = fake_chat
            try:
                output = directory / "output"
                summary = MODULE.run_pipeline(1, 3, output, candidates, sleep_seconds=0)
                self.assertEqual(summary["accepted_count"], 1)
                self.assertEqual(summary["attempts"], 1)
                resumed = MODULE.run_pipeline(2, 3, output, candidates, sleep_seconds=0)
                self.assertEqual(resumed["accepted_count"], 1)
                self.assertTrue((output / "progress.json").exists())
            finally:
                MODULE.chat = original_chat

    def test_attempt_limit_and_original_dataset_unchanged(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            candidates = self.candidate_file(directory)
            original = directory / "train.json"
            original.write_text('{"prompt":"keep","label":0}\n', encoding="utf-8")

            def reject_chat(prompt, schema, max_tokens):
                if "sentence" in schema["properties"]:
                    return {
                        "raw": '{"sentence":"Alpha"}',
                        "answer": '{"sentence":"Alpha"}',
                        "done_reason": "stop",
                    }
                return {}

            original_chat = MODULE.chat
            MODULE.chat = reject_chat
            before = original.read_bytes()
            try:
                summary = MODULE.run_pipeline(2, 2, directory / "output", candidates, sleep_seconds=0)
            finally:
                MODULE.chat = original_chat
            self.assertEqual(summary["accepted_count"], 0)
            self.assertEqual(summary["attempts"], 2)
            self.assertEqual(original.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
