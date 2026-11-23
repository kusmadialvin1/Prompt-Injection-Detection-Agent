"""Offline HTTP-mock checks for V2 diagnostics; never connects to Ollama."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from agent import OllamaLLM
from ollama_diagnostics_v2 import DiagnosticOllamaLLM


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "diagnostics.jsonl"
        self.llm = DiagnosticOllamaLLM(self.path)
        self.llm.case_id = "tool-normal-01"

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, payload=None, *, json_error=None, http_error=None, status=200):
        response = Mock(status_code=status)
        response.raise_for_status.side_effect = http_error
        response.json.side_effect = json_error
        if json_error is None:
            response.json.return_value = payload
        with patch("ollama_diagnostics_v2.requests.post", return_value=response) as post:
            try:
                result = self.llm.generate("private prompt")
            finally:
                post.assert_called_once_with(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": "qwen3:4b-nothink", "prompt": "private prompt",
                          "stream": False, "options": {"num_predict": 512}}, timeout=120)
        return result

    def event(self):
        events = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(events), 1)
        self.assertNotIn("private prompt", events[0])
        return json.loads(events[0])

    def test_normal_response_and_metadata(self):
        self.assertEqual(self.invoke({"response": "Answer", "done": True,
                                      "done_reason": "stop", "eval_count": 7,
                                      "prompt_eval_count": 34, "total_duration": 12345}), "Answer")
        event = self.event()
        self.assertEqual(event["case_id"], "tool-normal-01")
        self.assertEqual(event["model"], "qwen3:4b-nothink")
        self.assertEqual(event["request_options"], {"num_predict": 512})
        self.assertEqual(event["http_status"], 200)
        self.assertTrue(event["json_parsed"])
        self.assertTrue(event["response_field_exists"])
        self.assertEqual(event["response_field_type"], "str")
        self.assertTrue(event["usable_text"])
        self.assertEqual(event["done_reason"], "stop")
        self.assertEqual(event["eval_count"], 7)
        self.assertEqual(event["prompt_eval_count"], 34)
        self.assertEqual(event["total_duration"], 12345)
        self.assertFalse(event["thinking_field_exists"])
        self.assertIsNone(event["thinking_field_type"])
        self.assertIsNone(event["thinking_has_text"])
        self.assertIsNone(event["thinking_char_length"])

    def test_thinking_is_measured_separately_without_saving_content(self):
        trace = "private reasoning trace"
        self.assertEqual(self.invoke({"thinking": trace, "response": "Final answer",
                                      "done_reason": "length", "eval_count": 256,
                                      "total_duration": 12345}), "Final answer")
        event = self.event()
        self.assertTrue(event["thinking_field_exists"])
        self.assertEqual(event["thinking_field_type"], "str")
        self.assertTrue(event["thinking_has_text"])
        self.assertEqual(event["thinking_char_length"], len(trace))
        self.assertTrue(event["usable_text"])
        self.assertEqual(event["eval_count"], 256)
        self.assertNotIn(trace, self.path.read_text(encoding="utf-8"))

    def test_thinking_with_empty_final_response_is_still_logged(self):
        with self.assertRaisesRegex(RuntimeError, "Ollama returned no text response"):
            self.invoke({"thinking": "reasoning", "response": "", "done_reason": "length",
                         "eval_count": 256})
        event = self.event()
        self.assertTrue(event["thinking_has_text"])
        self.assertFalse(event["usable_text"])
        self.assertEqual(event["error_category"], "empty_response")

    def test_blank_thinking(self):
        self.assertEqual(self.invoke({"thinking": "  ", "response": "Answer"}), "Answer")
        event = self.event()
        self.assertTrue(event["thinking_field_exists"])
        self.assertFalse(event["thinking_has_text"])
        self.assertEqual(event["thinking_char_length"], 2)

    def test_non_string_thinking(self):
        self.assertEqual(self.invoke({"thinking": ["not text"], "response": "Answer"}), "Answer")
        event = self.event()
        self.assertEqual(event["thinking_field_type"], "list")
        self.assertFalse(event["thinking_has_text"])
        self.assertIsNone(event["thinking_char_length"])

    def test_empty_response(self):
        with self.assertRaisesRegex(RuntimeError, "Ollama returned no text response"):
            self.invoke({"response": "  ", "done": True})
        self.assertEqual(self.event()["error_category"], "empty_response")

    def test_missing_response(self):
        with self.assertRaisesRegex(RuntimeError, "Ollama returned no text response"):
            self.invoke({"done": True})
        event = self.event()
        self.assertFalse(event["response_field_exists"])
        self.assertEqual(event["error_category"], "missing_response")

    def test_non_string_response(self):
        with self.assertRaisesRegex(RuntimeError, "Ollama returned no text response"):
            self.invoke({"response": 42})
        event = self.event()
        self.assertEqual(event["response_field_type"], "int")
        self.assertEqual(event["error_category"], "non_string_response")

    def test_malformed_json(self):
        with self.assertRaises(ValueError):
            self.invoke(json_error=ValueError("bad JSON"))
        event = self.event()
        self.assertFalse(event["json_parsed"])
        self.assertEqual(event["error_category"], "malformed_json")

    def test_http_error(self):
        with self.assertRaises(requests.HTTPError):
            self.invoke(http_error=requests.HTTPError("503 unavailable"), status=503)
        event = self.event()
        self.assertEqual(event["http_status"], 503)
        self.assertEqual(event["error_category"], "http_error")

    def test_legacy_client_unchanged(self):
        self.assertIsNot(OllamaLLM._generate, DiagnosticOllamaLLM._generate)
        self.assertIsNot(OllamaLLM.generate, DiagnosticOllamaLLM.generate)
        with patch.object(OllamaLLM, "_generate", return_value="legacy answer") as generate:
            self.assertEqual(OllamaLLM().generate("prompt"), "legacy answer")
            generate.assert_called_once_with("prompt", {})


if __name__ == "__main__":
    unittest.main()
