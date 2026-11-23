import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("mof_pilot_qwen3_4b_v4.py")
SPEC = importlib.util.spec_from_file_location("mof_pilot_v4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class V4ExtractionTests(unittest.TestCase):
    def test_extracts_sentence_from_structured_generation_response(self):
        sentence = "The employee completed their duties before leaving work."
        generation = {
            "raw": '{"sentence": "The employee completed their duties before leaving work."}',
            "answer": '{"sentence": "The employee completed their duties before leaving work."}',
        }
        self.assertEqual(MODULE.extract_sentence(generation), sentence)
        self.assertNotEqual(MODULE.extract_sentence(generation), "duties")

    def test_standalone_candidate_remains_rejected(self):
        checks, reasons = MODULE.deterministic_checks("duties", "duties")
        self.assertFalse(checks["not_standalone_candidate"])
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
