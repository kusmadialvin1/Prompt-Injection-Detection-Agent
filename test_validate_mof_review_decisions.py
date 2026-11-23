import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "validate_mof_review_decisions",
    ROOT / "validate_mof_review_decisions.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReviewDecisionTests(unittest.TestCase):
    def write_files(self, directory, rows):
        audit = directory / "audit.csv"
        flagged = directory / "flagged.csv"
        unflagged = directory / "unflagged.csv"
        fields = ["sample_id", "prompt", "candidate_tokens", "label", "flags", "manual_review"]
        with audit.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "manual_review": ""})
        with flagged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[:1])
        with unflagged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[1:])
        return audit, flagged, unflagged

    def test_valid_decisions_and_case_insensitive_values(self):
        with tempfile.TemporaryDirectory() as raw:
            rows = [
                {
                    "sample_id": "1",
                    "prompt": "The Alpha sample is clearly harmless today.",
                    "candidate_tokens": json.dumps(["Alpha"]),
                    "label": "0",
                    "flags": "flag",
                    "manual_review": "KEEP",
                },
                {
                    "sample_id": "2",
                    "prompt": "The Beta sample is clearly harmless today.",
                    "candidate_tokens": json.dumps(["Beta"]),
                    "label": "0",
                    "flags": "",
                    "manual_review": "reject",
                },
            ]
            audit, flagged, unflagged = self.write_files(Path(raw), rows)
            result = MODULE.validate_review_decisions(audit, flagged, unflagged)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["files"]["flagged"]["counts"], {"keep": 1})
            self.assertEqual(result["files"]["unflagged"]["counts"], {"reject": 1})

    def test_reports_blank_missing_token_malformed_list_and_duplicates(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            rows = [
                {
                    "sample_id": "1",
                    "prompt": "The Alpha sample is clearly harmless today.",
                    "candidate_tokens": json.dumps(["Alpha"]),
                    "label": "0",
                    "flags": "flag",
                    "manual_review": "keep",
                },
                {
                    "sample_id": "2",
                    "prompt": "",
                    "candidate_tokens": "not-a-list",
                    "label": "0",
                    "flags": "",
                    "manual_review": "",
                },
            ]
            audit, flagged, unflagged = self.write_files(directory, rows)
            with unflagged.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "prompt", "candidate_tokens", "label", "flags", "manual_review"],
                )
                writer.writeheader()
                writer.writerow(rows[0])
                writer.writerow(rows[1])
            result = MODULE.validate_review_decisions(audit, flagged, unflagged)
            problems = "\n".join(item["problem"] for item in result["problems"])
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("Duplicate sample ID", problems)
            self.assertIn("Blank manual_review", problems)
            self.assertIn("Malformed candidate-token list", problems)
            self.assertIn("Prompt is empty", problems)

    def test_reports_missing_candidate_for_keep(self):
        with tempfile.TemporaryDirectory() as raw:
            rows = [
                {
                    "sample_id": "1",
                    "prompt": "The sample is clearly harmless today.",
                    "candidate_tokens": json.dumps(["Alpha"]),
                    "label": "0",
                    "flags": "flag",
                    "manual_review": "keep",
                }
            ]
            audit, flagged, unflagged = self.write_files(Path(raw), rows)
            result = MODULE.validate_review_decisions(audit, flagged, unflagged)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("missing candidate token", result["problems"][0]["problem"].lower())


if __name__ == "__main__":
    unittest.main()
