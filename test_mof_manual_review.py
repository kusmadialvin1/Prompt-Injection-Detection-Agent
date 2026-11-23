import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_mof_manual_review", ROOT / "prepare_mof_manual_review.py"
)
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(PREPARE)
VALIDATE_SPEC = importlib.util.spec_from_file_location(
    "validate_mof_manual_review", ROOT / "validate_mof_manual_review.py"
)
VALIDATE = importlib.util.module_from_spec(VALIDATE_SPEC)
VALIDATE_SPEC.loader.exec_module(VALIDATE)


class ManualReviewTests(unittest.TestCase):
    def write_audit(self, directory, count=20):
        path = directory / "audit.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREPARE.REVIEW_FIELDS)
            writer.writeheader()
            for identifier in range(1, count + 1):
                flagged = identifier in PREPARE.PROVISIONAL_DECISIONS
                writer.writerow(
                    {
                        "sample_id": identifier,
                        "prompt": f"Sample {identifier} is harmless text.",
                        "candidate_tokens": f'["token{identifier}"]',
                        "label": "0",
                        "flags": "possible_instruction" if flagged else "",
                        "manual_review": "",
                    }
                )
        return path

    def test_deterministic_selection_and_preservation(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            audit = self.write_audit(directory, 900)
            flagged_path = directory / "flagged.csv"
            random_path = directory / "random.csv"
            flagged, random_rows = PREPARE.prepare_reviews(
                audit, flagged_path, random_path, seed=42, unflagged_count=5
            )
            self.assertEqual(len(flagged), 13)
            self.assertEqual(len(random_rows), 5)
            self.assertEqual(
                [row["sample_id"] for row in random_rows],
                [row["sample_id"] for row in PREPARE.prepare_reviews(
                    audit, directory / "flagged2.csv", directory / "random2.csv",
                    seed=42, unflagged_count=5
                )[1]],
            )
            self.assertEqual(flagged[0]["prompt"], "Sample 5 is harmless text.")
            self.assertEqual(flagged[0]["candidate_tokens"], '["token5"]')
            self.assertEqual({row["manual_review"] for row in flagged}, {"keep", "edit"})
            self.assertEqual({row["manual_review"] for row in random_rows}, {""})

    def test_validation_errors_for_overlap_and_changed_prompt(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            audit = self.write_audit(directory, 900)
            flagged_path = directory / "flagged.csv"
            random_path = directory / "random.csv"
            PREPARE.prepare_reviews(audit, flagged_path, random_path, seed=42, unflagged_count=5)
            with random_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["prompt"] = "changed"
            with random_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=PREPARE.REVIEW_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                VALIDATE.validate_reviews(
                    audit, flagged_path, random_path, directory / "missing-train.json",
                    expected_train_sha256="unused",
                )

    def test_train_hash_verification(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            train = directory / "train.json"
            train.write_text('{"label": 0}\n', encoding="utf-8")
            self.assertEqual(len(VALIDATE.sha256_file(train)), 64)


if __name__ == "__main__":
    unittest.main()
