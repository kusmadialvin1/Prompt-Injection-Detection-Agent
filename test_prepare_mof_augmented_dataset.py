import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "prepare_mof_augmented_dataset",
    Path(__file__).with_name("prepare_mof_augmented_dataset.py"),
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareMofAugmentedDatasetTests(unittest.TestCase):
    def write_files(self, directory, original, reviewed):
        original_path = directory / "train.json"
        reviewed_path = directory / "reviewed.jsonl"
        output_path = directory / "augmented.json"
        original_path.write_text(json.dumps(original), encoding="utf-8")
        reviewed_path.write_text(
            "".join(json.dumps(record) + "\n" for record in reviewed),
            encoding="utf-8",
        )
        return original_path, reviewed_path, output_path

    def records(self, count, prefix="original", label=0):
        return [
            {"prompt": f"{prefix} prompt {index}", "label": label, "source": prefix}
            for index in range(count)
        ]

    def test_normal_success(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            original = self.records(3)
            reviewed = self.records(2, prefix="reviewed")
            paths = self.write_files(directory, original, reviewed)
            original_path, reviewed_path, output_path = paths
            old_counts = (
                MODULE.EXPECTED_ORIGINAL_COUNT,
                MODULE.EXPECTED_REVIEWED_COUNT,
            )
            MODULE.EXPECTED_ORIGINAL_COUNT = 3
            MODULE.EXPECTED_REVIEWED_COUNT = 2
            try:
                result = MODULE.prepare_dataset(original_path, reviewed_path, output_path)
            finally:
                MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT = old_counts
            self.assertEqual(result["output_count"], 5)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), original + reviewed)

    def test_invalid_boolean_label(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            original = self.records(3)
            original[0]["label"] = True
            reviewed = self.records(2, prefix="reviewed")
            paths = self.write_files(directory, original, reviewed)
            old_counts = MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT
            MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT = 3, 2
            try:
                with self.assertRaisesRegex(ValueError, "invalid label"):
                    MODULE.prepare_dataset(*paths)
            finally:
                MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT = old_counts

    def test_duplicate_reviewed_prompts(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            original = self.records(3)
            reviewed = self.records(2, prefix="reviewed")
            reviewed[1]["prompt"] = reviewed[0]["prompt"]
            paths = self.write_files(directory, original, reviewed)
            old_counts = MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT
            MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT = 3, 2
            try:
                with self.assertRaisesRegex(ValueError, "duplicate reviewed prompts"):
                    MODULE.prepare_dataset(*paths)
            finally:
                MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT = old_counts

    def test_incorrect_input_count(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            paths = self.write_files(directory, self.records(2), self.records(2, prefix="reviewed"))
            old_counts = MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT
            MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT = 3, 2
            try:
                with self.assertRaisesRegex(ValueError, "original records"):
                    MODULE.prepare_dataset(*paths)
            finally:
                MODULE.EXPECTED_ORIGINAL_COUNT, MODULE.EXPECTED_REVIEWED_COUNT = old_counts

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            paths = self.write_files(directory, self.records(3), self.records(2, prefix="reviewed"))
            paths[2].write_text("sentinel", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                MODULE.prepare_dataset(*paths)
            self.assertEqual(paths[2].read_text(encoding="utf-8"), "sentinel")


if __name__ == "__main__":
    unittest.main()
