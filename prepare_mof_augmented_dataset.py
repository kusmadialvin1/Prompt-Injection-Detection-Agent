import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORIGINAL_PATH = ROOT / "datasets" / "train.json"
REVIEWED_PATH = (
    ROOT
    / "logs"
    / "mof_baseline_20261023"
    / "mof_generate_1000"
    / "reviewed_accepted_samples.jsonl"
)
OUTPUT_PATH = ROOT / "datasets" / "train_mof_reviewed_54.json"
EXPECTED_ORIGINAL_COUNT = 76_735
EXPECTED_REVIEWED_COUNT = 54
VALID_LABELS = {0, 1}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_original(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read original JSON array {path}: {error}") from error
    if not isinstance(records, list):
        raise ValueError(f"Original dataset must be a JSON array: {path}")
    return records


def read_reviewed_jsonl(path):
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed reviewed JSONL at {path}:{line_number}: {error}"
                    ) from error
                records.append(record)
    except OSError as error:
        raise ValueError(f"Could not read reviewed JSONL {path}: {error}") from error
    return records



def validate_record(record, source, index, allow_empty_prompt=False):
    if not isinstance(record, dict):
        raise ValueError(f"{source} record {index} must be a JSON object.")

    prompt = record.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"{source} record {index} has an invalid prompt type.")

    if not prompt.strip() and not allow_empty_prompt:
        raise ValueError(f"{source} record {index} has an empty or invalid prompt.")

    label = record.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label not in VALID_LABELS:
        raise ValueError(
            f"{source} record {index} has invalid label {label!r}; expected integer 0 or 1."
        )



def duplicate_values(values):
    seen = set()
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def validate_inputs(original, reviewed):
    if len(original) != EXPECTED_ORIGINAL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ORIGINAL_COUNT} original records, found {len(original)}."
        )
    if len(reviewed) != EXPECTED_REVIEWED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_REVIEWED_COUNT} reviewed records, found {len(reviewed)}."
        )

    for index, record in enumerate(original, 1):
        validate_record(record, "Original", index, allow_empty_prompt=True)
    for index, record in enumerate(reviewed, 1):
        validate_record(record, "Reviewed", index)
        if record["label"] != 0:
            raise ValueError(f"Reviewed record {index} must have label 0.")

    original_prompts = {record["prompt"] for record in original}
    reviewed_prompts = [record["prompt"] for record in reviewed]
    duplicates = duplicate_values(reviewed_prompts)
    if duplicates:
        raise ValueError(
            f"Found {len(duplicates)} duplicate reviewed prompts; examples: {duplicates[:3]!r}"
        )
    overlap = sorted(set(reviewed_prompts) & original_prompts)
    if overlap:
        raise ValueError(
            f"Found {len(overlap)} reviewed prompts already in the original dataset; "
            f"examples: {overlap[:3]!r}"
        )


def validate_output(output, original, reviewed):
    expected_count = len(original) + len(reviewed)
    if len(output) != expected_count:
        raise ValueError(f"Expected {expected_count} output records, found {len(output)}.")
    for index, record in enumerate(output, 1):
        validate_record(record, "Output", index, allow_empty_prompt=(index <= len(original)),)
    if output[: len(original)] != original:
        raise ValueError("Original records were not preserved exactly and in order.")
    appended = output[len(original) :]
    if appended != reviewed:
        raise ValueError("Reviewed records were not appended exactly.")
    if any(record["label"] != 0 for record in appended):
        raise ValueError("Appended records contain a nonzero label.")
    original_prompts = {record["prompt"] for record in original}
    appended_prompts = [record["prompt"] for record in appended]
    if len(set(appended_prompts)) != len(appended_prompts):
        raise ValueError("Appended prompts are not unique.")
    if original_prompts.intersection(appended_prompts):
        raise ValueError("An appended prompt occurs in the original dataset.")


def write_atomically(output_path, records):
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=output_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # On Windows, rename fails if another process created output_path.
        os.rename(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()



def prepare_dataset(original_path=ORIGINAL_PATH, reviewed_path=REVIEWED_PATH, output_path=OUTPUT_PATH):
    original_path = Path(original_path).resolve()
    reviewed_path = Path(reviewed_path).resolve()
    output_path = Path(output_path).resolve()

    if original_path == reviewed_path:
        raise ValueError("Original and reviewed paths must be different.")
    if output_path == original_path or output_path == reviewed_path:
        raise ValueError("Output path must be different from both input paths.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    original = read_original(original_path)
    reviewed = read_reviewed_jsonl(reviewed_path)
    validate_inputs(original, reviewed)

    output = original + reviewed
    write_atomically(output_path, output)

    try:
        reloaded = read_original(output_path)
        validate_output(reloaded, original, reviewed)
    except Exception:
        # Remove only the output created by this run, and only if it
        # still contains the data this run wrote.
        if output_path.exists():
            try:
                current = read_original(output_path)
                if current == output:
                    output_path.unlink()
            except Exception:
                pass
        raise

    return {
        "status": "PASS",
        "original_count": len(original),
        "reviewed_count": len(reviewed),
        "output_count": len(reloaded),
        "appended_label_counts": dict(
            Counter(record["label"] for record in reviewed)
        ),
        "final_label_counts": dict(
            Counter(record["label"] for record in reloaded)
        ),
        "original_sha256": sha256_file(original_path),
        "reviewed_sha256": sha256_file(reviewed_path),
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
    }



def main():
    parser = argparse.ArgumentParser(description="Create the reviewed MOF augmented dataset.")
    parser.add_argument("--original", type=Path, default=ORIGINAL_PATH)
    parser.add_argument("--reviewed", type=Path, default=REVIEWED_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.original, args.reviewed, args.output), indent=2))


if __name__ == "__main__":
    main()
