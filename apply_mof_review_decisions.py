import argparse
import ast
import csv
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "logs" / "mof_baseline_20261023" / "mof_generate_1000"
GENERATED_PATH = EXPERIMENT_DIR / "accepted_samples.jsonl"
AUDIT_PATH = EXPERIMENT_DIR / "audit_review.csv"
FLAGGED_REVIEW_PATH = EXPERIMENT_DIR / "flagged_manual_review.csv"
UNFLAGGED_REVIEW_PATH = EXPERIMENT_DIR / "unflagged_random_manual_review.csv"
OUTPUT_PATH = EXPERIMENT_DIR / "reviewed_accepted_samples.jsonl"
VALID_DECISIONS = {"keep", "edit", "reject"}


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object.")
            rows.append(value)
    return rows


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def parse_id(value, source):
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid sample_id {value!r} in {source}.") from error


def parse_tokens(value, source):
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"Malformed candidate_tokens in {source}.") from error
    else:
        raise ValueError(f"Missing candidate_tokens in {source}.")
    if not isinstance(parsed, list) or any(
        not isinstance(token, str) or not token for token in parsed
    ):
        raise ValueError(f"candidate_tokens must be a list of nonempty strings in {source}.")
    return parsed


def index_unique(rows, name, id_getter):
    indexed = {}
    duplicates = []
    for row in rows:
        identifier = id_getter(row)
        if identifier in indexed:
            duplicates.append(identifier)
        indexed[identifier] = row
    if duplicates:
        raise ValueError(f"Duplicate sample IDs in {name}: {sorted(set(duplicates))}.")
    return indexed


def normalized_label(value, source):
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean label in {source}.")
    try:
        label = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid label {value!r} in {source}.") from error
    if label != 0:
        raise ValueError(f"Expected benign label 0 in {source}, found {value!r}.")
    return label


def validate_and_build(generated_rows, audit_rows, review_rows):
    generated_by_id = {
        identifier: row
        for identifier, row in enumerate(generated_rows, 1)
    }
    audit_by_id = index_unique(
        audit_rows,
        "audit_review.csv",
        lambda row: parse_id(row.get("sample_id"), "audit_review.csv"),
    )
    if set(audit_by_id) != set(generated_by_id):
        raise ValueError("Audit IDs do not exactly reconcile with generated JSONL order.")

    for identifier, generated in generated_by_id.items():
        audit = audit_by_id[identifier]
        generated_prompt = generated.get("prompt")
        if not isinstance(generated_prompt, str) or not generated_prompt.strip():
            raise ValueError(f"Generated prompt is empty for sample {identifier}.")
        generated_tokens = parse_tokens(generated.get("candidate_tokens"), f"generated sample {identifier}")
        normalized_label(generated.get("label"), f"generated sample {identifier}")
        if audit.get("prompt", "") != generated_prompt.strip():
            raise ValueError(f"Audit prompt does not reconcile for sample {identifier}.")
        if parse_tokens(audit.get("candidate_tokens"), f"audit sample {identifier}") != generated_tokens:
            raise ValueError(f"Audit candidate_tokens do not reconcile for sample {identifier}.")
        if normalized_label(audit.get("label"), f"audit sample {identifier}") != generated["label"]:
            raise ValueError(f"Audit label does not reconcile for sample {identifier}.")

    review_by_id = index_unique(
        review_rows,
        "combined review CSVs",
        lambda row: parse_id(row.get("sample_id"), "review CSV"),
    )
    if not review_by_id:
        raise ValueError("No review decisions were found.")

    selected = []
    counts = {"keep": 0, "edit": 0, "reject": 0}
    for identifier, review in review_by_id.items():
        if identifier not in generated_by_id or identifier not in audit_by_id:
            raise ValueError(f"Review sample ID {identifier} is not in both source files.")
        decision = review.get("manual_review", "").strip().lower()
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid or blank decision for sample {identifier}: {review.get('manual_review')!r}.")
        counts[decision] += 1
        generated = generated_by_id[identifier]
        source_tokens = parse_tokens(generated["candidate_tokens"], f"generated sample {identifier}")
        review_tokens = parse_tokens(review.get("candidate_tokens"), f"review sample {identifier}")
        if review_tokens != source_tokens:
            raise ValueError(f"Candidate tokens changed for sample {identifier}.")
        review_label = normalized_label(review.get("label"), f"review sample {identifier}")
        if review_label != generated["label"]:
            raise ValueError(f"Label changed for sample {identifier}.")

        selected_prompt = generated["prompt"]
        prompt_source = "generated"
        if decision == "edit":
            selected_prompt = review.get("prompt", "")
            prompt_source = "manual_edit"
        if not isinstance(selected_prompt, str) or not selected_prompt.strip():
            raise ValueError(f"Selected prompt is empty for sample {identifier}.")
        missing = [token for token in source_tokens if token not in selected_prompt]
        if decision in {"keep", "edit"} and missing:
            raise ValueError(f"Sample {identifier} is missing candidate tokens: {missing!r}.")
        if decision == "keep" and review.get("prompt", "") != generated["prompt"].strip():
            # Keep deliberately ignores an accidental prompt edit in the CSV.
            prompt_source = "generated"

        if decision != "reject":
            output = dict(generated)
            output["prompt"] = selected_prompt
            output["sample_id"] = identifier
            output["manual_review_decision"] = decision
            output["prompt_source"] = prompt_source
            selected.append(output)

    prompts = {}
    for row in selected:
        prompt = row["prompt"]
        if prompt in prompts:
            raise ValueError(
                f"Duplicate accepted prompt for sample IDs {prompts[prompt]} and {row['sample_id']}."
            )
        prompts[prompt] = row["sample_id"]
    return selected, counts


def create_output(output_path=OUTPUT_PATH):
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    generated = read_jsonl(GENERATED_PATH)
    audit = read_csv(AUDIT_PATH)
    flagged = read_csv(FLAGGED_REVIEW_PATH)
    unflagged = read_csv(UNFLAGGED_REVIEW_PATH)
    selected, counts = validate_and_build(generated, audit, flagged + unflagged)

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
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link creation fails rather than replacing a concurrently created target.
        os.link(temporary_path, output_path)
        temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return {
        "output": str(output_path),
        "approved_count": len(selected),
        "decision_counts": counts,
        "reviewed_count": len(flagged) + len(unflagged),
        "unreviewed_excluded": len(generated) - len(flagged) - len(unflagged),
        "source_generated_count": len(generated),
        "source_audit_count": len(audit),
    }


def main():
    parser = argparse.ArgumentParser(description="Create approved MOF samples from manual decisions.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = create_output(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
