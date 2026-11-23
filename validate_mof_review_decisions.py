import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_mof_manual_review import (
    AUDIT_PATH,
    FLAGGED_REVIEW_PATH,
    REVIEW_FIELDS,
    UNFLAGGED_REVIEW_PATH,
    read_audit_rows,
    sample_id,
)


ROOT = Path(__file__).resolve().parent
VALID_DECISIONS = {"keep", "edit", "reject"}


def read_review(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(REVIEW_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Missing review columns in {path}")
        return list(reader)


def parse_candidate_tokens(value):
    if not isinstance(value, str) or not value.strip():
        return None, "candidate_tokens is blank."
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None, "candidate_tokens is not valid JSON/list syntax."
    if not isinstance(parsed, list) or any(
        not isinstance(token, str) or not token for token in parsed
    ):
        return None, "candidate_tokens must be a list of nonempty strings."
    return parsed, None


def validate_review_decisions(
    audit_path=AUDIT_PATH,
    flagged_path=FLAGGED_REVIEW_PATH,
    unflagged_path=UNFLAGGED_REVIEW_PATH,
):
    audit_rows = read_audit_rows(audit_path)
    audit_by_id = {}
    duplicate_audit_ids = []
    for row in audit_rows:
        identifier = sample_id(row)
        if identifier in audit_by_id:
            duplicate_audit_ids.append(identifier)
        audit_by_id[identifier] = row

    rows_by_file = {
        "flagged": read_review(flagged_path),
        "unflagged": read_review(unflagged_path),
    }
    counts = {}
    problems = []
    seen_ids = {}
    prompt_changes = []
    candidate_token_changes = []

    for file_name, rows in rows_by_file.items():
        file_counts = Counter()
        for row in rows:
            raw_id = row.get("sample_id", "")
            try:
                identifier = sample_id(row)
            except ValueError as error:
                problems.append(
                    {"file": file_name, "sample_id": raw_id, "problem": str(error)}
                )
                continue

            if identifier in seen_ids:
                problems.append(
                    {
                        "file": file_name,
                        "sample_id": identifier,
                        "problem": (
                            "Duplicate sample ID across review files "
                            f"(also in {seen_ids[identifier]})."
                        ),
                    }
                )
            else:
                seen_ids[identifier] = file_name

            decision = row.get("manual_review", "").strip().lower()
            if decision not in VALID_DECISIONS:
                file_counts["blank" if not decision else "invalid"] += 1
                problems.append(
                    {
                        "file": file_name,
                        "sample_id": identifier,
                        "problem": (
                            "Blank manual_review decision."
                            if not decision
                            else f"Invalid manual_review decision: {row.get('manual_review')!r}."
                        ),
                    }
                )
            else:
                file_counts[decision] += 1

            source = audit_by_id.get(identifier)
            if source is None:
                problems.append(
                    {
                        "file": file_name,
                        "sample_id": identifier,
                        "problem": "Sample ID is absent from the original audit report.",
                    }
                )
                continue

            source_tokens, source_error = parse_candidate_tokens(
                source.get("candidate_tokens", "")
            )
            review_tokens, review_error = parse_candidate_tokens(
                row.get("candidate_tokens", "")
            )
            prompt = row.get("prompt", "")
            if not prompt.strip():
                problems.append(
                    {
                        "file": file_name,
                        "sample_id": identifier,
                        "problem": "Prompt is empty.",
                    }
                )
            elif prompt != source.get("prompt", ""):
                prompt_changes.append(identifier)
            if source_error:
                problems.append(
                    {
                        "file": "audit_review.csv",
                        "sample_id": identifier,
                        "problem": source_error,
                    }
                )
            if review_error:
                problems.append(
                    {
                        "file": file_name,
                        "sample_id": identifier,
                        "problem": f"Malformed candidate-token list: {review_error}",
                    }
                )

            if source_tokens is not None and review_tokens is not None:
                if review_tokens != source_tokens:
                    candidate_token_changes.append(identifier)
                    problems.append(
                        {
                            "file": file_name,
                            "sample_id": identifier,
                            "problem": "Candidate-token list changed from the audit report.",
                        }
                    )

                if decision in {"keep", "edit"}:
                    missing = [token for token in review_tokens if token not in prompt]
                    if missing:
                        problems.append(
                            {
                                "file": file_name,
                                "sample_id": identifier,
                                "problem": (
                                    "Keep/edit prompt is missing candidate token(s): "
                                    + repr(missing)
                                ),
                            }
                        )
        counts[file_name] = dict(file_counts)

    if duplicate_audit_ids:
        problems.append(
            {
                "file": "audit_review.csv",
                "sample_id": duplicate_audit_ids,
                "problem": "Duplicate sample IDs in original audit report.",
            }
        )

    return {
        "status": "PASS" if not problems else "FAIL",
        "files": {
            name: {
                "rows": len(rows),
                "counts": counts[name],
            }
            for name, rows in rows_by_file.items()
        },
        "total_rows": sum(len(rows) for rows in rows_by_file.values()),
        "unique_review_ids": len(seen_ids),
        "prompt_changed_ids": sorted(set(prompt_changes)),
        "candidate_token_changed_ids": sorted(set(candidate_token_changes)),
        "problem_count": len(problems),
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate saved MOF review decisions.")
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--flagged-review", type=Path, default=FLAGGED_REVIEW_PATH)
    parser.add_argument("--unflagged-review", type=Path, default=UNFLAGGED_REVIEW_PATH)
    args = parser.parse_args()
    result = validate_review_decisions(
        args.audit,
        args.flagged_review,
        args.unflagged_review,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
