import argparse
import csv
import hashlib
import json
from pathlib import Path

from prepare_mof_manual_review import (
    AUDIT_PATH,
    FLAGGED_REVIEW_PATH,
    PROVISIONAL_DECISIONS,
    REVIEW_FIELDS,
    UNFLAGGED_REVIEW_PATH,
    read_audit_rows,
    sample_id,
)


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "datasets" / "train.json"
# This is the baseline fingerprint captured before creating the review files.
BASELINE_TRAIN_SHA256 = "656ffff8c99d1dbf18a9f34a6a0ca7e88df946a40586c783f9a166309a891f3a"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_review(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Review CSV is empty: {path}")
    if not set(REVIEW_FIELDS).issubset(rows[0]):
        raise ValueError(f"Review CSV has missing required columns: {path}")
    return rows


def validate_reviews(
    audit_path=AUDIT_PATH,
    flagged_path=FLAGGED_REVIEW_PATH,
    unflagged_path=UNFLAGGED_REVIEW_PATH,
    train_path=TRAIN_PATH,
    expected_train_sha256=BASELINE_TRAIN_SHA256,
):
    audit_rows = read_audit_rows(audit_path)
    audit_by_id = {sample_id(row): row for row in audit_rows}
    if len(audit_by_id) != len(audit_rows):
        raise ValueError("Original audit report contains duplicate sample IDs.")

    flagged_rows = read_review(flagged_path)
    unflagged_rows = read_review(unflagged_path)
    flagged_ids = [sample_id(row) for row in flagged_rows]
    unflagged_ids = [sample_id(row) for row in unflagged_rows]
    if len(flagged_rows) != 13 or set(flagged_ids) != set(PROVISIONAL_DECISIONS):
        raise ValueError("Flagged review must contain exactly the 13 expected flagged IDs.")
    if len(unflagged_rows) != 50 or len(set(unflagged_ids)) != 50:
        raise ValueError("Random review must contain exactly 50 unique IDs.")
    if set(flagged_ids) & set(unflagged_ids):
        raise ValueError("A sample appears in both review files.")

    for row in flagged_rows + unflagged_rows:
        identifier = sample_id(row)
        if identifier not in audit_by_id:
            raise ValueError(f"Review sample ID is absent from audit report: {identifier}")
        source = audit_by_id[identifier]
        for field in ("prompt", "candidate_tokens", "label", "flags"):
            if row[field] != source[field]:
                raise ValueError(f"Field {field!r} changed for sample {identifier}.")
    for row in flagged_rows:
        if row["manual_review"] != PROVISIONAL_DECISIONS[sample_id(row)]:
            raise ValueError(f"Unexpected provisional decision for sample {sample_id(row)}.")
    for row in unflagged_rows:
        if row["manual_review"] != "":
            raise ValueError(f"Unflagged sample {sample_id(row)} must remain blank.")

    actual_hash = sha256_file(train_path)
    if actual_hash.lower() != expected_train_sha256.lower():
        raise ValueError(
            "Training dataset hash differs from the recorded baseline. "
            f"expected={expected_train_sha256}, actual={actual_hash}"
        )
    return {
        "audit_rows": len(audit_rows),
        "flagged_rows": len(flagged_rows),
        "unflagged_rows": len(unflagged_rows),
        "train_sha256": actual_hash,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description="Validate MOF manual-review CSV integrity.")
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--flagged-review", type=Path, default=FLAGGED_REVIEW_PATH)
    parser.add_argument("--unflagged-review", type=Path, default=UNFLAGGED_REVIEW_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    args = parser.parse_args()
    result = validate_reviews(args.audit, args.flagged_review, args.unflagged_review, args.train)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
