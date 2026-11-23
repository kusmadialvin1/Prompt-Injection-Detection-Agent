import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "logs" / "mof_baseline_20261023" / "mof_generate_1000"
AUDIT_PATH = EXPERIMENT_DIR / "audit_review.csv"
FLAGGED_REVIEW_PATH = EXPERIMENT_DIR / "flagged_manual_review.csv"
UNFLAGGED_REVIEW_PATH = EXPERIMENT_DIR / "unflagged_random_manual_review.csv"
REVIEW_FIELDS = [
    "sample_id",
    "prompt",
    "candidate_tokens",
    "label",
    "flags",
    "manual_review",
]
PROVISIONAL_DECISIONS = {
    7: "keep",
    28: "keep",
    170: "keep",
    296: "keep",
    467: "keep",
    817: "keep",
    871: "keep",
    5: "edit",
    232: "edit",
    317: "edit",
    327: "edit",
    479: "edit",
    863: "edit",
}


def read_audit_rows(path=AUDIT_PATH):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = set(REVIEW_FIELDS)
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Audit CSV must contain columns: {', '.join(REVIEW_FIELDS)}")
    return rows


def sample_id(row):
    try:
        return int(row["sample_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid sample_id: {row.get('sample_id')!r}") from error


def write_review(path, rows, overwrite=False):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Review file already exists: {path}. "
            "Refusing to overwrite manual work; use --overwrite explicitly."
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_reviews(
    audit_path=AUDIT_PATH,
    flagged_path=FLAGGED_REVIEW_PATH,
    unflagged_path=UNFLAGGED_REVIEW_PATH,
    seed=42,
    unflagged_count=50,
    overwrite=False,
):
    rows = read_audit_rows(audit_path)
    flagged = [row for row in rows if row["flags"].strip()]
    unflagged = [row for row in rows if not row["flags"].strip()]
    flagged_ids = {sample_id(row) for row in flagged}
    expected_ids = set(PROVISIONAL_DECISIONS)
    if flagged_ids != expected_ids:
        raise ValueError(
            f"Flagged IDs do not match provisional decision list. "
            f"Expected {sorted(expected_ids)}, found {sorted(flagged_ids)}."
        )
    if len(unflagged) < unflagged_count:
        raise ValueError("Not enough unflagged samples for requested random review.")

    flagged_review = []
    for row in flagged:
        copied = {field: row[field] for field in REVIEW_FIELDS}
        copied["manual_review"] = PROVISIONAL_DECISIONS[sample_id(row)]
        flagged_review.append(copied)

    selected = random.Random(seed).sample(unflagged, unflagged_count)
    random_review = []
    for row in selected:
        copied = {field: row[field] for field in REVIEW_FIELDS}
        copied["manual_review"] = ""
        random_review.append(copied)

    flagged_path.parent.mkdir(parents=True, exist_ok=True)
    write_review(flagged_path, flagged_review, overwrite)
    write_review(unflagged_path, random_review, overwrite)
    return flagged_review, random_review


def main():
    parser = argparse.ArgumentParser(description="Prepare isolated MOF manual-review CSVs.")
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--flagged-output", type=Path, default=FLAGGED_REVIEW_PATH)
    parser.add_argument("--unflagged-output", type=Path, default=UNFLAGGED_REVIEW_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unflagged-count", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    flagged, unflagged = prepare_reviews(
        args.audit,
        args.flagged_output,
        args.unflagged_output,
        args.seed,
        args.unflagged_count,
        args.overwrite,
    )
    print(f"Flagged review rows: {len(flagged)}")
    print(f"Random unflagged review rows: {len(unflagged)}")
    print(f"Flagged review: {args.flagged_output}")
    print(f"Unflagged review: {args.unflagged_output}")


if __name__ == "__main__":
    main()
