import csv
from pathlib import Path

path = Path(
    "logs/mof_baseline_20261023/mof_generate_1000/audit_review.csv"
)

with path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

flagged = [row for row in rows if row["flags"].strip()]

print(f"Flagged samples: {len(flagged)}\n")

for row in flagged:
    print(f'ID: {row["sample_id"]}')
    print(f'Text: {row["prompt"]}')
    print(f'Tokens: {row["candidate_tokens"]}')
    print(f'Flags: {row["flags"]}')
    print("-" * 70)
