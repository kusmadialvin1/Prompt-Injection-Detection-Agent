# MOF manual review

The review files are derived from `logs/mof_baseline_20261023/mof_generate_1000/audit_review.csv`.
The original audit CSV, generated JSONL, training data, and checkpoints are not edited.

## Create or recreate the review files

From PowerShell:

```powershell
Set-Location -LiteralPath 'D:\新建文件夹\School\毕业设计\PIGuard'
.\.venv\Scripts\python.exe .\prepare_mof_manual_review.py
```

The command refuses to overwrite existing review CSVs. To deliberately recreate
them (which discards edits in those two review files), use:

```powershell
.\.venv\Scripts\python.exe .\prepare_mof_manual_review.py --overwrite
```

It creates:

- `logs/mof_baseline_20261023/mof_generate_1000/flagged_manual_review.csv`
- `logs/mof_baseline_20261023/mof_generate_1000/unflagged_random_manual_review.csv`

## Review in Excel

Open either CSV in Excel and edit only the `manual_review` column. The flagged
file contains provisional `keep` and `edit` values; these are not final decisions.
The random unflagged file intentionally starts with blank decisions. Do not
rewrite `prompt`, `candidate_tokens`, `label`, or `flags`.

Save the CSV as CSV UTF-8. Do not merge the reviewed rows into
`datasets/train.json` with this workflow.

## Validate integrity

```powershell
.\.venv\Scripts\python.exe .\validate_mof_manual_review.py
```

Validation checks the expected 13 flagged rows, 50 unique unflagged rows,
source-field preservation, no overlap, and the SHA-256 fingerprint of
`datasets/train.json`.

The recorded hash is a baseline captured during setup. A matching hash proves
the file has not changed since that baseline was captured; it cannot prove that
the file matched an earlier external or paper release before the baseline was
recorded.
