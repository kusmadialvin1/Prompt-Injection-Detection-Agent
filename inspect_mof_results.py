import json
from pathlib import Path
from collections import Counter

folder = Path("logs/mof_baseline_20261023/mof_generate_1000")
accepted_file = folder / "accepted_samples.jsonl"
attempts_file = folder / "all_attempts.jsonl"

samples = [
    json.loads(line)
    for line in accepted_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

print("Accepted records:", len(samples))
print("Unique prompts:", len({s["prompt"] for s in samples}))
print("Label counts:", dict(Counter(s.get("label") for s in samples)))
print("Combination sizes:", dict(Counter(len(s.get("candidate_tokens", [])) for s in samples)))

print("\nFirst 10 accepted samples:")
for i, sample in enumerate(samples[:10], 1):
    print(f'{i}. {sample["prompt"]}')
    print(f'   Tokens: {sample.get("candidate_tokens", [])}')

errors = []
attempt_count = 0
if attempts_file.exists():
    for line in attempts_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        attempt = json.loads(line)
        attempt_count += 1
        if attempt.get("error"):
            errors.append({
                "attempt_index": attempt.get("attempt_index"),
                "error": attempt.get("error")
            })

print("\nTotal attempt records:", attempt_count)
print("Error records:", len(errors))
for error in errors[:10]:
    print(error)
