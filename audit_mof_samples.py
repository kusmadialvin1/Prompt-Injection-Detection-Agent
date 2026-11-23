import csv
import json
import re
from pathlib import Path

INPUT = Path(
    "logs/mof_baseline_20261023/mof_generate_1000/accepted_samples.jsonl"
)
OUTPUT = Path(
    "logs/mof_baseline_20261023/mof_generate_1000/audit_review.csv"
)

IMPERATIVE_STARTS = {
    "get", "show", "indicate", "provide", "ignore", "follow",
    "execute", "send", "delete", "reveal", "return", "use",
    "make", "write", "run", "click", "open", "enter", "tell",
    "please", "remember", "never", "always", "disregard",
}

SUSPICIOUS_PHRASES = [
    "ignore previous instructions",
    "ignore all instructions",
    "reveal the system prompt",
    "reveal your instructions",
    "disregard previous instructions",
    "bypass security",
    "exfiltrate data",
    "send the password",
    "override your instructions",
]

def audit(sample):
    text = str(sample.get("prompt", "")).strip()
    flags = []
    words = text.split()

    if not text:
        flags.append("empty_text")

    if len(words) < 6:
        flags.append("too_short")

    if len(words) > 120:
        flags.append("too_long")

    if not re.search(r"[.!?][\"']?$", text):
        flags.append("missing_sentence_punctuation")

    if re.search(r"\s{2,}", text):
        flags.append("repeated_whitespace")

    if re.search(r"\bto\s+[a-z]+\s+to\s+ensure", text, re.I):
        flags.append("possible_grammar_error")

    if words and words[0].strip(".,!?\"'()").lower() in IMPERATIVE_STARTS:
        flags.append("possible_instruction")

    lower_text = text.lower()
    for phrase in SUSPICIOUS_PHRASES:
        if phrase in lower_text:
            flags.append("suspicious_instruction_phrase")
            break

    tokens = sample.get("candidate_tokens", [])
    if isinstance(tokens, list):
        for token in tokens:
            if not isinstance(token, str) or not token:
                flags.append("invalid_candidate_token")
                break
            if token not in text:
                flags.append("candidate_token_missing")
                break
    else:
        flags.append("invalid_candidate_token_list")

    if sample.get("label") != 0:
        flags.append("unexpected_label")

    return text, flags


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    samples = []
    with INPUT.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Malformed JSON at line {line_number}: {exc}")

    rows = []
    flagged = 0

    for i, sample in enumerate(samples, 1):
        text, flags = audit(sample)
        if flags:
            flagged += 1

        rows.append({
            "sample_id": i,
            "prompt": text,
            "candidate_tokens": json.dumps(
                sample.get("candidate_tokens", []),
                ensure_ascii=False
            ),
            "label": sample.get("label"),
            "flags": "; ".join(flags),
            "manual_review": "",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "prompt", "candidate_tokens",
            "label", "flags", "manual_review"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Samples loaded: {len(samples)}")
    print(f"Samples flagged for review: {flagged}")
    print(f"Samples without heuristic flags: {len(samples) - flagged}")
    print(f"Review file: {OUTPUT}")


if __name__ == "__main__":
    main()