"""Probe every tokenizer vocabulary entry with a trained PIGuard detector.

Paper interpretation
--------------------
Section 4.2 of the paper says that, after standard supervised training, the
authors input every tokenizer-vocabulary token individually into the trained
model and treat tokens predicted as attacks as biased candidates. It does not
state whether "token" means a raw tokenizer string, decoded text, or a direct
token ID. This script therefore uses decoded-text probing as an explicitly
documented engineering interpretation; it does not claim that this choice is
paper-specified.

The script does not train, call an LLM, generate examples, or modify datasets.
It expects the model-only state_dict written by train.py.

Outputs
-------
The output directory is isolated from the repository's evaluation results:

* token_probe.jsonl: one record for every successfully tested vocabulary entry;
* token_probe_failures.jsonl: one record for every entry that could not be
  tested; failures are never represented as benign predictions;
* manifest.json: immutable run identity, configuration, and final counts;
* progress.json: record-derived resume state and recovery information.

Resume policy
-------------
Records, not progress.json, are the source of truth. On resume, the success
and failure JSONL files are validated together: their token IDs must be unique
and form a contiguous prefix of the scheduled token IDs, and the next index is
derived from that combined status ledger. A final unterminated JSONL line is
treated as a crash-truncated record: valid earlier lines are preserved, the
incomplete tail is removed, and the recovery is recorded in progress.json. A
malformed line with later content, a duplicate, an out-of-order record, or an
incompatible manifest is rejected rather than silently rewritten. This
prevents duplicate records after a crash between record fsync and progress
update, including when a token was recorded as failed.

Empty decoded text policy
-------------------------
The default policy is ``preserve``: an empty decoded string is passed as the
classifier input. ``--empty-policy fallback_raw`` may be selected explicitly;
then the raw tokenizer token is passed instead and the record identifies that
fallback. The policy is part of the immutable resume identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import tokenizers
from transformers import AutoTokenizer

from PIGuard import PIGuard


DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_OUTPUT_DIR = Path("logs") / "mof_token_probe"
DEFAULT_SAVE_INTERVAL = 100
CLASSIFY_MAX_LENGTH = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Model-only state_dict produced by train.py.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Probe only this prefix; such a run is explicitly incomplete.",
    )
    parser.add_argument(
        "--empty-policy",
        choices=("preserve", "fallback_raw"),
        default="preserve",
        help="How to handle a vocabulary entry whose decoded text is empty.",
    )
    return parser.parse_args()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def token_text(
    tokenizer: Any,
    token_id: int,
    raw_token: str,
    empty_policy: str,
) -> tuple[str, str, str]:
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded:
        return decoded, decoded, "decoded"
    if empty_policy == "preserve":
        return "", "", "empty_preserved"
    return raw_token, "", "fallback_raw"


def tokenize_classifier_input(tokenizer: Any, input_text: str) -> list[int]:
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=CLASSIFY_MAX_LENGTH,
    )
    return [int(value) for value in encoded["input_ids"][0].tolist()]


def model_config_metadata(model: Any) -> dict[str, Any]:
    return {
        "class": model.__class__.__name__,
        "num_labels": int(model.classifier.out_features),
        "config": model.config.to_dict(),
        "config_sha256": sha256_json(model.config.to_dict()),
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "content") and value.__class__.__name__ == "AddedToken":
        return {
            "__type__": "AddedToken",
            "content": value.content,
            "single_word": bool(value.single_word),
            "lstrip": bool(value.lstrip),
            "rstrip": bool(value.rstrip),
            "normalized": bool(value.normalized),
            "special": bool(getattr(value, "special", False)),
        }
    if hasattr(value, "__dict__"):
        return {
            "__type__": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "attributes": json_safe(vars(value)),
        }
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    ordered_vocabulary = sorted(vocabulary.items(), key=lambda item: item[1])
    vocabulary_hash = sha256_json(ordered_vocabulary)
    special_map = dict(tokenizer.special_tokens_map)
    return {
        "class": tokenizer.__class__.__name__,
        "name_or_path": tokenizer.name_or_path,
        "init_kwargs": json_safe(tokenizer.init_kwargs),
        "special_tokens_map": json_safe(special_map),
        "all_special_ids": [int(value) for value in tokenizer.all_special_ids],
        "vocabulary_size": len(vocabulary),
        "vocabulary_hash": vocabulary_hash,
    }


def environment_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        result.update(
            {
                "cuda_version": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_index": torch.cuda.current_device(),
            }
        )
    return result


def validate_vocab(vocabulary: dict[str, int], config_vocab_size: int) -> dict[str, Any]:
    ids = sorted(int(value) for value in vocabulary.values())
    max_token_id = ids[-1] if ids else None
    expected = list(range(max_token_id + 1)) if max_token_id is not None else []
    unused_count = (
        config_vocab_size - (max_token_id + 1)
        if max_token_id is not None
        else config_vocab_size
    )
    metadata = {
        "tokenizer_vocab_size": len(vocabulary),
        "max_tokenizer_id": max_token_id,
        "model_config_vocab_size": int(config_vocab_size),
        "unused_embedding_rows": {
            "count": unused_count,
            "id_range": (
                [max_token_id + 1, config_vocab_size - 1]
                if unused_count > 0 and max_token_id is not None
                else None
            ),
        },
        "id_min": ids[0] if ids else None,
        "id_max": max_token_id,
        "ids_are_unique": len(ids) == len(set(ids)),
        "ids_are_contiguous_from_zero": ids == expected,
        "max_tokenizer_id_below_model_config_vocab_size": (
            max_token_id is not None and max_token_id < config_vocab_size
        ),
    }
    if (
        not metadata["ids_are_unique"]
        or not metadata["ids_are_contiguous_from_zero"]
        or not metadata["max_tokenizer_id_below_model_config_vocab_size"]
    ):
        raise ValueError(
            "Tokenizer vocabulary IDs must be unique, contiguous from 0, "
            "and below the model config vocabulary size: "
            f"{metadata}"
        )
    return metadata


def read_jsonl_prefix(
    path: Path,
    scheduled_ids: list[int],
    status: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], {"truncated_tail_recovered": False}

    records: list[dict[str, Any]] = []
    truncated_tail_recovered = False
    valid_end = 0
    with path.open("rb") as handle:
        data = handle.read()
    lines = data.splitlines(keepends=True)
    for line_number, raw_line in enumerate(lines, start=1):
        has_newline = raw_line.endswith(b"\n") or raw_line.endswith(b"\r")
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if line_number == len(lines) and not has_newline:
                truncated_tail_recovered = True
                break
            raise ValueError(
                f"Malformed JSONL record at line {line_number}; refusing resume."
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record {line_number} is not an object.")
        if status == "success" and record.get("status") == "failed":
            raise ValueError(f"Success JSONL contains a failed record at line {line_number}.")
        if status == "failed" and record.get("status") != "failed":
            raise ValueError(f"Failure JSONL contains a non-failed record at line {line_number}.")
        records.append(record)
        valid_end += len(raw_line)

    if truncated_tail_recovered:
        with path.open("r+b") as handle:
            handle.truncate(valid_end)

    return records, {"truncated_tail_recovered": truncated_tail_recovered}


def compatibility_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "model_name": manifest["model_name"],
        "model_config_sha256": manifest["model"]["config_sha256"],
        "tokenizer_identity": {
            "name_or_path": manifest["tokenizer"]["name_or_path"],
            "vocabulary_hash": manifest["tokenizer"]["vocabulary_hash"],
            "vocabulary_size": manifest["tokenizer"]["vocabulary_size"],
            "special_tokens_map": manifest["tokenizer"]["special_tokens_map"],
        },
        "input_construction": manifest["inference"]["input_construction"],
        "tokenizer_call": manifest["inference"]["tokenizer_call"],
        "empty_policy": manifest["inference"]["empty_decoding_policy"],
        "max_tokens": manifest["scope"]["max_tokens"],
    }


def reconcile_entries(
    success_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
    scheduled_ids: list[int],
) -> tuple[int, dict[int, str]]:
    statuses: dict[int, str] = {}
    scheduled_set = set(scheduled_ids)
    for status, records in (("success", success_records), ("failed", failure_records)):
        previous_schedule_index = -1
        for record in records:
            token_id = record.get("token_id")
            if not isinstance(token_id, int) or token_id not in scheduled_set:
                raise ValueError(f"Existing {status} record has an invalid token_id.")
            if token_id in statuses:
                raise ValueError(f"Token ID {token_id} appears in multiple output records.")
            schedule_index = scheduled_ids.index(token_id)
            if schedule_index <= previous_schedule_index:
                raise ValueError(
                    f"{status.capitalize()} records are not in scheduled-token order."
                )
            previous_schedule_index = schedule_index
            statuses[token_id] = status

    prefix_length = 0
    while (
        prefix_length < len(scheduled_ids)
        and scheduled_ids[prefix_length] in statuses
    ):
        prefix_length += 1

    expected_prefix = set(scheduled_ids[:prefix_length])
    if set(statuses) != expected_prefix:
        missing_token_id = (
            scheduled_ids[prefix_length]
            if prefix_length < len(scheduled_ids)
            else None
        )
        raise ValueError(
            "Existing records do not form exactly one contiguous prefix: "
            f"first_missing_token_id={missing_token_id}, "
            f"recorded_token_ids_after_gap="
            f"{sorted(set(statuses) - expected_prefix)}"
        )

    return prefix_length, statuses


def main() -> int:
    args = parse_args()
    if args.save_interval < 1:
        raise ValueError("--save-interval must be at least 1.")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1 when provided.")

    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "token_probe.jsonl"
    failures_path = output_dir / "token_probe_failures.jsonl"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "manifest.json"

    set_deterministic_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PIGuard(args.model_name, num_labels=2, device=device)
    # Use the tokenizer owned by PIGuard.classify so the recorded
    # re-tokenization matches the classifier's actual tokenizer instance.
    tokenizer = model.tokenizer
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a model state_dict mapping.")
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint is incompatible with PIGuard strict state-dict validation."
        ) from error
    model.eval()

    vocabulary = tokenizer.get_vocab()
    vocabulary_metadata = validate_vocab(
        vocabulary,
        int(model.config.vocab_size),
    )
    ordered_tokens = sorted(vocabulary.items(), key=lambda item: item[1])
    scheduled_tokens = (
        ordered_tokens[: args.max_tokens]
        if args.max_tokens is not None
        else ordered_tokens
    )
    scheduled_ids = [int(token_id) for _, token_id in scheduled_tokens]

    manifest = {
        "manifest_version": 2,
        "paper_interpretation": {
            "statement": (
                "Section 4.2 specifies inputting every tokenizer-vocabulary "
                "token individually, but does not resolve raw strings, decoded "
                "strings, or direct token IDs."
            ),
            "implemented_interpretation": "decoded_text_then_retokenize",
            "selection_rule": "argmax predicted label == 1",
            "additional_score_threshold": None,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "format": "model-only state_dict from train.py",
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_name": args.model_name,
        "model": model_config_metadata(model),
        "tokenizer": tokenizer_metadata(tokenizer),
        "vocabulary": vocabulary_metadata,
        "inference": {
            "input_construction": (
                "tokenizer.decode([token_id], skip_special_tokens=False, "
                "clean_up_tokenization_spaces=False)"
            ),
            "tokenizer_call": {
                "return_tensors": "pt",
                "truncation": True,
                "padding": "max_length",
                "max_length": CLASSIFY_MAX_LENGTH,
                "add_special_tokens": "tokenizer_default",
            },
            "sequence_length": CLASSIFY_MAX_LENGTH,
            "special_tokens_included": True,
            "empty_decoding_policy": args.empty_policy,
            "inference_batch_size": 1,
            "deterministic": True,
            "seed": args.seed,
            "device": environment_metadata(device),
        },
        "scope": {
            "start_index": 0,
            "max_tokens": args.max_tokens,
            "tokens_scheduled": len(scheduled_tokens),
            "complete_coverage_requested": args.max_tokens is None,
        },
        "outputs": {
            "records_path": str(records_path),
            "failures_path": str(failures_path),
            "progress_path": str(progress_path),
        },
    }
    identity = compatibility_identity(manifest)
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if compatibility_identity(existing_manifest) != identity:
            raise ValueError(
                "Existing output manifest is incompatible with this resume "
                "configuration; use a new output directory."
            )
    else:
        atomic_json_write(manifest_path, manifest)

    records, recovery = read_jsonl_prefix(records_path, scheduled_ids, "success")
    failure_records, failure_recovery = read_jsonl_prefix(
        failures_path,
        scheduled_ids,
        "failed",
    )
    processed_count, _ = reconcile_entries(
        records,
        failure_records,
        scheduled_ids,
    )
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(progress.get("next_index", processed_count)) != processed_count:
            progress = {
                "reconciled": True,
                "previous_progress": progress,
                "next_index": processed_count,
            }
            atomic_json_write(progress_path, progress)

    mode = "a" if records else "w"
    failures = len(failure_records)
    with records_path.open(mode, encoding="utf-8") as record_handle, failures_path.open(
        "a", encoding="utf-8"
    ) as failure_handle:
        for index, (raw_token, token_id) in enumerate(
            scheduled_tokens[processed_count:],
            start=processed_count,
        ):
            try:
                input_text, decoded_text, input_policy = token_text(
                    tokenizer,
                    int(token_id),
                    raw_token,
                    args.empty_policy,
                )
                retokenized_ids = tokenize_classifier_input(tokenizer, input_text)
                with torch.no_grad():
                    logits = model.classify(input_text)
                    probabilities = torch.softmax(logits, dim=-1)[0]
                predicted_label = int(logits.argmax(dim=-1).item())
                record = {
                    "token_id": int(token_id),
                    "token": raw_token,
                    "decoded_text": decoded_text,
                    "input_text": input_text,
                    "input_policy": input_policy,
                    "retokenized_input_ids": retokenized_ids,
                    "original_token_id_preserved": (
                        retokenized_ids.count(int(token_id)) == 1
                        and len(retokenized_ids) == 1
                    ),
                    "is_special_token": bool(
                        int(token_id) in tokenizer.all_special_ids
                    ),
                    "predicted_label": predicted_label,
                    "predicted_class": (
                        "injection" if predicted_label == 1 else "benign"
                    ),
                    "logits": [float(value) for value in logits[0].detach().cpu()],
                    "class_scores": [
                        float(value) for value in probabilities.detach().cpu()
                    ],
                    "selected_as_over_defended": predicted_label == 1,
                }
                write_jsonl(record_handle, record)
            except Exception as error:
                failures += 1
                write_jsonl(
                    failure_handle,
                    {
                        "token_id": int(token_id),
                        "token": raw_token,
                        "index": index,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
            next_index = index + 1
            if next_index % args.save_interval == 0 or next_index == len(scheduled_tokens):
                atomic_json_write(
                    progress_path,
                    {
                        "next_index": next_index,
                        "processed": next_index,
                        "successful_records": next_index - failures,
                        "failed_records": failures,
                        "total": len(scheduled_tokens),
                        "recovery": {
                            "success": recovery,
                            "failure": failure_recovery,
                        },
                    },
                )

    final_records, final_recovery = read_jsonl_prefix(
        records_path,
        scheduled_ids,
        "success",
    )
    final_failure_records, final_failure_recovery = read_jsonl_prefix(
        failures_path,
        scheduled_ids,
        "failed",
    )
    final_processed_count, _ = reconcile_entries(
        final_records,
        final_failure_records,
        scheduled_ids,
    )
    final_count = len(final_records)
    failed_count = len(final_failure_records)
    manifest["completion"] = {
        "successful_records": final_count,
        "failed_records": failed_count,
        "untested_entries": len(scheduled_tokens) - final_processed_count,
        "complete_coverage": (
            args.max_tokens is None
            and final_count == len(scheduled_tokens)
            and failed_count == 0
        ),
        "truncated_tail_recovered": (
            recovery["truncated_tail_recovered"]
            or final_recovery["truncated_tail_recovered"]
            or failure_recovery["truncated_tail_recovered"]
            or final_failure_recovery["truncated_tail_recovered"]
        ),
    }
    atomic_json_write(manifest_path, manifest)
    atomic_json_write(
        progress_path,
        {
            "next_index": final_processed_count,
            "processed": final_processed_count,
            "successful_records": final_count,
            "failed_records": failed_count,
            "total": len(scheduled_tokens),
            "recovery": {
                "success": final_recovery,
                "failure": final_failure_recovery,
            },
        },
    )
    return (
        0
        if manifest["completion"]["complete_coverage"]
        and manifest["completion"]["failed_records"] == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
