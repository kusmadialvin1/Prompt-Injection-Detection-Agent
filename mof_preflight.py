"""Small, isolated preflight checks for the PIGuard MOF experiment.

This script does not train the model, probe the vocabulary, call an LLM, or
modify the original training and evaluation modules.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "logs" / "mof_preflight_20261022"
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "train.json"
MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_SAMPLE_SIZE = 32
DEFAULT_BATCH_SIZE = 2
SEQUENCE_LENGTHS = (32, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Training JSON file to sample for format and label checks.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Directory for the JSON report and temporary files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of training records to inspect.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Small batch size used for model probes.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def inspect_dataset(dataset_path: Path, sample_size: int) -> dict[str, Any]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError("Training dataset must be a JSON list.")
    if not records:
        raise ValueError("Training dataset is empty.")

    selected = records[:sample_size]
    required_fields = {"prompt", "label"}
    missing_fields = [
        sorted(required_fields - set(record))
        for record in selected
        if not isinstance(record, dict)
    ]
    if missing_fields:
        raise ValueError("Each sampled record must be a JSON object.")

    invalid_labels = [
        record["label"]
        for record in selected
        if record.get("label") not in (0, 1, False, True)
    ]
    invalid_prompts = [
        index
        for index, record in enumerate(selected)
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip()
    ]
    if invalid_labels:
        raise ValueError(f"Invalid labels in sample: {invalid_labels[:5]}")
    if invalid_prompts:
        raise ValueError(f"Invalid prompts at sampled indexes: {invalid_prompts}")

    label_counts = {"0": 0, "1": 0}
    for record in selected:
        label_counts[str(int(record["label"]))] += 1

    return {
        "path": str(dataset_path),
        "total_records": len(records),
        "sample_size": len(selected),
        "required_fields": sorted(required_fields),
        "label_counts_in_sample": label_counts,
        "status": "pass",
    }


def run_probe_child(sequence_length: int, batch_size: int) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from PIGuard import PIGuard

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(
        ["MOF preflight prompt"] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    labels = torch.zeros(batch_size, dtype=torch.long)
    model = PIGuard(MODEL_NAME, num_labels=2, device=torch.device("cpu"))
    model.train()

    started = time.perf_counter()
    logits, loss = model(encoded["input_ids"], encoded["attention_mask"], labels)
    loss.backward()
    elapsed = time.perf_counter() - started

    return {
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "finite_loss": bool(torch.isfinite(loss.detach()).item()),
        "status": "pass",
    }


def run_isolated_probe(
    sequence_length: int,
    batch_size: int,
    script_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script_path),
        "--child-probe",
        str(sequence_length),
        "--child-batch-size",
        str(batch_size),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    result: dict[str, Any] = {
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "exit_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }
    if completed.returncode != 0:
        result["status"] = "fail"
        result["error"] = "Probe child exited unsuccessfully."
        return result

    try:
        child_result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        result["status"] = "fail"
        result["error"] = f"Could not parse child result: {error}"
        return result

    result.update(child_result)
    return result


def checkpoint_round_trip(experiment_dir: Path) -> dict[str, Any]:
    import torch
    from PIGuard import PIGuard

    checkpoint_path: Path | None = None
    try:
        model = PIGuard(MODEL_NAME, num_labels=2, device=torch.device("cpu"))
        with tempfile.NamedTemporaryFile(
            prefix="checkpoint_",
            suffix=".pth",
            dir=experiment_dir,
            delete=False,
        ) as handle:
            checkpoint_path = Path(handle.name)

        torch.save(model.state_dict(), checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reloaded = PIGuard(MODEL_NAME, num_labels=2, device=torch.device("cpu"))
        reloaded.load_state_dict(state, strict=True)
        return {
            "status": "pass",
            "checkpoint_bytes": checkpoint_path.stat().st_size,
        }
    finally:
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()


def run_parent_checks(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers

    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "started_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "experiment_dir": str(experiment_dir),
        "model_name": MODEL_NAME,
        "full_training_started": False,
        "llm_called": False,
        "vocabulary_probe_started": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "torch_threads": torch.get_num_threads(),
        },
    }

    try:
        report["dataset"] = inspect_dataset(
            args.dataset.resolve(),
            args.sample_size,
        )
    except Exception as error:
        report["dataset"] = {"status": "fail", "error": str(error)}

    tokenizer_result: dict[str, Any]
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer_result = {
            "status": "pass",
            "vocab_size": len(tokenizer),
            "model_max_length": tokenizer.model_max_length,
        }
    except Exception as error:
        tokenizer_result = {"status": "fail", "error": str(error)}
    report["tokenizer"] = tokenizer_result

    report["training_probes"] = [
        run_isolated_probe(
            sequence_length,
            args.batch_size,
            Path(__file__).resolve(),
        )
        for sequence_length in SEQUENCE_LENGTHS
    ]

    try:
        report["checkpoint_round_trip"] = checkpoint_round_trip(experiment_dir)
    except Exception as error:
        report["checkpoint_round_trip"] = {
            "status": "fail",
            "error": str(error),
        }

    report["finished_at"] = utc_now()
    report["summary"] = {
        "dataset": report["dataset"].get("status"),
        "tokenizer": report["tokenizer"].get("status"),
        "training_probe_statuses": [
            probe.get("status") for probe in report["training_probes"]
        ],
        "checkpoint_round_trip": report["checkpoint_round_trip"].get("status"),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child-probe", type=int)
    parser.add_argument("--child-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    child_args, _ = parser.parse_known_args()
    if child_args.child_probe is not None:
        result = run_probe_child(
            child_args.child_probe,
            child_args.child_batch_size,
        )
        print(json.dumps(result))
        return 0

    args = parse_args()
    try:
        report = run_parent_checks(args)
    except Exception as error:
        report = {
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "status": "fail",
            "error": str(error),
            "full_training_started": False,
        }

    report_path = args.experiment_dir.resolve() / "preflight_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report.get("status", "pass") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
