"""Run a bounded CPU training smoke test for the PIGuard model.

This script intentionally does not import or launch train.py/eval.py. It uses
the existing PIGuard architecture and JSON dataset format, performs at most
MAX_OPTIMIZER_STEPS optimizer updates, and writes an isolated JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "logs" / "mof_smoke_test"
DEFAULT_TRAIN_DATASET = PROJECT_ROOT / "datasets" / "train.json"
DEFAULT_VALID_DATASET = PROJECT_ROOT / "datasets" / "valid.json"
MODEL_NAME = "microsoft/deberta-v3-base"
SAMPLE_SIZE = 64
VALIDATION_SAMPLE_SIZE = 8
BATCH_SIZE = 2
MAX_LENGTH = 512
MAX_OPTIMIZER_STEPS = 10
LEARNING_RATE = 2e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--valid-dataset", type=Path, default=DEFAULT_VALID_DATASET)
    parser.add_argument(
        "--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_records(path: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Dataset must be a non-empty JSON list: {path}")
    total_records = len(records)
    selected = records if limit is None else records[:limit]
    for index, record in enumerate(selected):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object: {path}")
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"Record {index} has an invalid prompt: {path}")
        if record.get("label") not in (0, 1, False, True):
            raise ValueError(f"Record {index} has an invalid label: {path}")
    return selected, total_records


def label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for record in records:
        counts[str(int(record["label"]))] += 1
    return counts


def run_smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, get_scheduler

    from PIGuard import PIGuard

    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_dir / "smoke_checkpoint.pth"
    report_path = experiment_dir / "smoke_report.json"
    report_started_at = utc_now()

    train_records, train_total = load_records(
        args.train_dataset.resolve(), SAMPLE_SIZE
    )
    valid_records, valid_total = load_records(
        args.valid_dataset.resolve(), VALIDATION_SAMPLE_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(records: list[dict[str, Any]]) -> TensorDataset:
        encoded = tokenizer(
            [record["prompt"] for record in records],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [int(record["label"]) for record in records], dtype=torch.long
        )
        return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)

    train_loader = DataLoader(
        encode(train_records), batch_size=BATCH_SIZE, shuffle=False
    )
    valid_loader = DataLoader(
        encode(valid_records), batch_size=BATCH_SIZE, shuffle=False
    )

    model = PIGuard(MODEL_NAME, num_labels=2, device=device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=MAX_OPTIMIZER_STEPS,
    )

    step_records: list[dict[str, Any]] = []
    tracemalloc.start()
    started_at = time.perf_counter()
    optimizer_steps = 0

    for batch in train_loader:
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        input_ids, attention_mask, labels = (
            tensor.to(device) for tensor in batch
        )
        _, loss = model(input_ids, attention_mask, labels, mode="train")
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer_steps += 1
        step_records.append(
            {
                "step": optimizer_steps,
                "loss": float(loss.detach().cpu()),
                "duration_seconds": time.perf_counter() - step_started,
            }
        )
        if optimizer_steps >= MAX_OPTIMIZER_STEPS:
            break

    training_elapsed = time.perf_counter() - started_at
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    model.eval()
    validation_losses: list[float] = []
    validation_predictions: list[int] = []
    validation_labels: list[int] = []
    with torch.no_grad():
        for batch in valid_loader:
            input_ids, attention_mask, labels = (
                tensor.to(device) for tensor in batch
            )
            logits, loss = model(input_ids, attention_mask, labels, mode="eval")
            validation_losses.append(float(loss.detach().cpu()))
            validation_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            validation_labels.extend(labels.cpu().tolist())

    validation_accuracy = sum(
        prediction == label
        for prediction, label in zip(validation_predictions, validation_labels)
    ) / len(validation_labels)

    torch.save(model.state_dict(), checkpoint_path)
    checkpoint_bytes = checkpoint_path.stat().st_size
    reloaded = PIGuard(MODEL_NAME, num_labels=2, device=device)
    reloaded.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        strict=True,
    )
    del reloaded
    checkpoint_reload_status = "pass"

    report = {
        "started_at": report_started_at,
        "finished_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "experiment_dir": str(experiment_dir),
        "model_name": MODEL_NAME,
        "device": str(device),
        "configuration": {
            "train_sample_size": SAMPLE_SIZE,
            "train_total_records": train_total,
            "validation_sample_size": VALIDATION_SAMPLE_SIZE,
            "validation_total_records": valid_total,
            "train_label_counts": label_counts(train_records),
            "validation_label_counts": label_counts(valid_records),
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "learning_rate": LEARNING_RATE,
        },
        "optimizer_steps": optimizer_steps,
        "step_records": step_records,
        "training_elapsed_seconds": training_elapsed,
        "resource_measurements": {
            "python_tracemalloc_current_bytes": current_memory,
            "python_tracemalloc_peak_bytes": peak_memory,
            "note": "Python allocations only; native PyTorch memory is not included.",
        },
        "validation": {
            "loss_mean": sum(validation_losses) / len(validation_losses),
            "accuracy": validation_accuracy,
            "records_evaluated": len(validation_labels),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_bytes,
            "reload_status": checkpoint_reload_status,
        },
        "full_training_started": False,
        "vocabulary_probe_started": False,
        "llm_called": False,
        "mof_samples_generated": False,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    checkpoint_path.unlink()
    report["checkpoint"]["temporary_file_removed"] = not checkpoint_path.exists()
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> int:
    try:
        report = run_smoke_test(parse_args())
    except Exception as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(
        {
            "report": str(DEFAULT_EXPERIMENT_DIR / "smoke_report.json"),
            "optimizer_steps": report["optimizer_steps"],
            "checkpoint_reload": report["checkpoint"]["reload_status"],
            "temporary_file_removed": report["checkpoint"]["temporary_file_removed"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
