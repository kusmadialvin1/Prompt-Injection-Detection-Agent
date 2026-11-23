"""Run a bounded CUDA training smoke test for the PIGuard model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "logs" / "mof_gpu_smoke_test"
DEFAULT_TRAIN_DATASET = PROJECT_ROOT / "datasets" / "train.json"
DEFAULT_VALID_DATASET = PROJECT_ROOT / "datasets" / "valid.json"
MODEL_NAME = "microsoft/deberta-v3-base"
TRAIN_SAMPLE_SIZE = 64
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
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(2, 4, 8),
        default=BATCH_SIZE,
        help="CUDA smoke-test batch size; test 4 and 8 only as separate runs.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_records(path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Dataset must be a non-empty JSON list: {path}")
    selected = records[:limit]
    for index, record in enumerate(selected):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not an object: {path}")
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"Record {index} has an invalid prompt: {path}")
        if record.get("label") not in (0, 1, False, True):
            raise ValueError(f"Record {index} has an invalid label: {path}")
    return selected, len(records)


def label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for record in records:
        counts[str(int(record["label"]))] += 1
    return counts


def cuda_memory() -> dict[str, int]:
    import torch

    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def run_smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, get_scheduler

    from PIGuard import PIGuard

    report_started_at = utc_now()
    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report_path = experiment_dir / "gpu_smoke_report.json"
    checkpoint_path = experiment_dir / "gpu_smoke_checkpoint.pth"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run a GPU smoke test.")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    device_name = torch.cuda.get_device_name(device)
    train_records, train_total = load_records(
        args.train_dataset.resolve(), TRAIN_SAMPLE_SIZE
    )
    valid_records, valid_total = load_records(
        args.valid_dataset.resolve(), VALIDATION_SAMPLE_SIZE
    )
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
        encode(train_records), batch_size=args.batch_size, shuffle=False
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
    optimizer_steps = 0
    training_started = time.perf_counter()
    oom_error: str | None = None
    try:
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
            torch.cuda.synchronize(device)
            step_records.append(
                {
                    "step": optimizer_steps,
                    "loss": float(loss.detach().cpu()),
                    "duration_seconds": time.perf_counter() - step_started,
                    "memory": cuda_memory(),
                }
            )
            if optimizer_steps >= MAX_OPTIMIZER_STEPS:
                break
    except torch.cuda.OutOfMemoryError as error:
        oom_error = str(error)
        torch.cuda.empty_cache()
    training_elapsed = time.perf_counter() - training_started

    validation_accuracy: float | None = None
    validation_loss: float | None = None
    validation_records_evaluated = 0
    if oom_error is None:
        model.eval()
        validation_losses: list[float] = []
        predictions: list[int] = []
        labels_seen: list[int] = []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attention_mask, labels = (
                    tensor.to(device) for tensor in batch
                )
                logits, loss = model(input_ids, attention_mask, labels, mode="eval")
                validation_losses.append(float(loss.detach().cpu()))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                labels_seen.extend(labels.cpu().tolist())
        validation_records_evaluated = len(labels_seen)
        validation_loss = sum(validation_losses) / len(validation_losses)
        validation_accuracy = sum(
            prediction == label
            for prediction, label in zip(predictions, labels_seen)
        ) / len(labels_seen)

    checkpoint_reload_status = "not_attempted"
    checkpoint_bytes: int | None = None
    temporary_file_removed = False
    if oom_error is None:
        torch.cuda.synchronize(device)
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
        reloaded = PIGuard(MODEL_NAME, num_labels=2, device=device)
        reloaded.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True),
            strict=True,
        )
        del reloaded
        checkpoint_reload_status = "pass"
        checkpoint_path.unlink()
        temporary_file_removed = not checkpoint_path.exists()

    report = {
        "started_at": report_started_at,
        "finished_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "experiment_dir": str(experiment_dir),
        "model_name": MODEL_NAME,
        "device": str(device),
        "cuda_used": True,
        "cuda_device_name": device_name,
        "configuration": {
            "train_sample_size": TRAIN_SAMPLE_SIZE,
            "train_total_records": train_total,
            "validation_sample_size": VALIDATION_SAMPLE_SIZE,
            "validation_total_records": valid_total,
            "train_label_counts": label_counts(train_records),
            "validation_label_counts": label_counts(valid_records),
            "batch_size": args.batch_size,
            "max_length": MAX_LENGTH,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "learning_rate": LEARNING_RATE,
        },
        "optimizer_steps": optimizer_steps,
        "step_records": step_records,
        "training_elapsed_seconds": training_elapsed,
        "peak_gpu_memory": cuda_memory(),
        "out_of_memory": oom_error is not None,
        "out_of_memory_error": oom_error,
        "validation": {
            "loss_mean": validation_loss,
            "accuracy": validation_accuracy,
            "records_evaluated": validation_records_evaluated,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_bytes,
            "reload_status": checkpoint_reload_status,
            "temporary_file_removed": temporary_file_removed,
        },
        "full_training_started": False,
        "vocabulary_probe_started": False,
        "llm_called": False,
        "mof_samples_generated": False,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> int:
    try:
        report = run_smoke_test(parse_args())
    except Exception as error:
        print(f"GPU smoke test failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report": str(
                    report["experiment_dir"]
                    + "\\gpu_smoke_report.json"
                    if isinstance(report["experiment_dir"], str)
                    else report["experiment_dir"] / "gpu_smoke_report.json"
                ),
                "cuda_used": report["cuda_used"],
                "optimizer_steps": report["optimizer_steps"],
                "out_of_memory": report["out_of_memory"],
                "checkpoint_reload": report["checkpoint"]["reload_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
