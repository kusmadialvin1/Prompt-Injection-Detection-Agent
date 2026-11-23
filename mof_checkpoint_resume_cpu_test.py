"""CPU-only checkpoint/resume test using the real PIGuard architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "train.json"
MODEL_NAME = "microsoft/deberta-v3-base"
SEED = 42
SAMPLE_SIZE = 8
MAX_LENGTH = 32
MICRO_BATCH_SIZE = 1
ACCUMULATION_STEPS = 1
TOTAL_STEPS = 6
CHECKPOINT_STEP = 3
LEARNING_RATE = 2e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("all", "uninterrupted", "staged", "resume"), default="all"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "mof_checkpoint_resume_cpu_test",
    )
    return parser.parse_args()


def set_seed() -> None:
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_records() -> list[dict[str, Any]]:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    selected = records[:SAMPLE_SIZE]
    if len(selected) != SAMPLE_SIZE:
        raise ValueError("The requested bounded dataset subset is unavailable.")
    for index, record in enumerate(selected):
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("prompt"), str)
            or not record["prompt"].strip()
            or record.get("label") not in (0, 1, False, True)
        ):
            raise ValueError(f"Invalid training record at index {index}.")
    return selected


def capture_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


def atomic_save(payload: dict[str, Any], path: Path) -> int:
    import torch

    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path.stat().st_size


def state_digest(value: Any, digest: hashlib._Hash | None = None) -> str:
    import torch

    digest = digest or hashlib.sha256()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            digest.update(repr(key).encode())
            state_digest(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            state_digest(item, digest)
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def encode_records(records: list[dict[str, Any]]) -> tuple[Any, Any, Any, list[int]]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
    generator = torch.Generator().manual_seed(SEED)
    data_order = torch.randperm(len(records), generator=generator).tolist()
    return encoded["input_ids"], encoded["attention_mask"], labels, data_order


def configuration() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "dataset": str(DATASET_PATH),
        "sample_size": SAMPLE_SIZE,
        "max_length": MAX_LENGTH,
        "device": "cpu",
        "dtype": "float32",
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_size": MICRO_BATCH_SIZE * ACCUMULATION_STEPS,
        "optimizer": "Adam(foreach=False)",
        "total_steps": TOTAL_STEPS,
        "checkpoint_step": CHECKPOINT_STEP,
        "gradient_scaler": None,
    }


def train_child(mode: str, experiment_dir: Path) -> dict[str, Any]:
    import torch
    from transformers import get_scheduler

    from PIGuard import PIGuard

    set_seed()
    records = load_records()
    input_ids, attention_mask, labels, data_order = encode_records(records)
    device = torch.device("cpu")
    model = PIGuard(MODEL_NAME, num_labels=2, device=device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, foreach=False
    )
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=TOTAL_STEPS,
    )
    checkpoint_path = experiment_dir / "checkpoint_step_3.pth"
    start_step = 0
    data_position = 0
    checkpoint_bytes: int | None = None

    if mode == "resume":
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["global_step"]
        data_position = checkpoint["data_position"]
        data_order = checkpoint["data_order"]
        restore_rng_state(checkpoint["rng"])

    target_step = CHECKPOINT_STEP if mode == "staged" else TOTAL_STEPS
    losses: list[float] = []
    learning_rates: list[float] = []
    step_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for global_step in range(start_step + 1, target_step + 1):
        step_started = time.perf_counter()
        for _ in range(ACCUMULATION_STEPS):
            batch_indices = data_order[
                data_position : data_position + MICRO_BATCH_SIZE
            ]
            if len(batch_indices) != MICRO_BATCH_SIZE:
                raise RuntimeError("The bounded data order ended unexpectedly.")
            index_tensor = torch.tensor(batch_indices, dtype=torch.long)
            _, loss = model(
                input_ids[index_tensor],
                attention_mask[index_tensor],
                labels[index_tensor],
                mode="train",
            )
            (loss / ACCUMULATION_STEPS).backward()
            data_position += MICRO_BATCH_SIZE
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        loss_value = float(loss.detach())
        learning_rate = float(scheduler.get_last_lr()[0])
        losses.append(loss_value)
        learning_rates.append(learning_rate)
        step_records.append(
            {
                "global_step": global_step,
                "loss": loss_value,
                "learning_rate": learning_rate,
                "duration_seconds": time.perf_counter() - step_started,
            }
        )
        if mode == "staged" and global_step == CHECKPOINT_STEP:
            checkpoint_bytes = atomic_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": 0,
                    "global_step": global_step,
                    "data_position": data_position,
                    "data_order": data_order,
                    "configuration": configuration(),
                    "rng": capture_rng_state(),
                    "gradient_scaler": None,
                },
                checkpoint_path,
            )

    result = {
        "mode": mode,
        "start_step": start_step,
        "global_step": target_step,
        "data_position": data_position,
        "losses": losses,
        "learning_rates": learning_rates,
        "step_records": step_records,
        "elapsed_seconds": time.perf_counter() - started,
        "model_digest": state_digest(model.state_dict()),
        "optimizer_digest": state_digest(optimizer.state_dict()),
        "scheduler_digest": state_digest(scheduler.state_dict()),
        "checkpoint_bytes": checkpoint_bytes
        if checkpoint_bytes is not None
        else (checkpoint_path.stat().st_size if checkpoint_path.exists() else None),
    }
    result_path = experiment_dir / f"{mode}_result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    return result


def run_child(mode: str, experiment_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--experiment-dir",
        str(experiment_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} process failed with exit code {completed.returncode}:\n"
            f"{completed.stderr[-4000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def compare(experiment_dir: Path) -> dict[str, Any]:
    def read(mode: str) -> dict[str, Any]:
        with (experiment_dir / f"{mode}_result.json").open(
            "r", encoding="utf-8"
        ) as handle:
            return json.load(handle)

    uninterrupted = read("uninterrupted")
    staged = read("staged")
    resumed = read("resume")
    resumed_losses = staged["losses"] + resumed["losses"]
    resumed_rates = staged["learning_rates"] + resumed["learning_rates"]
    return {
        "losses_uninterrupted": uninterrupted["losses"],
        "losses_resumed": resumed_losses,
        "loss_absolute_differences": [
            abs(left - right)
            for left, right in zip(uninterrupted["losses"], resumed_losses)
        ],
        "learning_rates_uninterrupted": uninterrupted["learning_rates"],
        "learning_rates_resumed": resumed_rates,
        "learning_rate_absolute_differences": [
            abs(left - right)
            for left, right in zip(uninterrupted["learning_rates"], resumed_rates)
        ],
        "model_parameters_equal": (
            uninterrupted["model_digest"] == resumed["model_digest"]
        ),
        "optimizer_state_equal": (
            uninterrupted["optimizer_digest"] == resumed["optimizer_digest"]
        ),
        "scheduler_state_equal": (
            uninterrupted["scheduler_digest"] == resumed["scheduler_digest"]
        ),
        "global_step_equal": (
            uninterrupted["global_step"] == resumed["global_step"] == TOTAL_STEPS
        ),
        "checkpoint_bytes": staged["checkpoint_bytes"],
        "uninterrupted_elapsed_seconds": uninterrupted["elapsed_seconds"],
        "staged_elapsed_seconds": staged["elapsed_seconds"],
        "resumed_elapsed_seconds": resumed["elapsed_seconds"],
        "limitations": [
            "This is a bounded CPU test with eight real training records and sequence length 32.",
            "It verifies checkpoint mechanics for the real PIGuard architecture, not full-training performance.",
            "Exact equality assumes the same PyTorch, Transformers, CPU, and deterministic settings.",
        ],
    }


def main() -> int:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.mode != "all":
        train_child(args.mode, experiment_dir)
        return 0

    for path in experiment_dir.glob("*_result.json"):
        path.unlink()
    uninterrupted = run_child("uninterrupted", experiment_dir)
    staged = run_child("staged", experiment_dir)
    resumed = run_child("resume", experiment_dir)
    comparison = compare(experiment_dir)
    report = {
        "configuration": configuration(),
        "fresh_process_resume": True,
        "uninterrupted": uninterrupted,
        "staged": staged,
        "resumed": resumed,
        "comparison": comparison,
    }
    report_path = experiment_dir / "cpu_checkpoint_resume_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"report": str(report_path), **comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
