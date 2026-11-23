"""Instrumented two-step real-PIGuard checkpoint/resume diagnostic."""

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
RECORD_COUNT = 4
MAX_LENGTH = 32
BATCH_SIZE = 1
TOTAL_STEPS = 2
CHECKPOINT_STEP = 1
LEARNING_RATE = 2e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("all", "reference", "staged", "resume"), default="all"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "mof_minimal_checkpoint_diagnostic",
    )
    return parser.parse_args()


def marker_path(directory: Path) -> Path:
    return directory / "progress_markers.jsonl"


def mark(directory: Path, stage: str, event: str, details: dict[str, Any] | None = None) -> None:
    record = {
        "timestamp": time.time(),
        "stage": stage,
        "event": event,
    }
    if details:
        record["details"] = details
    with marker_path(directory).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def set_seed() -> None:
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_records() -> list[dict[str, Any]]:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    selected = records[:RECORD_COUNT]
    if len(selected) != RECORD_COUNT:
        raise ValueError("Insufficient records in the existing training dataset.")
    return selected


def capture_rng() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all(
        [item.cpu() for item in state["torch_cuda_all"]]
    )


def atomic_save(payload: dict[str, Any], path: Path) -> int:
    import torch

    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path.stat().st_size


def digest(value: Any, result: hashlib._Hash | None = None) -> str:
    import torch

    result = result or hashlib.sha256()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            result.update(repr(key).encode())
            digest(value[key], result)
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest(item, result)
    else:
        result.update(repr(value).encode())
    return result.hexdigest()


def configuration() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "dataset": str(DATASET_PATH),
        "record_count": RECORD_COUNT,
        "max_length": MAX_LENGTH,
        "device": "cuda",
        "dtype": "float32",
        "autocast": False,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "optimizer": "Adam(foreach=False)",
        "total_steps": TOTAL_STEPS,
        "checkpoint_step": CHECKPOINT_STEP,
        "gradient_scaler": None,
    }


def train_child(mode: str, directory: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, get_scheduler

    from PIGuard import PIGuard

    marker = lambda stage, event, details=None: mark(directory, stage, event, details)
    marker("cuda_check", "before")
    if not torch.cuda.is_available():
        marker("cuda_check", "failed", {"reason": "CUDA unavailable"})
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device("cuda")
    marker(
        "cuda_check",
        "after",
        {"device": str(device), "name": torch.cuda.get_device_name(device)},
    )

    set_seed()
    records = load_records()
    marker("tokenizer_load", "before")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    marker("tokenizer_load", "after")

    marker("encoding", "before")
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
    data_order = list(range(RECORD_COUNT))
    marker("encoding", "after", {"records": RECORD_COUNT, "sequence_length": MAX_LENGTH})

    marker("model_construction", "before")
    model = PIGuard(MODEL_NAME, num_labels=2, device=device)
    marker("model_construction", "after", {"dtype": "float32"})
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, foreach=False)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=TOTAL_STEPS,
    )
    checkpoint_path = directory / "checkpoint_step_1.pth"
    start_step = 0
    data_position = 0
    checkpoint_bytes: int | None = None

    if mode == "resume":
        marker("fresh_process_load", "before")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["global_step"]
        data_position = checkpoint["data_position"]
        data_order = checkpoint["data_order"]
        restore_rng(checkpoint["rng"])
        marker(
            "fresh_process_load",
            "after",
            {"global_step": start_step, "data_position": data_position},
        )

    target_step = TOTAL_STEPS if mode != "staged" else CHECKPOINT_STEP
    losses: list[float] = []
    learning_rates: list[float] = []
    step_records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)

    for global_step in range(start_step + 1, target_step + 1):
        index = data_order[data_position]
        input_ids = encoded["input_ids"][index : index + 1].to(device)
        attention_mask = encoded["attention_mask"][index : index + 1].to(device)
        batch_labels = labels[index : index + 1].to(device)

        marker("forward", "before", {"global_step": global_step})
        _, loss = model(input_ids, attention_mask, batch_labels, mode="train")
        marker("forward", "after", {"global_step": global_step})

        marker("backward", "before", {"global_step": global_step})
        loss.backward()
        marker("backward", "after", {"global_step": global_step})

        marker("optimizer_step", "before", {"global_step": global_step})
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        torch.cuda.synchronize(device)
        marker("optimizer_step", "after", {"global_step": global_step})

        data_position += 1
        losses.append(float(loss.detach().cpu()))
        learning_rates.append(float(scheduler.get_last_lr()[0]))
        step_records.append(
            {
                "global_step": global_step,
                "loss": losses[-1],
                "learning_rate": learning_rates[-1],
            }
        )

        if mode == "staged" and global_step == CHECKPOINT_STEP:
            marker("checkpoint_save", "before", {"global_step": global_step})
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
                    "rng": capture_rng(),
                    "gradient_scaler": None,
                },
                checkpoint_path,
            )
            marker(
                "checkpoint_save",
                "after",
                {"global_step": global_step, "size_bytes": checkpoint_bytes},
            )

    result = {
        "mode": mode,
        "global_step": target_step,
        "data_position": data_position,
        "losses": losses,
        "learning_rates": learning_rates,
        "step_records": step_records,
        "model_digest": digest(model.state_dict()),
        "optimizer_digest": digest(optimizer.state_dict()),
        "scheduler_digest": digest(scheduler.state_dict()),
        "checkpoint_bytes": checkpoint_bytes
        or (checkpoint_path.stat().st_size if checkpoint_path.exists() else None),
        "peak_cuda_memory": {
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    with (directory / f"{mode}_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    return result


def run_child(mode: str, directory: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            mode,
            "--experiment-dir",
            str(directory),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} process failed with exit code {completed.returncode}: "
            f"{completed.stderr[-3000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def compare(directory: Path) -> dict[str, Any]:
    def read(mode: str) -> dict[str, Any]:
        with (directory / f"{mode}_result.json").open(
            "r", encoding="utf-8"
        ) as handle:
            return json.load(handle)

    reference = read("reference")
    staged = read("staged")
    resumed = read("resume")
    resumed_losses = staged["losses"] + resumed["losses"]
    resumed_rates = staged["learning_rates"] + resumed["learning_rates"]
    return {
        "losses_reference": reference["losses"],
        "losses_resumed": resumed_losses,
        "loss_absolute_differences": [
            abs(left - right)
            for left, right in zip(reference["losses"], resumed_losses)
        ],
        "learning_rates_reference": reference["learning_rates"],
        "learning_rates_resumed": resumed_rates,
        "learning_rate_absolute_differences": [
            abs(left - right)
            for left, right in zip(reference["learning_rates"], resumed_rates)
        ],
        "model_state_equal": reference["model_digest"] == resumed["model_digest"],
        "optimizer_state_equal": (
            reference["optimizer_digest"] == resumed["optimizer_digest"]
        ),
        "scheduler_state_equal": (
            reference["scheduler_digest"] == resumed["scheduler_digest"]
        ),
        "global_step_equal": (
            reference["global_step"] == resumed["global_step"] == TOTAL_STEPS
        ),
        "data_position_equal": reference["data_position"] == resumed["data_position"],
        "checkpoint_bytes": staged["checkpoint_bytes"],
        "peak_cuda_memory_reference": reference["peak_cuda_memory"],
        "peak_cuda_memory_staged": staged["peak_cuda_memory"],
        "peak_cuda_memory_resumed": resumed["peak_cuda_memory"],
    }


def main() -> int:
    args = parse_args()
    directory = args.experiment_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.mode != "all":
        train_child(args.mode, directory)
        return 0

    for path in directory.glob("*_result.json"):
        path.unlink()
    marker_file = marker_path(directory)
    if marker_file.exists():
        marker_file.unlink()

    try:
        run_child("reference", directory)
        run_child("staged", directory)
        run_child("resume", directory)
    except RuntimeError as error:
        failure = {
            "status": "failed",
            "error": str(error),
            "progress_marker_file": str(marker_file),
        }
        with (directory / "diagnostic_failure.json").open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2)
        print(json.dumps(failure, indent=2))
        return 1

    comparison = compare(directory)
    report = {
        "status": "passed",
        "configuration": configuration(),
        "fresh_process_resume": True,
        "comparison": comparison,
        "progress_marker_file": str(marker_file),
    }
    report_path = directory / "minimal_checkpoint_diagnostic_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"report": str(report_path), **comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
