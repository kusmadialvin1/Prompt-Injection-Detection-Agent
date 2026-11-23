"""Validate checkpoint/resume mechanics with a tiny CPU synthetic classifier."""

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
SEED = 42
TOTAL_STEPS = 6
CHECKPOINT_STEP = 3
BATCH_SIZE = 2
ACCUMULATION_STEPS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("all", "uninterrupted", "staged", "resume"), default="all"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "mof_checkpoint_resume_cpu_synthetic_test",
    )
    return parser.parse_args()


def seed() -> None:
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def digest(value: Any, result: hashlib._Hash | None = None) -> str:
    import torch

    result = result or hashlib.sha256()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.numpy().tobytes())
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


def train_child(mode: str, experiment_dir: Path) -> dict[str, Any]:
    import torch
    from torch import nn
    from transformers import get_scheduler

    seed()
    features = torch.arange(48, dtype=torch.float32).reshape(24, 2) / 24
    labels = torch.tensor([0, 1] * 12, dtype=torch.long)
    data_order = list(range(len(labels)))
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=TOTAL_STEPS,
    )
    checkpoint_path = experiment_dir / "synthetic_checkpoint_step_3.pth"
    start_step = 0
    data_position = 0
    checkpoint_bytes = None

    if mode == "resume":
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["global_step"]
        data_position = checkpoint["data_position"]
        data_order = checkpoint["data_order"]
        restore_rng(checkpoint["rng"])

    target_step = CHECKPOINT_STEP if mode == "staged" else TOTAL_STEPS
    losses = []
    rates = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for global_step in range(start_step + 1, target_step + 1):
        for _ in range(ACCUMULATION_STEPS):
            indices = data_order[data_position : data_position + BATCH_SIZE]
            index_tensor = torch.tensor(indices, dtype=torch.long)
            logits = model(features[index_tensor])
            loss = nn.functional.cross_entropy(
                logits, labels[index_tensor]
            )
            (loss / ACCUMULATION_STEPS).backward()
            data_position += BATCH_SIZE
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        losses.append(float(loss))
        rates.append(float(scheduler.get_last_lr()[0]))
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
                    "configuration": {
                        "synthetic": True,
                        "dtype": "float32",
                        "batch_size": BATCH_SIZE,
                        "gradient_accumulation_steps": ACCUMULATION_STEPS,
                    },
                    "rng": rng_state(),
                    "gradient_scaler": None,
                },
                checkpoint_path,
            )
    result = {
        "mode": mode,
        "global_step": target_step,
        "losses": losses,
        "learning_rates": rates,
        "model_digest": digest(model.state_dict()),
        "optimizer_digest": digest(optimizer.state_dict()),
        "scheduler_digest": digest(scheduler.state_dict()),
        "checkpoint_bytes": checkpoint_bytes
        or (checkpoint_path.stat().st_size if checkpoint_path.exists() else None),
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (experiment_dir / f"{mode}_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    return result


def child(mode: str, directory: Path) -> dict[str, Any]:
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
            f"{mode} child failed with exit code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    args = parse_args()
    directory = args.experiment_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.mode != "all":
        train_child(args.mode, directory)
        return 0
    for path in directory.glob("*_result.json"):
        path.unlink()
    uninterrupted = child("uninterrupted", directory)
    staged = child("staged", directory)
    resumed = child("resume", directory)
    resumed_losses = staged["losses"] + resumed["losses"]
    resumed_rates = staged["learning_rates"] + resumed["learning_rates"]
    comparison = {
        "losses_uninterrupted": uninterrupted["losses"],
        "losses_resumed": resumed_losses,
        "loss_absolute_differences": [
            abs(a - b)
            for a, b in zip(uninterrupted["losses"], resumed_losses)
        ],
        "learning_rates_uninterrupted": uninterrupted["learning_rates"],
        "learning_rates_resumed": resumed_rates,
        "learning_rate_absolute_differences": [
            abs(a - b)
            for a, b in zip(uninterrupted["learning_rates"], resumed_rates)
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
        "global_step_equal": uninterrupted["global_step"]
        == resumed["global_step"]
        == TOTAL_STEPS,
        "checkpoint_bytes": staged["checkpoint_bytes"],
        "fresh_process_resume": True,
    }
    report = {
        "test_type": "synthetic_checkpoint_mechanics_only",
        "configuration": {
            "dtype": "float32",
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "total_steps": TOTAL_STEPS,
            "checkpoint_step": CHECKPOINT_STEP,
        },
        "comparison": comparison,
        "limitations": [
            "This does not validate the real PIGuard DeBERTa model.",
            "The real CPU test crashed with Windows access violation 0xC0000005 before completion.",
        ],
    }
    report_path = directory / "synthetic_checkpoint_resume_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"report": str(report_path), **comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
