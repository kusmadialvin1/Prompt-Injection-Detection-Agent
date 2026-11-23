"""Bounded checkpoint/resume fidelity test for the PIGuard training setup."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "logs" / "mof_checkpoint_resume_test"
TRAIN_DATASET = PROJECT_ROOT / "datasets" / "train.json"
MODEL_NAME = "microsoft/deberta-v3-base"
SEED = 42
SAMPLE_SIZE = 64
MICRO_BATCH_SIZE = 2
ACCUMULATION_STEPS = 2
MAX_LENGTH = 512
TOTAL_OPTIMIZER_STEPS = 6
CHECKPOINT_STEP = 3
LEARNING_RATE = 2e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "uninterrupted", "staged", "resume"),
        default="all",
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    return parser.parse_args()


def load_records() -> list[dict[str, Any]]:
    with TRAIN_DATASET.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    selected = records[:SAMPLE_SIZE]
    if not isinstance(selected, list) or len(selected) < SAMPLE_SIZE:
        raise ValueError("The bounded training subset is unavailable.")
    for index, record in enumerate(selected):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not an object.")
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"Record {index} has an invalid prompt.")
        if record.get("label") not in (0, 1, False, True):
            raise ValueError(f"Record {index} has an invalid label.")
    return selected


def atomic_torch_save(payload: dict[str, Any], path: Path) -> int:
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


def capture_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all(
        [cuda_state.cpu() for cuda_state in state["torch_cuda_all"]]
    )


def set_seed() -> None:
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_encoded_data(records: list[dict[str, Any]]) -> tuple[Any, Any, Any, list[int]]:
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
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    order = torch.randperm(len(records), generator=generator).tolist()
    return encoded["input_ids"], encoded["attention_mask"], labels, order


def make_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    global_step: int,
    micro_position: int,
    data_order: list[int],
    configuration: dict[str, Any],
    path: Path,
) -> int:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "micro_position": micro_position,
        "data_order": data_order,
        "configuration": configuration,
        "rng": capture_rng_state(),
        "scaler": None,
    }
    return atomic_torch_save(payload, path)


def tensor_state_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return (
            left.keys() == right.keys()
            and all(tensor_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            tensor_state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def train_process(mode: str, experiment_dir: Path) -> dict[str, Any]:
    import torch
    from transformers import get_scheduler

    from PIGuard import PIGuard

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device("cuda")
    set_seed()
    records = load_records()
    input_ids, attention_mask, labels, data_order = build_encoded_data(records)
    model = PIGuard(MODEL_NAME, num_labels=2, device=torch.device("cpu"))
    model.deberta.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.half().to(device)
    model.device = device
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, foreach=False
    )
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=TOTAL_OPTIMIZER_STEPS,
    )
    checkpoint_path = experiment_dir / "checkpoint_step_3.pth"
    start_step = 0
    micro_position = 0
    if mode == "resume":
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["global_step"]
        micro_position = checkpoint["micro_position"]
        data_order = checkpoint["data_order"]
        restore_rng_state(checkpoint["rng"])

    target_step = TOTAL_OPTIMIZER_STEPS if mode != "staged" else CHECKPOINT_STEP
    losses: list[float] = []
    learning_rates: list[float] = []
    step_records: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for global_step in range(start_step + 1, target_step + 1):
        step_started = time.perf_counter()
        for accumulation_index in range(ACCUMULATION_STEPS):
            batch_start = micro_position * MICRO_BATCH_SIZE
            batch_indices = data_order[batch_start : batch_start + MICRO_BATCH_SIZE]
            if len(batch_indices) != MICRO_BATCH_SIZE:
                raise RuntimeError("Bounded data order ended before six steps.")
            batch_index_tensor = torch.tensor(batch_indices, dtype=torch.long)
            batch_input_ids = input_ids[batch_index_tensor].to(device)
            batch_attention_mask = attention_mask[batch_index_tensor].to(device)
            batch_labels = labels[batch_index_tensor].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss = model(
                    batch_input_ids,
                    batch_attention_mask,
                    batch_labels,
                    mode="train",
                )
                divided_loss = loss / ACCUMULATION_STEPS
            divided_loss.backward()
            micro_position += 1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        torch.cuda.synchronize(device)
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        learning_rates.append(float(scheduler.get_last_lr()[0]))
        step_records.append(
            {
                "global_step": global_step,
                "loss": loss_value,
                "learning_rate": learning_rates[-1],
                "duration_seconds": time.perf_counter() - step_started,
            }
        )
        if global_step == CHECKPOINT_STEP and mode == "staged":
            checkpoint_bytes = make_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=0,
                global_step=global_step,
                micro_position=micro_position,
                data_order=data_order,
                configuration=configuration(),
                path=checkpoint_path,
            )
    elapsed = time.perf_counter() - started
    result = {
        "mode": mode,
        "optimizer_steps": target_step - start_step,
        "start_step": start_step,
        "global_step": target_step,
        "micro_position": micro_position,
        "losses": losses,
        "learning_rates": learning_rates,
        "step_records": step_records,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory": {
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "checkpoint_bytes": (
            checkpoint_path.stat().st_size if checkpoint_path.exists() else None
        ),
    }
    result_path = experiment_dir / f"{mode}_result.pt"
    torch.save(result, result_path)
    return {
        "mode": mode,
        "result_path": str(result_path),
        "checkpoint_bytes": result["checkpoint_bytes"],
        "global_step": result["global_step"],
    }


def configuration() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_size": MICRO_BATCH_SIZE * ACCUMULATION_STEPS,
        "max_length": MAX_LENGTH,
        "precision": "float16 model weights and gradients with float16 autocast",
        "gradient_checkpointing": True,
        "optimizer": "Adam(foreach=False)",
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "checkpoint_step": CHECKPOINT_STEP,
        "scaler": None,
    }


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
            f"{mode} child failed with exit code {completed.returncode}:\n"
            f"{completed.stderr[-4000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def compare_runs(experiment_dir: Path) -> dict[str, Any]:
    import torch

    uninterrupted = torch.load(
        experiment_dir / "uninterrupted_result.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed = torch.load(
        experiment_dir / "resume_result.pt",
        map_location="cpu",
        weights_only=False,
    )
    staged = torch.load(
        experiment_dir / "staged_result.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_equal = tensor_state_equal(
        uninterrupted["model_state"], resumed["model_state"]
    )
    optimizer_equal = tensor_state_equal(
        uninterrupted["optimizer_state"], resumed["optimizer_state"]
    )
    scheduler_equal = tensor_state_equal(
        uninterrupted["scheduler_state"], resumed["scheduler_state"]
    )
    resumed_losses = staged["losses"] + resumed["losses"]
    resumed_learning_rates = (
        staged["learning_rates"] + resumed["learning_rates"]
    )
    loss_differences = [
        abs(a - b) for a, b in zip(uninterrupted["losses"], resumed_losses)
    ]
    lr_differences = [
        abs(a - b)
        for a, b in zip(uninterrupted["learning_rates"], resumed_learning_rates)
    ]
    return {
        "losses_uninterrupted": uninterrupted["losses"],
        "losses_resumed": resumed_losses,
        "loss_absolute_differences": loss_differences,
        "learning_rates_uninterrupted": uninterrupted["learning_rates"],
        "learning_rates_resumed": resumed_learning_rates,
        "learning_rate_absolute_differences": lr_differences,
        "model_state_equal": model_equal,
        "optimizer_state_equal": optimizer_equal,
        "scheduler_state_equal": scheduler_equal,
        "global_step_uninterrupted": uninterrupted["global_step"],
        "global_step_resumed": resumed["global_step"],
        "staged_checkpoint_bytes": staged["checkpoint_bytes"],
        "peak_cuda_memory_uninterrupted": uninterrupted["peak_cuda_memory"],
        "peak_cuda_memory_resumed": resumed["peak_cuda_memory"],
        "checkpoint_bytes": uninterrupted["checkpoint_bytes"],
        "limitations": [
            "This test uses a 64-record bounded subset, not the full training corpus.",
            "CUDA and fused/native kernels can introduce non-bitwise differences on other systems.",
            "The checkpoint stores an explicit data order and position for this test.",
        ],
    }


def main() -> int:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "all":
        for path in experiment_dir.glob("*_result.pt"):
            path.unlink()
        uninterrupted = run_child("uninterrupted", experiment_dir)
        staged = run_child("staged", experiment_dir)
        resumed = run_child("resume", experiment_dir)
        comparison = compare_runs(experiment_dir)
        report = {
            "configuration": configuration(),
            "uninterrupted": uninterrupted,
            "staged": staged,
            "resumed": resumed,
            "comparison": comparison,
            "fresh_process_resume_test": True,
        }
        report_path = experiment_dir / "checkpoint_resume_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps({"report": str(report_path), **comparison}, indent=2))
        return 0
    result = train_process(args.mode, experiment_dir)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
