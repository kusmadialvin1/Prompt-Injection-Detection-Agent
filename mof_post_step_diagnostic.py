"""Probe operations that follow the resumed optimizer step without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "logs"
    / "mof_minimal_checkpoint_diagnostic"
    / "checkpoint_step_1.pth"
)
DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "logs" / "mof_post_step_diagnostic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    return parser.parse_args()


def mark(directory: Path, stage: str, event: str, details: dict[str, Any] | None = None) -> None:
    record: dict[str, Any] = {
        "timestamp": time.time(),
        "stage": stage,
        "event": event,
    }
    if details:
        record["details"] = details
    path = directory / "progress_markers.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    directory = args.experiment_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    marker = lambda stage, event, details=None: mark(directory, stage, event, details)

    marker("checkpoint_load", "before", {"path": str(args.checkpoint.resolve())})
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    marker(
        "checkpoint_load",
        "after",
        {
            "global_step": checkpoint["global_step"],
            "data_position": checkpoint["data_position"],
        },
    )

    global_step = checkpoint["global_step"]
    data_position = checkpoint["data_position"]
    scheduler_state = checkpoint["scheduler"]
    step_records: list[dict[str, Any]] = []
    losses: list[float] = []
    learning_rates: list[float] = []
    results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "operations": [],
    }

    marker("read_global_step_data_position", "before")
    observed_position = data_position + 1
    observed_step = global_step + 1
    marker(
        "read_global_step_data_position",
        "after",
        {"global_step": observed_step, "data_position": observed_position},
    )
    results["operations"].append("read_global_step_data_position")

    marker("scheduler_learning_rate_lookup", "before")
    last_lr = scheduler_state["_last_lr"]
    observed_lr = float(last_lr[0])
    marker(
        "scheduler_learning_rate_lookup",
        "after",
        {"learning_rate": observed_lr},
    )
    learning_rates.append(observed_lr)
    results["operations"].append("scheduler_learning_rate_lookup")

    marker("loss_scalar_cpu_conversion", "before")
    loss = torch.tensor(0.9654205441474915, device="cuda")
    observed_loss = float(loss.detach().cpu())
    del loss
    marker("loss_scalar_cpu_conversion", "after", {"loss": observed_loss})
    losses.append(observed_loss)
    results["operations"].append("loss_scalar_cpu_conversion")

    marker("step_record_construction", "before")
    step_records.append(
        {
            "global_step": observed_step,
            "loss": losses[-1],
            "learning_rate": learning_rates[-1],
        }
    )
    marker("step_record_construction", "after")
    results["operations"].append("step_record_construction")

    marker("small_json_write", "before")
    small_path = directory / "small_operation_result.json"
    with small_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "global_step": observed_step,
                "data_position": observed_position,
                "loss": observed_loss,
                "learning_rate": observed_lr,
            },
            handle,
            indent=2,
        )
        handle.flush()
        os.fsync(handle.fileno())
    marker("small_json_write", "after", {"path": str(small_path)})
    results["operations"].append("small_json_write")

    marker("model_digest", "before")
    model_digest = digest(checkpoint["model"])
    marker("model_digest", "after", {"digest": model_digest})
    results["model_digest"] = model_digest
    results["operations"].append("model_digest")

    marker("optimizer_digest", "before")
    optimizer_digest = digest(checkpoint["optimizer"])
    marker("optimizer_digest", "after", {"digest": optimizer_digest})
    results["optimizer_digest"] = optimizer_digest
    results["operations"].append("optimizer_digest")

    marker("scheduler_digest", "before")
    scheduler_digest = digest(checkpoint["scheduler"])
    marker("scheduler_digest", "after", {"digest": scheduler_digest})
    results["scheduler_digest"] = scheduler_digest
    results["operations"].append("scheduler_digest")

    marker("cuda_memory_queries", "before")
    results["peak_cuda_memory"] = {
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    marker("cuda_memory_queries", "after", results["peak_cuda_memory"])
    results["operations"].append("cuda_memory_queries")

    results.update(
        {
            "global_step": observed_step,
            "data_position": observed_position,
            "losses": losses,
            "learning_rates": learning_rates,
            "step_records": step_records,
        }
    )
    marker("full_result_assembly", "before")
    result_path = directory / "post_step_result.json"
    marker("full_result_assembly", "after")

    marker("full_json_write", "before", {"path": str(result_path)})
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    marker("full_json_write", "after", {"path": str(result_path)})

    marker("stdout_serialization", "before")
    output = json.dumps(results)
    marker("stdout_serialization", "after", {"length": len(output)})
    print(output)
    return results


def main() -> int:
    args = parse_args()
    return 0 if run(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
