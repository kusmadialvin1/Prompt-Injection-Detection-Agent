"""Measure CUDA memory during FP32 checkpoint restoration without training."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME = "microsoft/deberta-v3-base"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "mof_minimal_checkpoint_diagnostic"
    / "checkpoint_step_1.pth"
)
EXPERIMENT_DIR = PROJECT_ROOT / "logs" / "mof_checkpoint_restore_memory_diagnostic"
LEARNING_RATE = 2e-5
TOTAL_STEPS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
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


def memory_snapshot(torch: Any, device: Any) -> dict[str, Any]:
    device_index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "driver_free_bytes": free_bytes,
        "driver_total_bytes": total_bytes,
    }


def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import get_scheduler
    from PIGuard import PIGuard

    directory = args.experiment_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    marker = lambda stage, event, details=None: mark(directory, stage, event, details)
    report: dict[str, Any] = {
        "status": "started",
        "checkpoint": str(checkpoint_path),
        "configuration": {
            "model_name": MODEL_NAME,
            "device": "cuda",
            "dtype": "float32",
            "forward": False,
            "backward": False,
            "optimizer_step": False,
            "checkpoint_restoration_only": True,
        },
        "measurements": {},
    }
    report_path = directory / "restore_memory_report.json"

    try:
        marker("cuda_check", "before")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        device = torch.device("cuda")
        marker("cuda_check", "after", {"device": torch.cuda.get_device_name(device)})

        marker("before_model_construction", "before")
        report["measurements"]["before_model_construction"] = memory_snapshot(torch, device)
        marker(
            "before_model_construction",
            "after",
            report["measurements"]["before_model_construction"],
        )

        marker("model_construction", "before")
        model = PIGuard(MODEL_NAME, num_labels=2, device=device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, foreach=False)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=TOTAL_STEPS,
        )
        torch.cuda.synchronize(device)
        report["measurements"]["after_model_construction"] = memory_snapshot(torch, device)
        marker(
            "model_construction",
            "after",
            report["measurements"]["after_model_construction"],
        )

        marker("checkpoint_load_cpu", "before", {"path": str(checkpoint_path)})
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        marker(
            "checkpoint_load_cpu",
            "after",
            {
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "global_step": checkpoint["global_step"],
                "data_position": checkpoint["data_position"],
            },
        )

        marker("checkpoint_restoration", "before")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.cuda.synchronize(device)
        report["measurements"]["after_checkpoint_restoration"] = memory_snapshot(
            torch, device
        )
        marker(
            "checkpoint_restoration",
            "after",
            report["measurements"]["after_checkpoint_restoration"],
        )

        marker("delete_cpu_checkpoint", "before")
        del checkpoint
        gc.collect()
        torch.cuda.synchronize(device)
        report["measurements"]["after_cpu_checkpoint_delete_gc"] = memory_snapshot(
            torch, device
        )
        marker(
            "delete_cpu_checkpoint",
            "after",
            report["measurements"]["after_cpu_checkpoint_delete_gc"],
        )

        report["status"] = "passed"
        report["note"] = (
            "Restoration-only diagnostic. No forward, backward, or optimizer step was run."
        )
    except Exception as error:
        report["status"] = "failed"
        report["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        marker(
            "exception",
            "recorded",
            {"type": type(error).__name__, "message": str(error)},
        )
    finally:
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"report": str(report_path), "status": report["status"]}))

    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
