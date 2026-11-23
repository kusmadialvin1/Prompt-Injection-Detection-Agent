"""Diagnose CUDA-resident post-step operations using an existing checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME = "microsoft/deberta-v3-base"
DATASET_PATH = PROJECT_ROOT / "datasets" / "train.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "mof_minimal_checkpoint_diagnostic"
    / "checkpoint_step_1.pth"
)
EXPERIMENT_DIR = PROJECT_ROOT / "logs" / "mof_cuda_post_step_diagnostic"
SEED = 42
RECORD_COUNT = 4
MAX_LENGTH = 32
LEARNING_RATE = 2e-5
TOTAL_STEPS = 2


def mark(directory: Path, stage: str, event: str, details: dict[str, Any] | None = None) -> None:
    record: dict[str, Any] = {"timestamp": time.time(), "stage": stage, "event": event}
    if details:
        record["details"] = details
    with (directory / "progress_markers.jsonl").open("a", encoding="utf-8") as handle:
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


def set_seed() -> None:
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def load_records() -> list[dict[str, Any]]:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return records[:RECORD_COUNT]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoTokenizer, get_scheduler
    from PIGuard import PIGuard

    directory = args.experiment_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    marker = lambda stage, event, details=None: mark(directory, stage, event, details)
    checkpoint_path = args.checkpoint.resolve()
    device = torch.device("cuda")

    marker("cuda_check", "before")
    if not torch.cuda.is_available():
        marker("cuda_check", "failed", {"reason": "CUDA unavailable"})
        raise RuntimeError("CUDA is unavailable.")
    marker("cuda_check", "after", {"device": torch.cuda.get_device_name(device)})

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
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.long)
    marker("encoding", "after")

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
    marker("model_construction", "after", {"dtype": "float32"})

    marker("checkpoint_load", "before", {"path": str(checkpoint_path)})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    marker(
        "checkpoint_load",
        "after",
        {"global_step": checkpoint["global_step"], "data_position": checkpoint["data_position"]},
    )

    data_position = int(checkpoint["data_position"])
    data_order = checkpoint["data_order"]
    index = data_order[data_position]
    input_ids = encoded["input_ids"][index : index + 1].to(device)
    attention_mask = encoded["attention_mask"][index : index + 1].to(device)
    batch_labels = labels[index : index + 1].to(device)

    marker("resumed_forward", "before", {"global_step": 2})
    _, loss = model(input_ids, attention_mask, batch_labels, mode="train")
    marker("resumed_forward", "after", {"global_step": 2})
    marker("resumed_backward", "before", {"global_step": 2})
    loss.backward()
    marker("resumed_backward", "after", {"global_step": 2})
    marker("resumed_optimizer_step", "before", {"global_step": 2})
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    torch.cuda.synchronize(device)
    marker("resumed_optimizer_step", "after", {"global_step": 2})

    data_position += 1
    observed_loss = float(loss.detach().cpu())
    observed_lr = float(scheduler.get_last_lr()[0])

    marker("model_state_digest", "before")
    model_digest = digest(model.state_dict())
    marker("model_state_digest", "after", {"digest": model_digest})

    marker("optimizer_state_digest", "before")
    optimizer_digest = digest(optimizer.state_dict())
    marker("optimizer_state_digest", "after", {"digest": optimizer_digest})

    marker("scheduler_state_digest", "before")
    scheduler_digest = digest(scheduler.state_dict())
    marker("scheduler_state_digest", "after", {"digest": scheduler_digest})

    marker("cuda_memory_queries", "before")
    peak_cuda_memory = {
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    marker("cuda_memory_queries", "after", peak_cuda_memory)

    marker("result_dictionary_construction", "before")
    result = {
        "status": "passed",
        "checkpoint": str(checkpoint_path),
        "global_step": 2,
        "data_position": data_position,
        "loss": observed_loss,
        "learning_rate": observed_lr,
        "model_digest": model_digest,
        "optimizer_digest": optimizer_digest,
        "scheduler_digest": scheduler_digest,
        "peak_cuda_memory": peak_cuda_memory,
    }
    marker("result_dictionary_construction", "after")

    result_path = directory / "cuda_post_step_result.json"
    marker("json_file_write", "before", {"path": str(result_path)})
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    marker("json_file_write", "after", {"path": str(result_path)})

    marker("json_stdout_serialization", "before")
    output = json.dumps(result)
    marker("json_stdout_serialization", "after", {"length": len(output)})
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
