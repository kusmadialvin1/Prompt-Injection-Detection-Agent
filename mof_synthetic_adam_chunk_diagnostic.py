"""Compare standard and chunked FP32 Adam updates on the largest PIGuard tensor shape."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_ROOT / "logs" / "mof_synthetic_adam_chunk_diagnostic"
SHAPE = (128100, 768)
CHUNK_ELEMENTS = 1_048_576
LEARNING_RATE = 2e-5
BETAS = (0.9, 0.999)
EPSILON = 1e-8
WEIGHT_DECAY = 0.0
INITIAL_STEP = 1
SEED = 42


def mark(directory: Path, stage: str, event: str, details: dict[str, Any] | None = None) -> None:
    record: dict[str, Any] = {"timestamp": time.time(), "stage": stage, "event": event}
    if details:
        record["details"] = details
    with (directory / "progress_markers.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cleanup_cuda(torch: Any, values: list[Any]) -> None:
    for value in values:
        del value
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def move_result_to_cpu(torch: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for name, value in result.items()
    }


def memory_snapshot(torch: Any) -> dict[str, int]:
    free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "driver_free_bytes": free_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def make_initial_state(torch: Any) -> tuple[Any, Any, Any, Any]:
    torch.manual_seed(SEED)
    parameter = torch.randn(SHAPE, device="cuda", dtype=torch.float32)
    gradient = torch.randn(SHAPE, device="cuda", dtype=torch.float32)
    exp_avg = torch.randn(SHAPE, device="cuda", dtype=torch.float32)
    exp_avg_sq = torch.rand(SHAPE, device="cuda", dtype=torch.float32)
    return parameter, gradient, exp_avg, exp_avg_sq


def standard_adam(torch: Any, initial: tuple[Any, Any, Any, Any]) -> dict[str, Any]:
    parameter, gradient, exp_avg, exp_avg_sq = initial
    parameter = parameter.clone()
    gradient = gradient.clone()
    exp_avg = exp_avg.clone()
    exp_avg_sq = exp_avg_sq.clone()
    parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [parameter],
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )
    optimizer.state[parameter]["step"] = torch.tensor(
        INITIAL_STEP, device="cuda", dtype=torch.float32
    )
    optimizer.state[parameter]["exp_avg"] = exp_avg
    optimizer.state[parameter]["exp_avg_sq"] = exp_avg_sq
    parameter.grad = gradient
    optimizer.step()
    return {
        "parameter": parameter.detach(),
        "gradient": gradient,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
        "step": optimizer.state[parameter]["step"].detach(),
    }


def chunked_adam(torch: Any, initial: tuple[Any, Any, Any, Any]) -> dict[str, Any]:
    parameter, gradient, exp_avg, exp_avg_sq = (value.clone() for value in initial)
    beta1, beta2 = BETAS
    step = INITIAL_STEP + 1
    bias_correction1 = 1 - beta1**step
    bias_correction2_sqrt = (1 - beta2**step) ** 0.5
    step_size = LEARNING_RATE / bias_correction1
    flat_parameter = parameter.view(-1)
    flat_gradient = gradient.view(-1)
    flat_exp_avg = exp_avg.view(-1)
    flat_exp_avg_sq = exp_avg_sq.view(-1)

    with torch.no_grad():
        flat_exp_avg.lerp_(flat_gradient, 1 - beta1)
        flat_exp_avg_sq.mul_(beta2).addcmul_(
            flat_gradient, flat_gradient, value=1 - beta2
        )
        for start in range(0, flat_parameter.numel(), CHUNK_ELEMENTS):
            end = min(start + CHUNK_ELEMENTS, flat_parameter.numel())
            denominator = (
                flat_exp_avg_sq[start:end].sqrt() / bias_correction2_sqrt
            ).add_(EPSILON)
            flat_parameter[start:end].addcdiv_(
                flat_exp_avg[start:end],
                denominator,
                value=-step_size,
            )
            del denominator

    return {
        "parameter": parameter,
        "gradient": gradient,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
        "step": torch.tensor(step, device="cuda", dtype=torch.float32),
    }


def max_differences(torch: Any, standard: dict[str, Any], chunked: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("parameter", "exp_avg", "exp_avg_sq", "step"):
        left = standard[name]
        right = chunked[name]
        difference = (left - right).abs()
        denominator = right.abs().clamp_min(torch.finfo(right.dtype).tiny)
        result[name] = {
            "max_absolute_difference": float(difference.max().cpu()),
            "max_relative_error": float((difference / denominator).max().cpu()),
        }
    return result


def run_method(
    torch: Any,
    directory: Path,
    name: str,
    initial: tuple[Any, Any, Any, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    mark(directory, name, "before")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        if name == "standard_adam":
            result = standard_adam(torch, initial)
        else:
            result = chunked_adam(torch, initial)
        torch.cuda.synchronize()
        memory = memory_snapshot(torch)
        mark(directory, name, "after", memory)
        return result, {"status": "passed", "memory": memory}
    except Exception as error:
        failure = {
            "status": "failed",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "memory": memory_snapshot(torch),
        }
        mark(directory, name, "failed", failure)
        return None, failure


def main() -> int:
    import torch

    directory = EXPERIMENT_DIR.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "synthetic_adam_chunk_report.json"
    report: dict[str, Any] = {
        "status": "started",
        "configuration": {
            "shape": SHAPE,
            "dtype": "float32",
            "chunk_elements": CHUNK_ELEMENTS,
            "learning_rate": LEARNING_RATE,
            "betas": BETAS,
            "epsilon": EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "initial_step": INITIAL_STEP,
            "foreach": False,
            "seed": SEED,
            "data_or_model_training": False,
        },
    }

    if not torch.cuda.is_available():
        report.update({"status": "failed", "error": "CUDA unavailable"})
    else:
        mark(directory, "initial_state", "before")
        initial = make_initial_state(torch)
        torch.cuda.synchronize()
        mark(directory, "initial_state", "after", memory_snapshot(torch))

        standard_gpu_result, standard_report = run_method(
            torch, directory, "standard_adam", initial
        )
        report["standard_adam"] = standard_report
        standard_result = None
        if standard_gpu_result is not None:
            standard_result = move_result_to_cpu(torch, standard_gpu_result)
            standard_gpu_result.clear()
            del standard_gpu_result
            cleanup_cuda(torch, [])

        chunked_gpu_result, chunked_report = run_method(
            torch, directory, "chunked_adam", initial
        )
        report["chunked_adam"] = chunked_report
        chunked_result = None
        if chunked_gpu_result is not None:
            chunked_result = move_result_to_cpu(torch, chunked_gpu_result)
            chunked_gpu_result.clear()
            del chunked_gpu_result
            cleanup_cuda(torch, [])
        if chunked_result is not None and standard_result is not None:
            report["numerical_comparison"] = max_differences(
                torch, standard_result, chunked_result
            )
        cleanup_cuda(torch, list(initial))
        report["status"] = (
            "passed"
            if chunked_report["status"] == "passed"
            else "failed"
        )

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"report": str(report_path), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
