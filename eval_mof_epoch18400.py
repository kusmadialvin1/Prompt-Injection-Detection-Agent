
import json
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from PIGuard import PIGuard


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"

CHECKPOINT = (
    BASE_DIR
    / "logs"
    / "mof_baseline_20261023"
    / "checkpoints"
    / "epoch_1_18400_model.pth"
)

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = (
    BASE_DIR
    / "logs"
    / f"recheck_epoch18400_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)


# ============================================================
# Dataset loading
# Label: 0 = benign, 1 = injection
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_datasets():
    datasets = {}

    # NotInject: benign samples
    for name in [
        "NotInject_one",
        "NotInject_two",
        "NotInject_three",
    ]:
        records = load_json(DATASET_DIR / f"{name}.json")
        datasets[name] = [
            (sample["prompt"], 0)
            for sample in records
        ]

    # WildGuard: benign samples
    records = load_json(DATASET_DIR / "wildguard.json")
    datasets["WildGuard-Benign"] = [
        (sample["prompt"], 0)
        for sample in records
    ]

    # BIPIA text: injection samples
    records = load_json(DATASET_DIR / "BIPIA_text.json")
    datasets["BIPIA-text"] = [
        (context, 1)
        for contexts in records.values()
        for context in contexts
    ]

    # BIPIA code: injection samples
    records = load_json(DATASET_DIR / "BIPIA_code.json")
    datasets["BIPIA-code"] = [
        (context, 1)
        for contexts in records.values()
        for context in contexts
    ]

    return datasets


# ============================================================
# Model loading
# ============================================================

def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    model = PIGuard(
        MODEL_NAME,
        num_labels=2,
        device=DEVICE
    )

    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint architecture mismatch. "
            "Evaluation stopped to avoid invalid results."
        )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Prediction and export
# ============================================================

def evaluate_dataset(model, name, samples, prediction_file):
    total = 0
    correct = 0
    correct_by_class = defaultdict(int)

    with open(prediction_file, "a", encoding="utf-8") as f:
        for index, (text, true_label) in enumerate(samples):

            with torch.inference_mode():
                logits = model.classify(text)
                probabilities = torch.softmax(logits, dim=-1)

                predicted_label = int(
                    logits.argmax(dim=-1).item()
                )

                confidence = float(
                    probabilities[0, predicted_label].item()
                )

            is_correct = predicted_label == true_label

            total += 1
            correct += int(is_correct)
            correct_by_class[true_label] += int(is_correct)

            record = {
                "dataset": name,
                "sample_index": index,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": is_correct,
                "confidence": confidence,
                "prompt": text,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            del logits, probabilities

    return {
        "dataset": name,
        "class": "benign" if samples[0][1] == 0 else "injection",
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else None,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\nLoading datasets...")
    datasets = load_datasets()

    for name, samples in datasets.items():
        if not samples:
            raise ValueError(f"Dataset is empty: {name}")

        print(f"{name}: {len(samples)} samples")

    print("\nLoading model...")
    model = load_model()

    prediction_file = OUTPUT_DIR / "predictions.jsonl"
    results = []

    print("\nStarting evaluation...")

    for name, samples in datasets.items():
        print(f"\nEvaluating {name}...")

        result = evaluate_dataset(
            model,
            name,
            samples,
            prediction_file
        )

        results.append(result)

        print(
            f"Accuracy: {result['correct']}/{result['total']} "
            f"= {result['accuracy']:.4%}"
        )

    # Pooled class-level metrics
    benign_results = [
        r for r in results if r["class"] == "benign"
    ]

    attack_results = [
        r for r in results if r["class"] == "injection"
    ]

    benign_total = sum(r["total"] for r in benign_results)
    benign_correct = sum(r["correct"] for r in benign_results)

    attack_total = sum(r["total"] for r in attack_results)
    attack_correct = sum(r["correct"] for r in attack_results)

    overall_total = benign_total + attack_total
    overall_correct = benign_correct + attack_correct

    summary = {
        "checkpoint": str(CHECKPOINT),
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "device": str(DEVICE),
        "label_mapping": {
            "0": "benign",
            "1": "injection"
        },
        "pint_status": "Not evaluated - access authorization pending",
        "datasets": results,
        "benign": {
            "correct": benign_correct,
            "total": benign_total,
            "accuracy": benign_correct / benign_total
        },
        "attack": {
            "correct": attack_correct,
            "total": attack_total,
            "accuracy": attack_correct / attack_total
        },
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": overall_correct / overall_total
        }
    }

    summary_file = OUTPUT_DIR / "summary.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)

    print(
        f"Benign: {benign_correct}/{benign_total} "
        f"= {benign_correct / benign_total:.4%}"
    )

    print(
        f"Attack: {attack_correct}/{attack_total} "
        f"= {attack_correct / attack_total:.4%}"
    )

    print(
        f"Overall: {overall_correct}/{overall_total} "
        f"= {overall_correct / overall_total:.4%}"
    )

    print("\nResults saved to:", OUTPUT_DIR)
    print("Prediction file:", prediction_file)
    print("Summary file:", summary_file)


if __name__ == "__main__":
    main()

