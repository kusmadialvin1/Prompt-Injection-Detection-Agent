import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
MAX_LENGTH = 256
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise ValueError(f"Required dataset file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON in {path}: {error}") from error


def require_prompt(record, path, index):
    if not isinstance(record, dict):
        raise ValueError(f"{path} record {index} must be an object.")
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path} record {index} has no non-empty string prompt.")
    return prompt


def load_datasets():
    datasets = {}
    for name in ("NotInject_one", "NotInject_two", "NotInject_three"):
        path = DATASET_DIR / f"{name}.json"
        records = load_json(path)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array.")
        datasets[name] = [
            (require_prompt(record, path, index), 0)
            for index, record in enumerate(records, 1)
        ]

    path = DATASET_DIR / "wildguard.json"
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array.")
    datasets["WildGuard-Benign"] = [
        (require_prompt(record, path, index), 0)
        for index, record in enumerate(records, 1)
    ]

    for name in ("BIPIA_text", "BIPIA_code"):
        path = DATASET_DIR / f"{name}.json"
        records = load_json(path)
        if not isinstance(records, dict):
            raise ValueError(f"{path} must contain a JSON object of lists.")
        samples = []
        for category, values in records.items():
            if not isinstance(values, list):
                raise ValueError(f"{path} category {category!r} must contain a list.")
            for index, prompt in enumerate(values, 1):
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        f"{path} category {category!r} item {index} has no non-empty string prompt."
                    )
                samples.append((prompt, 1))
        datasets[name.replace("_", "-")] = samples

    if any(not samples for samples in datasets.values()):
        empty = [name for name, samples in datasets.items() if not samples]
        raise ValueError(f"Evaluation datasets are empty: {empty}")
    return datasets


def resolve_label_mapping(config):
    raw_mapping = config.id2label
    if not isinstance(raw_mapping, dict):
        raise ValueError("ProtectAI model configuration has no usable id2label mapping.")

    mapping = {int(index): str(label) for index, label in raw_mapping.items()}
    if set(mapping) != set(range(config.num_labels)):
        raise ValueError(f"Unexpected id2label mapping: {raw_mapping!r}")

    benign = [
        index for index, label in mapping.items()
        if any(word in label.lower() for word in ("benign", "safe", "normal", "non-injection"))
    ]
    injection = [
        index for index, label in mapping.items()
        if any(word in label.lower() for word in ("injection", "malicious", "attack"))
    ]
    if len(benign) != 1 or len(injection) != 1 or benign[0] == injection[0]:
        raise ValueError(
            "Could not unambiguously resolve benign/injection labels from "
            f"model.config.id2label={mapping!r}; refusing to guess."
        )
    return {
        "benign": benign[0],
        "injection": injection[0],
        "id2label": {str(index): label for index, label in mapping.items()},
    }


def evaluate_dataset(model, tokenizer, name, samples, prediction_file):
    correct = 0
    rows = []
    with torch.inference_mode():
        for start in range(0, len(samples), BATCH_SIZE):
            batch = samples[start:start + BATCH_SIZE]
            texts = [text for text, _ in batch]
            labels = [label for _, label in batch]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1).cpu().tolist()
            confidences = probabilities.max(dim=-1).values.cpu().tolist()

            for offset, (text, true_label) in enumerate(batch):
                predicted_label = int(predictions[offset])
                is_correct = predicted_label == true_label
                correct += int(is_correct)
                rows.append({
                    "dataset": name,
                    "sample_index": start + offset,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": is_correct,
                    "confidence": float(confidences[offset]),
                    "prompt": text,
                })

    with prediction_file.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(samples)
    return {
        "dataset": name,
        "class": "benign" if samples[0][1] == 0 else "injection",
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
    }


def main():
    output_dir = BASE_DIR / "logs" / f"protectai_eval_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_file = output_dir / "predictions.jsonl"

    datasets = load_datasets()
    config = AutoConfig.from_pretrained(MODEL_NAME)
    label_mapping = resolve_label_mapping(config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
    ).to(DEVICE)
    model.eval()

    results = [
        evaluate_dataset(model, tokenizer, name, samples, prediction_file)
        for name, samples in datasets.items()
    ]

    benign_results = [result for result in results if result["class"] == "benign"]
    attack_results = [result for result in results if result["class"] == "injection"]
    notinject_results = [
        result for result in results if result["dataset"].startswith("NotInject_")
    ]
    notinject_total = sum(result["total"] for result in notinject_results)
    notinject_correct = sum(result["correct"] for result in notinject_results)
    benign_total = sum(result["total"] for result in benign_results)
    benign_correct = sum(result["correct"] for result in benign_results)
    attack_total = sum(result["total"] for result in attack_results)
    attack_correct = sum(result["correct"] for result in attack_results)

    predictions = [
        json.loads(line)
        for line in prediction_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    notinject_false_positives = sum(
        row["predicted_label"] == label_mapping["injection"]
        for row in predictions
        if row["dataset"].startswith("NotInject_")
    )
    summary = {
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "evaluation_config": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "pint_status": "not evaluated; pending authorization",
        },
        "label_mapping": label_mapping,
        "dataset_counts": {name: len(samples) for name, samples in datasets.items()},
        "datasets": results,
        "notinject_pooled": {
            "correct": notinject_correct,
            "total": notinject_total,
            "accuracy": notinject_correct / notinject_total,
            "false_positives": notinject_false_positives,
            "false_positive_rate": notinject_false_positives / notinject_total,
        },
        "pooled_benign": {
            "correct": benign_correct,
            "total": benign_total,
            "accuracy": benign_correct / benign_total,
        },
        "pooled_attack_detection": {
            "correct": attack_correct,
            "total": attack_total,
            "accuracy": attack_correct / attack_total,
        },
        "pooled_overall": {
            "correct": benign_correct + attack_correct,
            "total": benign_total + attack_total,
            "accuracy": (benign_correct + attack_correct) / (benign_total + attack_total),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
