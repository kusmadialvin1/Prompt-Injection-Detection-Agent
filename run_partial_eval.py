import json
import os

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from eval_hf import NotInject_eval, BIPIA_eval, wildguard_eval


DATASET_ROOT = "datasets"
MODEL_NAME = "leolee99/PIGuard"


def main():
    device = 0 if torch.cuda.is_available() else -1

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {'cuda' if device == 0 else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=2048,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        device=device,
    )

    notinject_overall, notinject_one, notinject_two, notinject_three = (
        NotInject_eval(classifier, DATASET_ROOT)
    )

    bipia_overall, bipia_text, bipia_code = BIPIA_eval(
        classifier,
        DATASET_ROOT,
    )

    wildguard_accuracy = wildguard_eval(
        classifier,
        DATASET_ROOT,
    )

    results = {
        "model": MODEL_NAME,
        "dataset_root": DATASET_ROOT,
        "device": "cuda" if device == 0 else "cpu",
        "metrics": {
            "NotInject": {
                "overall_accuracy": notinject_overall,
                "one_trigger_accuracy": notinject_one,
                "two_trigger_accuracy": notinject_two,
                "three_trigger_accuracy": notinject_three,
            },
            "BIPIA": {
                "overall_accuracy": bipia_overall,
                "text_accuracy": bipia_text,
                "code_accuracy": bipia_code,
            },
            "WildGuard-Benign": {
                "accuracy": wildguard_accuracy,
            },
        },
    }

    os.makedirs("logs", exist_ok=True)

    with open(
        "logs/partial_metrics_manual.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()