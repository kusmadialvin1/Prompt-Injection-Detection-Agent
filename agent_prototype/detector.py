from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIGuard import PIGuard


MODEL_NAME = "microsoft/deberta-v3-base"
CHECKPOINT = PROJECT_ROOT / "logs" / "mof_baseline_20261023" / "checkpoints" / "best_model.pth"
MAX_LENGTH = 256
LABEL_NAMES = {0: "benign", 1: "injection"}


@dataclass(frozen=True)
class DetectionResult:
    label: int
    label_name: str
    confidence: float | None
    error: str | None = None

    @property
    def is_injection(self) -> bool:
        return self.label == 1


class PIGuardDetector:
    """Reusable, argmax-only wrapper around the reproduced PIGuard checkpoint."""

    def __init__(self, device: torch.device | None = None, prediction_policy: str = "argmax"):
        if prediction_policy != "argmax":
            raise ValueError("Only the existing argmax classification policy is supported.")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_policy = prediction_policy
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> PIGuard:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"PIGuard checkpoint not found: {CHECKPOINT}")

        try:
            config = AutoConfig.from_pretrained(MODEL_NAME, local_files_only=True)
            AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        except Exception as error:
            raise RuntimeError(
                "The local microsoft/deberta-v3-base files are unavailable; "
                "no model download was attempted."
            ) from error

        if config.hidden_size != 768 or config.num_labels != 2:
            raise RuntimeError(
                f"Unexpected base-model configuration: hidden_size={config.hidden_size}, "
                f"num_labels={config.num_labels}"
            )

        state_dict = torch.load(CHECKPOINT, map_location=self.device, weights_only=True)
        if not isinstance(state_dict, dict):
            raise RuntimeError("The checkpoint is not a state dictionary.")
        classifier_weight = state_dict.get("classifier.weight")
        classifier_bias = state_dict.get("classifier.bias")
        if classifier_weight is None or classifier_bias is None:
            raise RuntimeError("The checkpoint has no classifier weight/bias keys.")
        if tuple(classifier_weight.shape) != (2, 768) or tuple(classifier_bias.shape) != (2,):
            raise RuntimeError(
                "Checkpoint classifier shapes do not match PIGuard's two-label architecture: "
                f"{tuple(classifier_weight.shape)}, {tuple(classifier_bias.shape)}"
            )

        model = PIGuard(MODEL_NAME, num_labels=2, device=self.device)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "Checkpoint architecture mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        if model.classifier.out_features != 2:
            raise RuntimeError("PIGuard classifier does not expose exactly two labels.")
        model.to(self.device)
        return model

    def detect(self, text: str) -> DetectionResult:
        if not isinstance(text, str) or not text.strip():
            return DetectionResult(1, LABEL_NAMES[1], None, "empty input")
        try:
            with torch.inference_mode():
                logits = self.model.classify(text)
                if logits.ndim != 2 or logits.shape[-1] != 2:
                    raise RuntimeError(f"Unexpected classifier output shape: {tuple(logits.shape)}")
                probabilities = torch.softmax(logits, dim=-1)
                label = int(logits.argmax(dim=-1).item())
                confidence = float(probabilities[0, label].item())
                if label not in LABEL_NAMES:
                    raise RuntimeError(f"Unexpected predicted label: {label}")
            return DetectionResult(label, LABEL_NAMES[label], confidence)
        except Exception as error:
            return DetectionResult(1, LABEL_NAMES[1], None, f"detector failure: {error}")
