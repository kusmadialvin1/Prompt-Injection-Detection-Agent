from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class SecurityLogger:
    def __init__(self, log_path: Path, preview_limit: int = 500):
        self.log_path = log_path
        self.preview_limit = preview_limit
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        stage: str,
        text: str,
        label: int,
        confidence: float | None,
        action: str,
        error: str | None = None,
        **extra,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "text_preview": text[: self.preview_limit],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "predicted_label": label,
            "confidence": confidence,
            "action": action,
        }
        if error:
            event["error"] = error
        if extra:
            event.update(extra)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
