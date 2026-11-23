"""V2-only Ollama diagnostics; does not record prompts or response text."""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from agent import OllamaLLM


class DiagnosticOllamaLLM(OllamaLLM):
    def __init__(self, diagnostics_path: Path, model: str = "qwen3:4b-nothink"):
        super().__init__(model=model)
        self.diagnostics_path = diagnostics_path
        self.case_id = None

    def generate(self, prompt: str) -> str:
        """V2 comparison arm: change only the output-token budget."""
        return self._generate(prompt, {"num_predict": 512})

    def _generate(self, prompt: str, options: dict) -> str:
        # Preserve the legacy request structure and timeout; V2 sets num_predict.
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": self.case_id,
            "model": self.model,
            "request_options": dict(options),
            "http_status": None,
            "json_parsed": False,
            "response_field_exists": None,
            "response_field_type": None,
            "usable_text": False,
            "thinking_field_exists": None,
            "thinking_field_type": None,
            "thinking_has_text": None,
            "thinking_char_length": None,
            "done": None,
            "done_reason": None,
            "eval_count": None,
            "prompt_eval_count": None,
            "total_duration": None,
            "error_category": None,
            "error_details": None,
        }
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": options},
                timeout=120,
            )
            event["http_status"] = response.status_code
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                event["error_category"] = "http_error"
                event["error_details"] = str(error)
                raise
            try:
                payload = response.json()
            except ValueError as error:
                event["error_category"] = "malformed_json"
                event["error_details"] = str(error)
                raise
            event["json_parsed"] = True
            if not isinstance(payload, dict):
                event["error_category"] = "invalid_json_shape"
                event["error_details"] = f"Expected object, got {type(payload).__name__}"
                raise TypeError(event["error_details"])
            for key in ("done", "done_reason", "eval_count", "prompt_eval_count", "total_duration"):
                event[key] = payload.get(key)
            event["thinking_field_exists"] = "thinking" in payload
            if "thinking" in payload:
                thinking = payload["thinking"]
                event["thinking_field_type"] = type(thinking).__name__
                event["thinking_has_text"] = isinstance(thinking, str) and bool(thinking.strip())
                if isinstance(thinking, str):
                    event["thinking_char_length"] = len(thinking)
            event["response_field_exists"] = "response" in payload
            answer = payload.get("response")
            event["response_field_type"] = type(answer).__name__ if "response" in payload else None
            event["usable_text"] = isinstance(answer, str) and bool(answer.strip())
            if not event["usable_text"]:
                event["error_category"] = (
                    "missing_response" if "response" not in payload else
                    "non_string_response" if not isinstance(answer, str) else "empty_response"
                )
                event["error_details"] = "Ollama returned no text response."
                raise RuntimeError("Ollama returned no text response.")
            return answer
        except requests.RequestException as error:
            if event["error_category"] is None:
                event["error_category"] = "request_error"
                event["error_details"] = str(error)
            raise
        finally:
            self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
