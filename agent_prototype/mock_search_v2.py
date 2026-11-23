"""Deterministic V2 fixture; the legacy mock_search implementation is untouched."""
import json
from pathlib import Path

from tools import ToolResult

RESULTS_PATH = Path(__file__).with_name("mock_search_results_v2.json")


class MockSearchV2:
    name = "mock_search"

    def __init__(self, path: Path = RESULTS_PATH):
        self.results = json.loads(path.read_text(encoding="utf-8"))

    def run(self, query: str) -> ToolResult:
        try:
            item = self.results[query]
        except KeyError as error:
            raise ValueError(f"No V2 search fixture for query: {query!r}") from error
        return ToolResult(self.name, f"Result ID: {item['id']}\nTitle: {item['title']}\nContent: {item['content']}")
