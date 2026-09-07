"""
Agent base class with logging, latency tracking, and call-count tracking.

All agents inherit from BaseAgent. Every agent call is logged with:
  - timestamp, agent name, action, latency, inputs/outputs summary
This log is used to generate the RQ6 system cost metrics.
"""

import time
import json
from pathlib import Path
from typing import Any


class BaseAgent:
    """
    Abstract base class for all UMiKT-GAT agents.

    Subclasses must implement the `run()` method.
    All calls are automatically logged for system cost tracking.
    """

    def __init__(self, name: str, log_dir: Path | None = None):
        self.name = name
        self.call_count = 0
        self.total_latency_sec = 0.0
        self._log_dir = log_dir
        self._log: list[dict] = []

    def run(self, **kwargs) -> Any:
        """Override this method in subclasses."""
        raise NotImplementedError

    def __call__(self, **kwargs) -> Any:
        """Invoke the agent, tracking latency and call count."""
        self.call_count += 1
        t0 = time.perf_counter()
        result = self.run(**kwargs)
        elapsed = time.perf_counter() - t0
        self.total_latency_sec += elapsed

        entry = {
            "agent": self.name,
            "call_index": self.call_count,
            "latency_sec": round(elapsed, 4),
            "cumulative_latency_sec": round(self.total_latency_sec, 4),
        }
        self._log.append(entry)
        return result

    def get_usage_stats(self) -> dict:
        return {
            "agent": self.name,
            "call_count": self.call_count,
            "total_latency_sec": round(self.total_latency_sec, 4),
            "avg_latency_sec": round(
                self.total_latency_sec / max(self.call_count, 1), 4
            ),
        }

    def save_log(self):
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            path = self._log_dir / f"{self.name}_log.json"
            with open(path, "w") as f:
                json.dump(self._log, f, indent=2)
