"""Structured episode tracing.

Every planner/executor/verifier event is appended to a JSONL trace so runs
are debuggable and eval metrics are computed from ground truth, not vibes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Trace:
    path: Path
    t0: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")  # truncate

    def log(self, event: str, **data) -> None:
        record = {"t": round(time.monotonic() - self.t0, 3), "event": event, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
