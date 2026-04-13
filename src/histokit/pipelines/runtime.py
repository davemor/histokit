from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any


@dataclass
class RuntimeContext:
    params: dict[str, Any] = field(default_factory=dict)
    pipeline_name: str = "pipeline"
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())