from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime import RuntimeContext
from .stage import Stage


@dataclass
class Pipeline:
    stages: list[Stage]

    def __rshift__(self, other: Stage) -> "Pipeline":
        return Pipeline(self.stages + [other])

    def run(self, data: Any, **params: Any) -> Any:
        runtime = RuntimeContext(params=params)
        current = data
        for stage in self.stages:
            resolved = stage.resolve(runtime)
            current = resolved.run(current, runtime)
        return current