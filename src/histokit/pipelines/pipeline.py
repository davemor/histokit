from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .runtime import RuntimeContext
from .stage import Stage


@dataclass
class Pipeline:
    stages: list[Stage]
    name: str = "pipeline"

    def __rshift__(self, other: Stage) -> Pipeline:
        return Pipeline(self.stages + [other], name=self.name)

    def _run_one(self, sample: Any, runtime: RuntimeContext) -> Any:
        current: Any = sample
        for stage in self.stages:
            resolved = stage.resolve(runtime)
            current = resolved.run(current, runtime)
        return current

    def run(self, dataset: Any, **params: Any) -> list[Any]:
        runtime = RuntimeContext(
            params=params,
            pipeline_name=self.name,
            dataset_summary={
                "num_samples": len(dataset.index),
            },
        )
        return [self._run_one(sample, runtime) for sample in dataset.samples()]