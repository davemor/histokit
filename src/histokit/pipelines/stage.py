from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Stage(ABC):
    def __rshift__(self, other: "Stage") -> "Pipeline":
        from .pipeline import Pipeline
        return Pipeline([self, other])

    @abstractmethod
    def run(self, data: Any, runtime: "RuntimeContext") -> Any:
        raise NotImplementedError

    def resolve(self, runtime: "RuntimeContext") -> "Stage":
        return self