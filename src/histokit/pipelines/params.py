from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from histokit.pipelines.runtime import RuntimeContext


@dataclass(frozen=True)
class Param:
    name: str
    default: Any = None

    def resolve(self, runtime: "RuntimeContext") -> Any:
        return runtime.params.get(self.name, self.default)


def param(name: str, default: Any = None) -> Param:
    return Param(name=name, default=default)