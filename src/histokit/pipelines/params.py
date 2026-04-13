from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import RuntimeContext


@dataclass(frozen=True)
class Param:
    name: str
    default: Any = None

    def resolve(self, runtime: RuntimeContext) -> Any:
        return runtime.params.get(self.name, self.default)


def param(name: str, default: Any = None) -> Param:
    return Param(name=name, default=default)


def resolve_value(value: Any, runtime: RuntimeContext) -> Any:
    if isinstance(value, Param):
        return value.resolve(runtime)
    return value