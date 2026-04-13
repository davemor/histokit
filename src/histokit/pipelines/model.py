from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import pandas as pd

from histokit.patchset.context import PatchContext


REQUIRED_GRID_COLUMNS = ["row", "column", "context_idx"]


@dataclass
class PatchCandidates:
    frame: pd.DataFrame
    contexts: list[PatchContext]
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> PatchCandidates:
        return PatchCandidates(
            frame=self.frame.copy(),
            contexts=list(self.contexts),
            metadata=dict(self.metadata),
        )

    def require_columns(self, *columns: str) -> None:
        missing = [col for col in columns if col not in self.frame.columns]
        if missing:
            raise ValueError(f"PatchCandidates is missing required columns: {missing}")

    def groupby_context(self) -> Iterator[tuple[int, PatchContext, pd.DataFrame]]:
        for context_idx, group in self.frame.groupby("context_idx"):
            yield context_idx, self.contexts[context_idx], group

    @classmethod
    def from_grid(
        cls,
        frame: pd.DataFrame,
        contexts: list[PatchContext],
        metadata: dict[str, Any] | None = None,
    ) -> PatchCandidates:
        missing = [col for col in REQUIRED_GRID_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"Grid frame is missing required columns: {missing}")
        return cls(
            frame=frame,
            contexts=contexts,
            metadata={} if metadata is None else metadata,
        )
