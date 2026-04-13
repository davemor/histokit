
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import pandas as pd

from histokit.dataset.sample import Sample
from histokit.patchset.context import PatchContext
from ..model import PatchCandidates
from ..params import resolve_value
from ..runtime import RuntimeContext
from ..stage import Stage


@dataclass(frozen=True)
class Grid(Stage):
    level: int | Any = 1
    patch_size: int | Any = 256
    stride: int | Any | None = None

    def resolve(self, runtime: RuntimeContext) -> Grid:
        values = {
            f.name: resolve_value(getattr(self, f.name), runtime)
            for f in fields(self)
        }
        if values["stride"] is None:
            values["stride"] = values["patch_size"]
        return replace(self, **values)

    def run(self, sample: Sample, runtime: RuntimeContext) -> PatchCandidates:
        context = PatchContext(
            sample=sample,
            level=self.level,
            patch_size=self.patch_size,
        )

        rows: list[tuple[int, int, int]] = []
        with sample.open_slide() as slide:
            size = slide.size_in_patches(self.patch_size, self.level)
            for r in range(size.height):
                for c in range(size.width):
                    rows.append((r, c, 0))

        frame = pd.DataFrame(rows, columns=["row", "column", "context_idx"])

        return PatchCandidates.from_grid(
            frame=frame,
            contexts=[context],
            metadata={
                "stage": "grid",
                "level": self.level,
                "patch_size": self.patch_size,
                "stride": self.stride,
            },
        )