from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import pandas as pd

from histokit.segmentation.registry import get_detector
from histokit.utils.convert import to_frame_with_locations
from ..model import PatchCandidates
from ..params import resolve_value
from ..runtime import RuntimeContext
from ..stage import Stage


@dataclass(frozen=True)
class TissueMask(Stage):
    method: str | Any = "per_patch_canny_ranker"

    def resolve(self, runtime: RuntimeContext) -> TissueMask:
        values = {
            f.name: resolve_value(getattr(self, f.name), runtime)
            for f in fields(self)
        }
        return replace(self, **values)

    def run(self, candidates: PatchCandidates, runtime: RuntimeContext) -> PatchCandidates:
        candidates = candidates.copy()
        candidates.require_columns("row", "column", "context_idx")

        context = candidates.contexts[0]
        factory = get_detector(self.method)
        detector = factory(
            patch_size=context.patch_size,
            patch_level=context.level,
        )
        with context.sample.open_slide() as slide:
            tissue_array = detector(slide)

        tissue_df = to_frame_with_locations(tissue_array, "tissue_score")
        candidates.frame = candidates.frame.merge(
            tissue_df[["row", "column", "tissue_score"]],
            on=["row", "column"],
            how="left",
        )

        candidates.metadata["tissue_method"] = self.method
        return candidates
