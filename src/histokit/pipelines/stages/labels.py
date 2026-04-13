from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import pandas as pd

from histokit.utils.convert import to_frame_with_locations
from histokit.utils.geometry import Shape
from ..model import PatchCandidates
from ..params import resolve_value
from ..runtime import RuntimeContext
from ..stage import Stage


@dataclass(frozen=True)
class AssignLabels(Stage):
    policy: str | Any = "majority"

    def resolve(self, runtime: RuntimeContext) -> AssignLabels:
        values = {
            f.name: resolve_value(getattr(self, f.name), runtime)
            for f in fields(self)
        }
        return replace(self, **values)

    def run(self, candidates: PatchCandidates, runtime: RuntimeContext) -> PatchCandidates:
        candidates = candidates.copy()
        candidates.require_columns("row", "column", "context_idx")

        context = candidates.contexts[0]
        annotation_set = context.sample.make_annotations()

        grid_h = int(candidates.frame["row"].max()) + 1
        grid_w = int(candidates.frame["column"].max()) + 1
        shape = Shape(grid_h, grid_w)

        if annotation_set is not None and len(annotation_set) > 0:
            label_grid = annotation_set.render_to_grid(
                shape, context.patch_size, context.level,
            )
            label_df = to_frame_with_locations(label_grid, "annotation_label")
            candidates.frame = candidates.frame.merge(
                label_df[["row", "column", "annotation_label"]],
                on=["row", "column"],
                how="left",
            )
        else:
            schema = context.sample.annotation_schema
            fill_val: int | float = np.nan
            if schema is not None:
                fill_val = schema.label_map[schema.fill_label]
            candidates.frame["annotation_label"] = fill_val

        candidates.metadata["label_policy"] = self.policy
        return candidates
