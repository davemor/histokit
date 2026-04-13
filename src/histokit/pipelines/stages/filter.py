from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from histokit.patchset.manifest import PatchSetManifest
from histokit.patchset.patchset import PatchSet
from ..model import PatchCandidates
from ..params import resolve_value
from ..runtime import RuntimeContext
from ..stage import Stage


@dataclass(frozen=True)
class FilterPatches(Stage):
    tissue_threshold: float | Any = 0.0
    drop_background: bool | Any = True
    require_label: bool | Any = False

    def resolve(self, runtime: RuntimeContext) -> FilterPatches:
        values = {
            f.name: resolve_value(getattr(self, f.name), runtime)
            for f in fields(self)
        }
        return replace(self, **values)

    def run(self, candidates: PatchCandidates, runtime: RuntimeContext) -> PatchSet:
        candidates.require_columns("row", "column", "context_idx")

        frame = candidates.frame.copy()

        # Mark which patches to keep
        keep = frame.index == frame.index  # all True initially

        if self.drop_background and "tissue_score" in frame.columns:
            keep = keep & (frame["tissue_score"] > self.tissue_threshold)

        if self.require_label and "annotation_label" in frame.columns:
            keep = keep & frame["annotation_label"].notna()

        frame["keep"] = keep
        # Rename context_idx -> context_id for PatchSet compatibility
        frame = frame.rename(columns={"context_idx": "context_id"})
        frame = frame.reset_index(drop=True)

        manifest = PatchSetManifest(
            created_at=runtime.created_at,
        )

        return PatchSet(
            frame=frame,
            contexts=candidates.contexts,
            manifest=manifest,
        )
