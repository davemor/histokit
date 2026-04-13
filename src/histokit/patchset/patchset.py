import pandas as pd

from .manifest import PatchSetManifest
from .context import PatchContext

class PatchSet:
    def __init__(
        self,
        frame: pd.DataFrame,  # columns: x, y, labels, context_id, keep - also any additional columns based on quantities extracted in the pipeline
        contexts: list[PatchContext],
        manifest: PatchSetManifest,
    ) -> None:
        self.frame = frame
        self.contexts = contexts
        self.manifest = manifest

    def __repr__(self) -> str:
        return f"PatchSet(num_patches={len(self.frame)}, contexts={self.contexts}, manifest={self.manifest})"
    
    def describe(self) -> pd.DataFrame:
        annotation_schema = None

        # find the first annotation schema from the contexts to get the label map for counting annotation labels - assumes all contexts in the patchset have the same annotation schema if they have annotations at all
        for ctx in self.contexts:
            if ctx.sample.annotation_schema is not None:
                annotation_schema = ctx.sample.annotation_schema
                break

        if annotation_schema is None:
            return pd.DataFrame()

        label_map = annotation_schema.label_map
        counts = {
            label: int((self.frame["annotation_label"] == value).sum())
            for label, value in label_map.items()
        }
        return pd.DataFrame([counts])