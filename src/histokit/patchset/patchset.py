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