from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatchContext:
    sample_id: str  # the unique identifier for the slide this patch came from in the dataset index
    slide_path: str # the path to the slide this patch came from, relative to the dataset's data root
    annotation_path: str | None # the path to the annotation file for the slide this patch came from, relative to the dataset's data root (or None if no annotations)
    metadata: dict[str, Any] # any additional metadata from the dataset index for the slide, such a slide level labels
    level: int  # the level of the slide pyramid this patch was extracted from
    patch_size: int  # the size of the patch in pixels (e.g. 512 for 512x512 patches)
