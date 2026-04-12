from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatchSetManifest:
    pipeline_name: str
    dataset_name: str
    sample_count: int
    created_at: str