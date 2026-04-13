from __future__ import annotations

from pathlib import Path

import pytest

from histokit.dataset.dataset import Dataset
from histokit.dataset.schema import AnnotationSchema, DatasetSchema, SlideSchema

# ── paths ────────────────────────────────────────────────────────────────────

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CERVICAL_MINI = DATA_ROOT / "icaird" / "cervical_mini"
CAMELYON16 = DATA_ROOT / "camelyon16"


# ── skip helpers ─────────────────────────────────────────────────────────────

needs_cervical_mini = pytest.mark.skipif(
    not (CERVICAL_MINI / "index.csv").exists(),
    reason="cervical_mini data not available",
)
needs_camelyon16 = pytest.mark.skipif(
    not (CAMELYON16 / "index.csv").exists(),
    reason="camelyon16 data not available",
)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cervical_mini_dataset() -> Dataset:
    return Dataset.from_index(
        CERVICAL_MINI / "index.csv",
        CERVICAL_MINI / "labels.json",
        slides_dir="slides",
        annotations_dir="annotations",
    )


@pytest.fixture(scope="session")
def cervical_mini_samples(cervical_mini_dataset: Dataset) -> list:
    return list(cervical_mini_dataset.samples())


@pytest.fixture(scope="session")
def cervical_mini_schema() -> DatasetSchema:
    return DatasetSchema.from_json(CERVICAL_MINI / "labels.json")
