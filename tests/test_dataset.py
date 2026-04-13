"""Tests for histokit.dataset.dataset and histokit.dataset.sample."""

from __future__ import annotations

import pytest

from histokit.dataset.dataset import Dataset
from histokit.dataset.sample import Sample
from conftest import needs_cervical_mini


# ── Dataset.from_index ───────────────────────────────────────────────────────

@needs_cervical_mini
class TestDatasetFromIndex:
    def test_index_shape(self, cervical_mini_dataset: Dataset):
        assert len(cervical_mini_dataset.index) == 3

    def test_schema_types(self, cervical_mini_dataset: Dataset):
        assert cervical_mini_dataset.slide_kind == "tiffslide"
        assert cervical_mini_dataset.annotation_kind == "geojson"

    def test_slide_column_prefixed(self, cervical_mini_dataset: Dataset):
        for val in cervical_mini_dataset.index["slide"]:
            assert val.startswith("slides/")

    def test_annotation_column_prefixed(self, cervical_mini_dataset: Dataset):
        for val in cervical_mini_dataset.index["annotation"].dropna():
            assert val.startswith("annotations/")


# ── Dataset.samples ──────────────────────────────────────────────────────────

@needs_cervical_mini
class TestDatasetSamples:
    def test_yields_three_samples(self, cervical_mini_samples):
        assert len(cervical_mini_samples) == 3

    def test_sample_type(self, cervical_mini_samples):
        for s in cervical_mini_samples:
            assert isinstance(s, Sample)

    def test_sample_ids(self, cervical_mini_samples):
        ids = {s.id for s in cervical_mini_samples}
        assert ids == {"IC-CX-00001-01", "IC-CX-00002-01", "IC-CX-00003-01"}


# ── Sample attributes ────────────────────────────────────────────────────────

@needs_cervical_mini
class TestSampleAttributes:
    def test_slide_path_exists(self, cervical_mini_samples):
        for s in cervical_mini_samples:
            assert s.slide_path.exists(), f"Missing slide: {s.slide_path}"

    def test_annotation_path_exists(self, cervical_mini_samples):
        for s in cervical_mini_samples:
            if s.annotation_path is not None:
                assert s.annotation_path.exists(), f"Missing annotation: {s.annotation_path}"

    def test_metadata_keys(self, cervical_mini_samples):
        for s in cervical_mini_samples:
            assert "category" in s.metadata
            assert "subcategory" in s.metadata
            assert "split" in s.metadata

    def test_slide_schema(self, cervical_mini_samples):
        for s in cervical_mini_samples:
            assert s.slide_schema.kind == "tiffslide"

    def test_annotation_schema(self, cervical_mini_samples):
        for s in cervical_mini_samples:
            assert s.annotation_schema is not None
            assert s.annotation_schema.kind == "geojson"
