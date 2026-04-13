"""Tests for histokit.patchset (PatchSet, PatchContext, PatchSetManifest, combine_patchsets)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from histokit.dataset.sample import Sample
from histokit.dataset.schema import AnnotationSchema, SlideSchema
from histokit.patchset.context import PatchContext
from histokit.patchset.manifest import PatchSetManifest
from histokit.patchset.patchset import PatchSet, combine_patchsets
from conftest import needs_cervical_mini


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_sample(sample_id: str = "test_sample", slide_path: str = "/tmp/slide.tiff") -> Sample:
    return Sample(
        id=sample_id,
        slide_path=Path(slide_path),
        slide_schema=SlideSchema(kind="tiffslide", label_schema={}),
        annotation_schema=AnnotationSchema(
            kind="geojson",
            label_map={"background": 0, "normal": 1, "tumor": 2},
            cutout_label="normal",
            fill_label="normal",
            label_order=["background", "tumor", "normal"],
        ),
        metadata={"category": "test"},
    )


def _make_patchset(n: int = 10, num_contexts: int = 1, with_keep: bool = True) -> PatchSet:
    rows = []
    for i in range(n):
        rows.append({
            "row": i // 5,
            "column": i % 5,
            "context_id": i % num_contexts,
            "annotation_label": (i % 3),  # 0, 1, 2
            "tissue_score": float(i) / n,
        })
    frame = pd.DataFrame(rows)
    if with_keep:
        frame["keep"] = frame["tissue_score"] > 0.3

    contexts = [
        PatchContext(
            sample=_make_sample(f"sample_{c}"),
            level=1,
            patch_size=256,
        )
        for c in range(num_contexts)
    ]

    manifest = PatchSetManifest(created_at="2025-01-01T00:00:00Z")
    return PatchSet(frame=frame, contexts=contexts, manifest=manifest)


# ── PatchContext ─────────────────────────────────────────────────────────────

class TestPatchContext:
    def test_fields(self):
        sample = _make_sample()
        ctx = PatchContext(sample=sample, level=2, patch_size=512)
        assert ctx.level == 2
        assert ctx.patch_size == 512
        assert ctx.sample.id == "test_sample"

    def test_frozen(self):
        sample = _make_sample()
        ctx = PatchContext(sample=sample, level=1, patch_size=256)
        with pytest.raises(AttributeError):
            ctx.level = 3


# ── PatchSetManifest ─────────────────────────────────────────────────────────

class TestPatchSetManifest:
    def test_fields(self):
        m = PatchSetManifest(created_at="2025-01-01T00:00:00Z")
        assert m.created_at == "2025-01-01T00:00:00Z"


# ── PatchSet ─────────────────────────────────────────────────────────────────

class TestPatchSet:
    def test_repr(self):
        ps = _make_patchset()
        r = repr(ps)
        assert "PatchSet" in r
        assert "num_patches=10" in r

    def test_describe(self):
        ps = _make_patchset()
        desc = ps.describe()
        assert isinstance(desc, pd.DataFrame)
        assert "background" in desc.columns
        assert "normal" in desc.columns
        assert "tumor" in desc.columns
        assert desc["background"].iloc[0] + desc["normal"].iloc[0] + desc["tumor"].iloc[0] == 10

    def test_describe_no_annotation_schema(self):
        sample = Sample(
            id="no_annot",
            slide_path=Path("/tmp/s.tiff"),
            slide_schema=SlideSchema(kind="tiffslide", label_schema={}),
            metadata={},
        )
        ctx = PatchContext(sample=sample, level=1, patch_size=256)
        frame = pd.DataFrame({"row": [0], "column": [0], "context_id": [0]})
        ps = PatchSet(frame=frame, contexts=[ctx], manifest=PatchSetManifest(created_at="now"))
        desc = ps.describe()
        assert desc.empty


class TestPatchSetSaveLoad:
    def test_save_creates_files(self, tmp_path):
        ps = _make_patchset()
        out = tmp_path / "patchset_out"
        ps.save(out)
        assert (out / "frame.parquet").exists()
        assert (out / "manifest.json").exists()

    def test_manifest_json_content(self, tmp_path):
        ps = _make_patchset()
        out = tmp_path / "patchset_out"
        ps.save(out)
        with open(out / "manifest.json") as f:
            data = json.load(f)
        assert "manifest" in data
        assert "contexts" in data
        assert data["manifest"]["created_at"] == "2025-01-01T00:00:00Z"
        assert len(data["contexts"]) == 1
        assert data["contexts"][0]["sample_id"] == "sample_0"

    @needs_cervical_mini
    def test_save_load_roundtrip(self, tmp_path, cervical_mini_dataset, cervical_mini_samples):
        """Full round-trip using real dataset samples so load() can reconstruct Sample objects."""
        sample = cervical_mini_samples[0]
        ctx = PatchContext(sample=sample, level=1, patch_size=256)
        frame = pd.DataFrame({
            "row": [0, 1, 2],
            "column": [0, 0, 0],
            "context_id": [0, 0, 0],
            "annotation_label": [1, 2, 1],
            "keep": [True, False, True],
        })
        original = PatchSet(
            frame=frame,
            contexts=[ctx],
            manifest=PatchSetManifest(created_at="2025-06-01T12:00:00Z"),
        )
        out = tmp_path / "rt"
        original.save(out)
        loaded = PatchSet.load(out, cervical_mini_dataset)
        assert len(loaded.frame) == 3
        assert len(loaded.contexts) == 1
        assert loaded.contexts[0].sample.id == sample.id
        assert loaded.manifest.created_at == "2025-06-01T12:00:00Z"
        pd.testing.assert_frame_equal(
            loaded.frame.reset_index(drop=True),
            original.frame.reset_index(drop=True),
        )


# ── combine_patchsets ────────────────────────────────────────────────────────

class TestCombinePatchsets:
    def test_combine_two(self):
        ps1 = _make_patchset(n=5, num_contexts=1)
        ps2 = _make_patchset(n=3, num_contexts=1)
        combined = combine_patchsets([ps1, ps2])
        assert len(combined.frame) == 8
        assert len(combined.contexts) == 2
        # context_ids in ps2 should have been offset
        assert set(combined.frame["context_id"].unique()) == {0, 1}

    def test_combine_single(self):
        ps = _make_patchset(n=4)
        combined = combine_patchsets([ps])
        assert combined is ps  # same object returned

    def test_combine_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            combine_patchsets([])

    def test_combine_preserves_columns(self):
        ps1 = _make_patchset(n=4)
        ps2 = _make_patchset(n=3)
        combined = combine_patchsets([ps1, ps2])
        for col in ["row", "column", "context_id", "annotation_label", "tissue_score", "keep"]:
            assert col in combined.frame.columns
