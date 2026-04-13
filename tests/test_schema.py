"""Tests for histokit.dataset.schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from histokit.dataset.schema import AnnotationSchema, DatasetSchema, SlideSchema
from conftest import CERVICAL_MINI


# ── SlideSchema ──────────────────────────────────────────────────────────────

class TestSlideSchema:
    def test_fields(self):
        s = SlideSchema(kind="tiffslide", label_schema={"a": 0, "b": 1})
        assert s.kind == "tiffslide"
        assert s.label_schema == {"a": 0, "b": 1}

    def test_frozen(self):
        s = SlideSchema(kind="tiffslide", label_schema={})
        with pytest.raises(AttributeError):
            s.kind = "openslide"


# ── AnnotationSchema ─────────────────────────────────────────────────────────

class TestAnnotationSchema:
    def test_fields(self):
        a = AnnotationSchema(
            kind="geojson",
            label_map={"bg": 0, "fg": 1},
            cutout_label="bg",
            fill_label="bg",
            label_order=["bg", "fg"],
        )
        assert a.kind == "geojson"
        assert a.label_map["fg"] == 1
        assert a.cutout_label == "bg"
        assert a.fill_label == "bg"
        assert a.label_order == ["bg", "fg"]

    def test_frozen(self):
        a = AnnotationSchema(
            kind="geojson",
            label_map={},
            cutout_label="bg",
            fill_label="bg",
            label_order=[],
        )
        with pytest.raises(AttributeError):
            a.kind = "xml"


# ── DatasetSchema ────────────────────────────────────────────────────────────

class TestDatasetSchemaFromDict:
    """Test DatasetSchema.from_dict with both raw JSON-style keys and already-renamed keys."""

    def test_from_dict_json_style_keys(self):
        d = {
            "slides": {
                "type": "tiffslide",
                "labellings": {"cat": {"a": 0}},
            },
            "annotations": {
                "type": "geojson",
                "labels": {"bg": 0, "fg": 1},
                "cutout": "bg",
                "fill": "bg",
                "order": ["bg", "fg"],
            },
        }
        schema = DatasetSchema.from_dict(d)
        assert schema.slides.kind == "tiffslide"
        assert schema.annotations.kind == "geojson"
        assert schema.annotations.label_map == {"bg": 0, "fg": 1}
        assert schema.annotations.cutout_label == "bg"
        assert schema.annotations.fill_label == "bg"
        assert schema.annotations.label_order == ["bg", "fg"]

    def test_from_dict_already_renamed_keys(self):
        d = {
            "slides": {
                "kind": "openslide",
                "label_schema": {"cat": {"a": 0}},
            },
            "annotations": {
                "kind": "asapxml",
                "label_map": {"bg": 0},
                "cutout_label": "bg",
                "fill_label": "bg",
                "label_order": ["bg"],
            },
        }
        schema = DatasetSchema.from_dict(d)
        assert schema.slides.kind == "openslide"
        assert schema.annotations.kind == "asapxml"


class TestDatasetSchemaFromJson:
    @pytest.mark.skipif(
        not (CERVICAL_MINI / "labels.json").exists(),
        reason="cervical_mini data not available",
    )
    def test_from_json_cervical_mini(self):
        schema = DatasetSchema.from_json(CERVICAL_MINI / "labels.json")
        assert schema.slides.kind == "tiffslide"
        assert schema.annotations.kind == "geojson"
        assert "background" in schema.annotations.label_map
        assert "normal" in schema.annotations.label_map
        assert schema.annotations.fill_label == "normal"
        assert schema.annotations.cutout_label == "normal"
        assert len(schema.annotations.label_order) == 5
