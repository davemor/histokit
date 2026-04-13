"""Tests for histokit.io (slides and annotations)."""

from __future__ import annotations

import numpy as np
import pytest

from histokit.io.slides.region import Region
from histokit.io.slides.registry import get_slide_cls, is_slide_format_supported, is_slide_extension_supported, get_slide_cls_for_path
from histokit.io.annotations.registry import get_annotation_loader
from histokit.io.annotations.annotation import AnnotationRegion, AnnotationSet
from histokit.dataset.sample import Sample
from histokit.utils.geometry import Point, Size, Shape
from conftest import needs_cervical_mini, CERVICAL_MINI


# ── Region ───────────────────────────────────────────────────────────────────

class TestRegion:
    def test_patch_factory(self):
        r = Region.patch(100, 200, 256, 1)
        assert r.level == 1
        assert r.location == Point(100, 200)
        assert r.size == Size(256, 256)

    def test_make_factory(self):
        r = Region.make(10, 20, 512, 256, 0)
        assert r.x == 10
        assert r.y == 20
        assert r.width == 512
        assert r.height == 256

    def test_area_level0(self):
        r = Region.patch(0, 0, 100, 0)
        assert r.area_level0(1.0) == 10000.0
        assert r.area_level0(2.0) == 40000.0


# ── Slide registry ───────────────────────────────────────────────────────────

class TestSlideRegistry:
    def test_tiffslide_registered(self):
        assert is_slide_format_supported("tiffslide")

    def test_ome_tiff_extension(self):
        assert is_slide_extension_supported(".ome.tiff")

    def test_unknown_format(self):
        assert not is_slide_format_supported("imaginary_format")


# ── TiffSlide reading ────────────────────────────────────────────────────────

@needs_cervical_mini
class TestTiffSlideReading:
    def test_open_and_close(self, cervical_mini_samples):
        sample = cervical_mini_samples[0]
        slide = sample.make_slide()
        with slide:
            assert slide.is_open
        assert not slide.is_open

    def test_dimensions(self, cervical_mini_samples):
        sample = cervical_mini_samples[0]
        with sample.open_slide() as slide:
            dims = slide.dimensions
            assert len(dims) >= 1
            assert dims[0].width > 0
            assert dims[0].height > 0

    def test_level_downsamples(self, cervical_mini_samples):
        sample = cervical_mini_samples[0]
        with sample.open_slide() as slide:
            ds = slide.level_downsamples()
            assert ds[0] == 1.0
            assert len(ds) == len(slide.dimensions)

    def test_read_region(self, cervical_mini_samples):
        sample = cervical_mini_samples[0]
        with sample.open_slide() as slide:
            region = Region.patch(0, 0, 256, 0)
            img = slide.read_region(region)
            assert img.size == (256, 256)

    def test_get_thumbnail(self, cervical_mini_samples):
        sample = cervical_mini_samples[0]
        with sample.open_slide() as slide:
            thumb = slide.get_thumbnail(level=len(slide.dimensions) - 1)
            assert isinstance(thumb, np.ndarray)
            assert thumb.ndim == 3
            assert thumb.shape[2] == 3

    def test_size_in_patches(self, cervical_mini_samples):
        sample = cervical_mini_samples[0]
        with sample.open_slide() as slide:
            s = slide.size_in_patches(256, 0)
            assert s.width > 0
            assert s.height > 0


# ── Annotation loading ───────────────────────────────────────────────────────

@needs_cervical_mini
class TestAnnotationLoading:
    def test_geojson_loader_registered(self):
        loader = get_annotation_loader("geojson")
        assert callable(loader)

    def test_sample_without_annotations_returns_none(self, cervical_mini_samples):
        """Sample IC-CX-00001-01 (normal_inflammation) has an annotation file but no matching features."""
        sample = cervical_mini_samples[0]
        ann = sample.make_annotations()
        # Either None or an empty AnnotationSet
        if ann is not None:
            assert len(ann) == 0

    @staticmethod
    def _sample_with_annotations(cervical_mini_samples):
        """Return the first sample that has actual annotation regions."""
        for s in cervical_mini_samples:
            ann = s.make_annotations()
            if ann is not None and len(ann) > 0:
                return s, ann
        pytest.skip("No sample with annotation regions found")

    def test_load_geojson(self, cervical_mini_samples):
        sample, ann = self._sample_with_annotations(cervical_mini_samples)
        assert isinstance(ann, AnnotationSet)
        assert len(ann) > 0

    def test_annotation_labels(self, cervical_mini_samples):
        sample, ann = self._sample_with_annotations(cervical_mini_samples)
        assert "background" in ann.label_map
        assert "normal" in ann.label_map

    def test_render(self, cervical_mini_samples):
        sample, ann = self._sample_with_annotations(cervical_mini_samples)
        rendered = ann.render(Shape(10, 10), factor=10000.0)
        assert rendered.shape == (10, 10)
        assert rendered.dtype == int

    def test_render_to_grid(self, cervical_mini_samples):
        sample, ann = self._sample_with_annotations(cervical_mini_samples)
        grid = ann.render_to_grid(Shape(5, 5), patch_size=256, patch_level=1)
        assert grid.shape == (5, 5)

    def test_bounds(self, cervical_mini_samples):
        sample, ann = self._sample_with_annotations(cervical_mini_samples)
        bounds = ann.bounds
        assert len(bounds) == 4
        assert bounds[2] > bounds[0]  # xmax > xmin
        assert bounds[3] > bounds[1]  # ymax > ymin

    def test_fill_index(self, cervical_mini_samples):
        sample, ann = self._sample_with_annotations(cervical_mini_samples)
        fill_idx = ann.get_fill_index()
        assert fill_idx == ann.label_map[ann.fill_label]


# ── AnnotationRegion geometry ────────────────────────────────────────────────

class TestAnnotationRegionGeometry:
    def test_polygon_geometry(self):
        vertices = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        region = AnnotationRegion("test", "Polygon", "label", vertices)
        geom = region.geometry
        assert geom.is_valid
        assert geom.area > 0
