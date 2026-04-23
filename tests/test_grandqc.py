"""Tests for GrandQC tissue detection integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import needs_cervical_mini

slow = pytest.mark.slow

_has_torch = pytest.importorskip.__module__ and True  # always True, just a namespace anchor
try:
    import torch
    import segmentation_models_pytorch as smp

    _has_torch = True
except ImportError:
    _has_torch = False

needs_torch = pytest.mark.skipif(
    not _has_torch, reason="torch/smp not installed"
)


# ── Tests that work without torch ───────────────────────────────────────


class TestRegistryWithoutGrandQC:
    def test_core_detectors_always_registered(self):
        from histokit.segmentation.registry import (
            detector_registry,
        )

        assert "per_patch_canny_ranker" in detector_registry

    def test_grandqc_absent_gives_clear_error(self):
        from histokit.segmentation.registry import (
            get_detector,
            detector_registry,
        )

        if "grandqc_tissue" not in detector_registry:
            with pytest.raises(ValueError, match="not found"):
                get_detector("grandqc_tissue")


# ── Tests requiring torch (skip otherwise) ──────────────────────────────


@needs_torch
class TestGrandQCRegistered:
    def test_grandqc_tissue_in_registry(self):
        from histokit.segmentation.registry import (
            detector_registry,
        )

        assert "grandqc_tissue" in detector_registry


@needs_torch
class TestJpegCompress:
    def test_roundtrip_preserves_shape(self):
        from histokit.segmentation.grandqc import _jpeg_compress

        img = np.random.randint(
            0, 255, (512, 512, 3), dtype=np.uint8
        )
        out = _jpeg_compress(img)
        assert out.shape == (512, 512, 3)
        assert out.dtype == np.uint8

    def test_white_image_stays_white(self):
        from histokit.segmentation.grandqc import _jpeg_compress

        img = np.full((512, 512, 3), 255, dtype=np.uint8)
        out = _jpeg_compress(img)
        assert out.mean() > 250.0


@needs_torch
class TestPreprocessTile:
    def test_output_shape_and_dtype(self):
        from histokit.segmentation.grandqc import _preprocess_tile

        tile = np.random.randint(
            0, 255, (512, 512, 3), dtype=np.uint8
        )
        out = _preprocess_tile(tile)
        assert out.shape == (3, 512, 512)
        assert out.dtype == np.float32

    def test_matches_smp_preprocessing(self):
        """Verify inline constants match smp's preprocessing."""
        from histokit.segmentation.grandqc import _preprocess_tile

        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            "timm-efficientnet-b0", "imagenet"
        )

        tile = np.random.randint(
            0, 255, (512, 512, 3), dtype=np.uint8
        )

        ours = _preprocess_tile(tile)

        smp_out = preprocessing_fn(tile.copy())
        smp_out = smp_out.astype(np.float32).transpose(2, 0, 1)

        np.testing.assert_allclose(ours, smp_out, atol=1e-5)


class TestTilingMath:
    def test_padding_to_512_multiple(self):
        h, w = 700, 1000
        pad_h = (-h) % 512
        pad_w = (-w) % 512
        assert (h + pad_h) % 512 == 0
        assert (w + pad_w) % 512 == 0

    def test_no_padding_when_exact(self):
        h, w = 1024, 512
        pad_h = (-h) % 512
        pad_w = (-w) % 512
        assert pad_h == 0
        assert pad_w == 0


@needs_torch
class TestGrandQCDetectorWithMockModel:
    def _make_mock_slide(self, width=5000, height=4000, mpp=0.25):
        from histokit.utils.geometry import Size

        slide = MagicMock()
        slide.mpp = mpp
        slide.dimensions = [Size(width, height)]

        grid_w = (width // 4 + 256 - 1) // 256
        grid_h = (height // 4 + 256 - 1) // 256
        slide.size_in_patches.return_value = Size(grid_w, grid_h)

        def fake_thumbnail(w, h, white_background=True):
            return np.full((h, w, 3), 200, dtype=np.uint8)

        slide.get_thumbnail_for_size.side_effect = fake_thumbnail
        return slide

    def test_output_shape_matches_grid(self, monkeypatch):
        from histokit.segmentation.grandqc import (
            GrandQCDetector,
        )

        def fake_load(model_path, device):
            model = MagicMock()

            def forward(x):
                b, c, h, w = x.shape
                logits = torch.zeros(b, 2, h, w)
                logits[:, 0] = 1.0
                logits[:, 1] = -1.0
                return logits

            model.side_effect = forward
            return model

        monkeypatch.setattr(
            "histokit.segmentation.grandqc._load_model",
            fake_load,
        )

        slide = self._make_mock_slide()
        detector = GrandQCDetector(
            patch_size=256,
            patch_level=2,
            model_path=Path("/fake/path"),
        )
        result = detector(slide)

        grid = slide.size_in_patches(256, 2)
        assert result.shape == (grid.height, grid.width)
        assert result.dtype == np.float64

    def test_tissue_probability_range(self, monkeypatch):
        from histokit.segmentation.grandqc import (
            GrandQCDetector,
        )

        def fake_load(model_path, device):
            model = MagicMock()

            def forward(x):
                b, c, h, w = x.shape
                return torch.randn(b, 2, h, w)

            model.side_effect = forward
            return model

        monkeypatch.setattr(
            "histokit.segmentation.grandqc._load_model",
            fake_load,
        )

        slide = self._make_mock_slide()
        detector = GrandQCDetector(
            patch_size=256,
            patch_level=2,
            model_path=Path("/fake/path"),
        )
        result = detector(slide)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_raises_without_mpp(self, monkeypatch):
        from histokit.segmentation.grandqc import (
            GrandQCDetector,
        )

        slide = self._make_mock_slide()
        slide.mpp = None

        detector = GrandQCDetector(
            patch_size=256,
            patch_level=2,
            model_path=Path("/fake/path"),
        )

        with pytest.raises(ValueError, match="MPP"):
            detector(slide)


# ── Integration tests (require model weights + data) ────────────────────

MODEL_PATH = Path(
    "/home/cggm1/grandqc/model_weights/tissue/"
    "Tissue_Detection_MPP10.pth"
)

needs_grandqc_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="GrandQC tissue model weights not available",
)


@slow
@needs_torch
@needs_cervical_mini
@needs_grandqc_model
class TestGrandQCIntegration:
    def test_detector_on_cervical_slide(
        self, cervical_mini_samples
    ):
        from histokit.segmentation.grandqc import (
            GrandQCDetector,
        )

        sample = cervical_mini_samples[0]
        detector = GrandQCDetector(
            patch_size=256,
            patch_level=1,
            model_path=MODEL_PATH,
        )
        with sample.open_slide() as slide:
            result = detector(slide)
            grid = slide.size_in_patches(256, 1)

        assert result.shape == (grid.height, grid.width)
        assert 0.0 <= result.min()
        assert result.max() <= 1.0
        assert result.mean() > 0.01
        assert result.mean() < 0.99

    def test_pipeline_with_grandqc(
        self, cervical_mini_dataset
    ):
        import os

        from histokit.pipelines.stages import (
            Grid,
            TissueMask,
            FilterPatches,
        )
        from histokit.patchset.patchset import PatchSet

        os.environ["HISTOKIT_GRANDQC_MODEL_DIR"] = str(
            MODEL_PATH.parent
        )

        pipeline = (
            Grid(level=1, patch_size=256)
            >> TissueMask(method="grandqc_tissue")
            >> FilterPatches(
                tissue_threshold=0.5,
                drop_background=True,
            )
        )

        sample = next(cervical_mini_dataset.samples())

        from histokit.pipelines.runtime import RuntimeContext

        runtime = RuntimeContext(
            params={},
            pipeline_name="grandqc_test",
        )
        result = pipeline._run_one(sample, runtime)

        assert isinstance(result, PatchSet)
        assert "tissue_score" in result.frame.columns
        assert "keep" in result.frame.columns
        scores = result.frame["tissue_score"]
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()
