"""GrandQC tissue detection (Schoemig-Markiefka et al., Nature Comms 2024).

Requires optional dependencies: ``pip install histokit[grandqc]``

Reference
---------
- Paper: https://www.nature.com/articles/s41467-024-54769-y
- Code:  https://github.com/cpath-ukk/grandqc
- Weights: https://zenodo.org/records/14507273
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from histokit.io.slides.slide import SlideBase
from histokit.segmentation.detector import TissueDetector
from histokit.segmentation.registry import register_tissue_detector

import torch
import segmentation_models_pytorch as smp

# ── constants ───────────────────────────────────────────────────────────

_GRANDQC_MPP = 10.0
_TILE_SIZE = 512
_JPEG_QUALITY = 80

# ImageNet normalisation (matches timm-efficientnet-b0 training)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_ZENODO_URL = "https://zenodo.org/records/14507273"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "histokit" / "grandqc"
_DEFAULT_MODEL_NAME = "Tissue_Detection_MPP10.pth"


# ── helpers ─────────────────────────────────────────────────────────────


def _default_model_path() -> Path:
    env = os.environ.get("HISTOKIT_GRANDQC_MODEL_DIR")
    if env:
        return Path(env) / _DEFAULT_MODEL_NAME
    return _DEFAULT_CACHE_DIR / _DEFAULT_MODEL_NAME


def _get_device() -> torch.device:
    env = os.environ.get("HISTOKIT_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=2)
def _load_model(
    model_path: str, device: str
) -> torch.nn.Module:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"GrandQC tissue model weights not found at {path}. "
            f"Download from {_ZENODO_URL} and place at "
            f"{_default_model_path()}, or set the "
            f"HISTOKIT_GRANDQC_MODEL_DIR environment variable."
        )
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=2,
        activation=None,
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _jpeg_compress(
    img: np.ndarray, quality: int = _JPEG_QUALITY
) -> np.ndarray:
    """Re-compress through JPEG to match training distribution."""
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf))


def _preprocess_tile(tile: np.ndarray) -> np.ndarray:
    """ImageNet normalisation, HWC uint8 -> CHW float32."""
    arr = tile.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1)  # CHW


# ── detector ────────────────────────────────────────────────────────────


class GrandQCTissueDetector(TissueDetector):
    """Tissue detector using the GrandQC UNet++ model.

    Produces a 2-D array of tissue probabilities (0.0 = background,
    1.0 = tissue) aligned to the patch grid defined by *patch_size*
    and *patch_level*.
    """

    def __init__(
        self,
        patch_size: int,
        patch_level: int,
        model_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.model_path = model_path or _default_model_path()

    def __call__(self, slide: SlideBase) -> np.ndarray:
        # 1. Compute thumbnail dimensions at MPP 10
        slide_mpp = slide.mpp
        if slide_mpp is None:
            raise ValueError(
                "Slide does not expose MPP metadata, which GrandQC "
                "requires. Check the slide file properties."
            )

        device = _get_device()
        model = _load_model(str(self.model_path), str(device))

        level0 = slide.dimensions[0]
        w_mpp10 = max(1, int(round(level0.width * slide_mpp / _GRANDQC_MPP)))
        h_mpp10 = max(1, int(round(level0.height * slide_mpp / _GRANDQC_MPP)))

        # 2. Read thumbnail
        thumb = slide.get_thumbnail_for_size(
            w_mpp10, h_mpp10, white_background=True
        )

        # 3. Pad to a multiple of 512 (white fill)
        pad_h = (-h_mpp10) % _TILE_SIZE
        pad_w = (-w_mpp10) % _TILE_SIZE
        if pad_h or pad_w:
            thumb = np.pad(
                thumb,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=255,
            )

        padded_h, padded_w = thumb.shape[:2]
        n_rows = padded_h // _TILE_SIZE
        n_cols = padded_w // _TILE_SIZE

        # 4. Tile -> JPEG compress -> preprocess -> infer -> stitch
        tissue_mask = np.zeros(
            (padded_h, padded_w), dtype=np.float32
        )

        with torch.no_grad():
            for r in range(n_rows):
                for c in range(n_cols):
                    y0 = r * _TILE_SIZE
                    x0 = c * _TILE_SIZE
                    tile = thumb[
                        y0 : y0 + _TILE_SIZE,
                        x0 : x0 + _TILE_SIZE,
                    ]
                    tile = _jpeg_compress(tile)
                    preprocessed = _preprocess_tile(tile)
                    tensor = (
                        torch.from_numpy(preprocessed)
                        .unsqueeze(0)
                        .to(device)
                    )
                    logits = model(tensor)
                    # logits: (1, 2, 512, 512)  class 0=tissue, 1=bg
                    probs = torch.softmax(logits, dim=1)
                    tissue_prob = probs[0, 0].cpu().numpy()
                    tissue_mask[
                        y0 : y0 + _TILE_SIZE,
                        x0 : x0 + _TILE_SIZE,
                    ] = tissue_prob

        # 5. Crop back to original thumbnail size
        tissue_mask = tissue_mask[:h_mpp10, :w_mpp10]

        # 6. Downsample to patch grid via BOX resampling (area average)
        grid = slide.size_in_patches(self.patch_size, self.patch_level)
        mask_img = Image.fromarray(tissue_mask, mode="F")
        mask_resized = mask_img.resize(
            (grid.width, grid.height),
            resample=Image.Resampling.BOX,
        )
        return np.asarray(mask_resized, dtype=np.float64)


# ── registration ────────────────────────────────────────────────────────


@register_tissue_detector("grandqc_tissue")
def grandqc_tissue(
    patch_size: int,
    patch_level: int,
    **_kwargs: object,
) -> GrandQCTissueDetector:
    return GrandQCTissueDetector(
        patch_size=patch_size,
        patch_level=patch_level,
    )
