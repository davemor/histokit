from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
from histokit.utils.geometry import Point, Size


class Region(NamedTuple):
    level: int
    location: Point
    size: Size

    @classmethod
    def patch(cls, x: int, y: int, size: int, level: int) -> "Region":
        location = Point(x, y)
        region_size = Size(size, size)
        return Region(level, location, region_size)

    @classmethod
    def make(cls, x: int, y: int, width: int, height: int, level: int) -> "Region":
        location = Point(x, y)
        region_size = Size(width, height)
        return Region(level, location, region_size)


class SlideBase(metaclass=ABCMeta):
    def __init__(self, path: Path) -> None:
        self.slide_path = path
        self.is_open = False

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        assert not self.is_open, "Cannot open slide that is already open."
        self.open()
        self.is_open = True
        return self

    def __exit__(self, *args: object) -> None:
        assert self.is_open, "Cannot close slide that is not open."
        self.close()
        self.is_open = False

    @property
    def path(self) -> Path:
        return self.slide_path

    @property
    @abstractmethod
    def mpp(self) -> float | None:
        """Microns per pixel at level 0, or ``None`` if unavailable."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> list[Size]:
        raise NotImplementedError

    @abstractmethod
    def level_downsamples(self) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def read_region(self, region: Region) -> Image.Image:
        raise NotImplementedError

    @abstractmethod
    def read_regions(self, regions: list[Region]) -> list[Image.Image]:
        raise NotImplementedError
    


    def get_thumbnail(self, level: int, white_background: bool = True) -> np.ndarray:
        """Generate a thumbnail of the slide at a given pyramid level.

        If the requested level exists in the slide pyramid, this returns an image read
        directly from that level. If the requested level is deeper than the available
        pyramid, the deepest available level is read and then downsampled further.

        Args:
            level: Requested pyramid level.
            white_background: If True, replace pure black pixels with white.

        Returns:
            Thumbnail image as an RGB numpy array of shape (H, W, 3).
        """
        if level < 0:
            raise ValueError(f"level must be non-negative, got {level}")

        max_level = len(self.dimensions) - 1
        request_level = min(level, max_level)

        size = self.dimensions[request_level]
        region = Region(level=request_level, location=Point(0, 0), size=size)
        im = self.read_region(region).convert("RGB")

        if level > max_level:
            # further downsample from the deepest available level using the actual
            # level-downsample relationship rather than assuming powers of two.
            if not hasattr(self, "level_downsamples"):
                raise AttributeError(
                    "Slide object does not provide level_downsamples; cannot safely "
                    "resample beyond the deepest pyramid level."
                )

            base_downsample = float(self.level_downsamples[request_level])

            # extrapolate the requested downsample relative to level 0 using the
            # ratio implied by one extra level step past the deepest available level.
            if max_level == 0:
                step_ratio = 2.0
            else:
                prev_downsample = float(self.level_downsamples[max_level - 1])
                curr_downsample = float(self.level_downsamples[max_level])
                step_ratio = curr_downsample / prev_downsample

            requested_downsample = base_downsample * (
                step_ratio ** (level - request_level)
            )
            resize_factor = base_downsample / requested_downsample

            w, h = im.size
            new_w = max(1, int(round(w * resize_factor)))
            new_h = max(1, int(round(h * resize_factor)))
            im = im.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        new_im = np.asarray(im)

        if white_background:
            black_pixels = np.all(new_im == 0, axis=2)
            new_im = new_im.copy()
            new_im[black_pixels] = [255, 255, 255]

        return new_im

    def get_thumbnail_for_size(
        self, width: int, height: int, white_background: bool = True
    ) -> np.ndarray:
        """Generate a thumbnail resized to the given pixel dimensions.

        Selects the deepest pyramid level whose native dimensions are still at
        least ``width`` x ``height`` (to avoid reading more data than necessary
        while preventing upscaling where possible). If every level is smaller
        than the requested size, the deepest (smallest) level is used as the
        source. The result is then resized to exactly ``(width, height)``.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.
            white_background: If True, replace pure black pixels with white.

        Returns:
            Thumbnail image as an RGB numpy array of shape (height, width, 3).
        """
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive, got width={width}, height={height}"
            )

        # Find the deepest level whose dimensions still cover the requested size.
        # Levels are ordered from largest (0) to smallest (max_level), so we scan
        # from the end and take the first level that satisfies both constraints.
        best_level = len(self.dimensions) - 1  # fallback: deepest level
        for level_idx in range(len(self.dimensions) - 1, -1, -1):
            dim = self.dimensions[level_idx]
            if dim.width >= width and dim.height >= height:
                best_level = level_idx
                break

        im_array = self.get_thumbnail(best_level, white_background=white_background)
        im = Image.fromarray(im_array)
        im = im.resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(im)


    def size_in_patches(self, patch_size: int, patch_level: int) -> Size:
        # work out the size of the thumbnail we need to read in
        dims = self.dimensions[patch_level]
        
        # compute the number of patches in each dimension
        width_patches = (dims[0] + patch_size - 1) // patch_size
        height_patches = (dims[1] + patch_size - 1) // patch_size
        
        return Size(width_patches, height_patches)