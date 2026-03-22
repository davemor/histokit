from abc import ABCMeta, abstractmethod
import copy
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from PIL import Image
from histokit.utils.geometry import Point, Size


class Region(NamedTuple):
    level: int
    location: Point
    size: Size

    @classmethod
    def patch(cls, x, y, size, level):
        location = Point(x, y)
        size = Size(size, size)
        return Region(level, location, size)

    @classmethod
    def make(cls, x, y, width, height, level):
        location = Point(x, y)
        size = Size(width, height)
        return Region(level, location, size)


class SlideBase(metaclass=ABCMeta):
    def __init__(self):
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

    def __exit__(self, *args):
        assert self.is_open, "Cannot close slide that is not open."
        self.close()
        self.is_open = False

    @property
    @abstractmethod
    def path(self) -> Path:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> List[Size]:
        raise NotImplementedError

    @abstractmethod
    def level_downsamples(self) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def read_region(self, region: Region) -> Image.Image:
        raise NotImplementedError

    @abstractmethod
    def read_regions(self, regions: List[Region]) -> Image.Image:
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
