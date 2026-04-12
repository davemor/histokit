# a tissue detector goes from a slide to an array of labels for pixels of the slide at some level
# this might be a slide -> array or slide -> thumbnail -> array
# TissueTransforms go from arrays (thumbnails) to arrays
# TissueDetectors go from slides to arrays

from abc import ABCMeta, abstractmethod

import numpy as np
from tqdm import tqdm

from histokit.utils.convert import pil_to_np
from histokit.utils.filters import PoolMode, pool2d

from .transforms import (
    TissueTransform,
    TissueTransforms,
)
from histokit.io.slides import SlideBase, Region


class TissueDetector(metaclass=ABCMeta):
    def __init__(self) -> None:
        self._name: str = "Unnamed Tissue Detector"

    @abstractmethod
    def __call__(self, slide: SlideBase) -> np.ndarray:
        pass

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name


class ThumbTissueDetector(TissueDetector):
    def __init__(self,
                 patch_size: int, 
                 patch_level: int,
                 features_level: int,
                 transforms: TissueTransforms) -> None:
        super().__init__()
        assert features_level <= patch_level, "Features level must be less than or equal to patch level."
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.features_level = features_level
        self.transforms = transforms

    def __call__(self, slide: SlideBase) -> np.ndarray:
        # work out the size of the thumbnail we need to read in
        dims = slide.dimensions[self.patch_level]
        
        # compute the number of patches in each dimension
        width_patches = (dims[0] + self.patch_size) // self.patch_size
        height_patches = (dims[1] + self.patch_size) // self.patch_size

        # compute the size of the thumbnail we need to read in
        scale = 2 ** (self.patch_level - self.features_level)
        thumb_width, thumb_height = (width_patches * scale, height_patches * scale)

        # load in the thumbnail and apply the transforms
        thumb = slide.get_thumbnail_for_size(thumb_width, thumb_height)
        tissue_mask = self.transforms(thumb)

        # downsample the tissue mask if necessary
        if self.patch_level != self.features_level:
            kernel_size = 2 ** (self.patch_level - self.features_level)
            tissue_mask = pool2d(tissue_mask, kernel_size, kernel_size, pool_mode=PoolMode.MAX)
        return tissue_mask


class PatchesTissueDetector(TissueDetector):
    def __init__(
        self,
        patch_size: int,
        patch_level: int,
        transform: TissueTransform,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.transform = transform

    def __call__(self, slide: SlideBase) -> np.ndarray:
        # get the size of the wsi at the level that we want to sample at
        width_pixels, height_pixels = slide.dimensions[self.patch_level]
        downsample = float(slide.level_downsamples()[self.patch_level])

        # compute the width and height in patches
        width_patches = (width_pixels + self.patch_size) // self.patch_size
        height_patches = (height_pixels + self.patch_size) // self.patch_size

        # compute the regions — locations must be in level 0 coordinates
        indices, regions = zip(
            *[
                (
                    (iy, ix),
                    Region.patch(
                        int(ix * self.patch_size * downsample),
                        int(iy * self.patch_size * downsample),
                        self.patch_size,
                        self.patch_level,
                    ),
                )
                for ix in range(0, width_patches)
                for iy in range(0, height_patches)
            ]
        )

        # read in all the regions and apply the transforms
        transformed_patches = [
            self.transform(pil_to_np(p)) for p in tqdm(slide.read_regions(regions))
        ]

        arr = np.full((height_patches, width_patches), -1.0)
        for (col, row), patch in zip(indices, transformed_patches):
            arr[col, row] = patch

        return arr


