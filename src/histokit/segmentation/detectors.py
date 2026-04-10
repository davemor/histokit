# a tissue detector goes from a slide to an array of labels for pixels of the slide at some level
# this might be a slide -> array or slide -> thumbnail -> array
# TissueTransforms go from arrays (thumbnails) to arrays
# TissueDetectors go from slides to arrays

from abc import ABCMeta, abstractmethod

import numpy as np
from tqdm import tqdm

from histokit.utils.convert import pil_to_np

from .transforms import (
    CannyEdgeTheshold,
    MaxPoolDownSample,
    MedianBlur,
    MorphologicalClosing,
    OTSU_H_S_Mask,
    PureBlackToPureWhite,
    RgbToHsv,
    ThresholdOTSU,
    TissueTransform,
    TissueTransforms,
    ToMask,
)
from histokit.data.slides import SlideBase, Region


class TissueDetector(metaclass=ABCMeta):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, slide: SlideBase) -> np.ndarray:
        pass


class ThumbTissueDetector(TissueDetector):
    def __init__(self, name: str, features_level: int, transforms: TissueTransforms) -> None:
        super().__init__(name)
        self.features_level = features_level
        self.transforms = transforms

    def __call__(self, slide: SlideBase) -> np.ndarray:
        thumb = slide.get_thumbnail(self.features_level)
        tissue_mask = self.transforms(thumb)
        return tissue_mask


class PatchesTissueDetector(TissueDetector):
    def __init__(
        self,
        name: str,
        transform: TissueTransform,
        patch_level: int = 1,
        patch_size: int = 224,
    ) -> None:
        super().__init__(name)
        self.transform = transform
        self.patch_level = patch_level
        self.patch_size = patch_size

    def __call__(self, slide: SlideBase) -> np.ndarray:
        # get the size of the wsi at the level that we want to sample at
        width_pixels, height_pixels = slide.dimensions[self.patch_level]

        # compute the width and height in patches
        width_patches = width_pixels // self.patch_size + self.patch_size
        height_patches = height_pixels // self.patch_size + self.patch_size

        # compute the regions
        indices, regions = zip(
            *[
                (
                    (iy, ix),
                    Region.patch(
                        ix * self.patch_size,
                        iy * self.patch_size,
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

        arr = np.full((height_patches, width_patches), -1)
        for (col, row), patch in zip(indices, transformed_patches):
            arr[col, row] = patch

        return arr > 0


# Define some tissue detectors


def clam_segmentation(features_level: int, labels_level: int, sthresh: int = 20, mthresh: int = 7, close: int = 0) -> ThumbTissueDetector:
    return ThumbTissueDetector(
        "clam_segmentation",
        features_level,
        TissueTransforms(
            PureBlackToPureWhite(),
            RgbToHsv(),
            MedianBlur(mthresh=mthresh),
            ThresholdFixed(sthresh=sthresh),
            MorphologicalClosing(close),
            ToMask(),
            MaxPoolDownSample(features_level, labels_level),
        ),
    )


def clam_segmentation_otsu(features_level: int, labels_level: int, close: int = 0) -> ThumbTissueDetector:
    return ThumbTissueDetector(
        "clam_segmentation_otsu",
        features_level,
        TissueTransforms(
            PureBlackToPureWhite(),
            RgbToHsv(),
            MedianBlur(mthresh=7),
            ThresholdOTSU(sthresh_up=255),
            MorphologicalClosing(close),
            ToMask(),
            MaxPoolDownSample(features_level, labels_level),
        ),
    )


def otsu_hs_segmentation(features_level: int, labels_level: int) -> ThumbTissueDetector:
    return ThumbTissueDetector(
        "otsu_hs_segmentation",
        features_level,
        TissueTransforms(
            PureBlackToPureWhite(),
            RgbToHsv(),
            OTSU_H_S_Mask(),
            ToMask(),
            MaxPoolDownSample(features_level, labels_level),
        ),
    )


def per_patch_canny_segmentation(patch_level: int = 1, patch_size: int = 224) -> PatchesTissueDetector:
    return PatchesTissueDetector(
        "per_patch_canny_segmentation",
        CannyEdgeTheshold(),
        patch_level=patch_level,
        patch_size=patch_size,
    )
