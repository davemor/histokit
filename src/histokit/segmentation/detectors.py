# a tissue detector goes from a slide to an array of labels for pixels of the slide at some level
# this might be a slide -> array or slide -> thumbnail -> array
# TissueTransforms go from arrays (thumbnails) to arrays
# TissueDetectors go from slides to arrays

from abc import ABCMeta

from .transforms import *
from histokit.data.slides import SlideBase


class TissueDetector(metaclass=ABCMeta):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, slide: SlideBase) -> np.array:
        pass


class ThumbTissueDetector(TissueDetector):
    def __init__(self, name, features_level: int, transforms: TissueTransforms):
        super().__init__(name)
        self.features_level = features_level
        self.transforms = transforms

    def __call__(self, slide: SlideBase) -> np.array:
        thumb = slide.get_thumbnail(self.features_level)
        tissue_mask = self.transforms(thumb)
        return tissue_mask


class PatchesTissueDetector(TissueDetector):
    def __init__(
        self,
        name,
        transform: TissueTransform,
        patch_level: int = 1,
        features_level: int = 6,
        labels_level: int = 9,
    ) -> None:
        super().__init__(name)
        self.transform = transform
        self.patch_level = patch_level
        self.features_level = features_level
        self.labels_level = labels_level
        self.patch_size = 2 ** (labels_level - patch_level)

    def __call__(self, slide: SlideBase) -> np.array:
        # render the annotations at features level
        slide_w, slide_h = slide.dimensions[self.features_level]

        # compute the size as if we had max pooled the annotations down to labels level
        kernel_size = 2 ** (self.labels_level - self.features_level)
        height, width = (
            (slide_h - kernel_size) // kernel_size + 1,
            (slide_w - kernel_size) // kernel_size + 1,
        )

        # work out the regions
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
                for ix in range(0, width)
                for iy in range(0, height)
            ]
        )

        # read in all the regions and apply the transforms
        transformed_patches = [
            self.transform(pil_to_np(p)) for p in slide.read_regions(regions)
        ]

        arr = np.full((height, width), -1)
        for (col, row), patch in zip(indices, transformed_patches):
            arr[col, row] = patch

        return arr > 0


# Define some tissue detectors


def clam_segmentation(features_level, labels_level, sthresh=20, mthresh=7, close=0):
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


def clam_segmentation_otsu(features_level, labels_level, close=0):
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


def otsu_hs_segmentation(features_level, labels_level):
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


def per_patch_canny_segmentation(patch_level=1, features_level=6, labels_level=9):
    return PatchesTissueDetector(
        "per_patch_canny_segmentation",
        CannyEdgeTheshold(),
        patch_level=patch_level,
        features_level=features_level,
        labels_level=labels_level,
    )
