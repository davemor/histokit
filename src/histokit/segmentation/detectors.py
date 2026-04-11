from histokit.segmentation.detector import PatchesTissueDetector, ThumbTissueDetector

from .transforms import (
    CannyEdgeTheshold,
    MaxPoolDownSample,
    MedianBlur,
    MorphologicalClosing,
    OTSU_H_S_Mask,
    PureBlackToPureWhite,
    RgbToHsv,
    ThresholdOTSU,
    TissueTransforms,
    ToMask,
)

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
