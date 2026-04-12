from histokit.pipelines.segmentation.detector import PatchesTissueDetector, ThumbTissueDetector

from .registry import register_tissue_detector
from .transforms import (
    CannyEdgeDetector,
    GreaterThan,
    MedianBlur,
    MorphologicalClosing,
    OTSU_H_S_Mask,
    PureBlackToPureWhite,
    RgbToHsv,
    ThresholdFixed,
    ThresholdOTSU,
    TissueTransforms,
    ToMask,
)


@register_tissue_detector("clam_segmentation")
def clam_segmentation(patch_size: int, patch_level: int, features_level: int, sthresh: int = 20, mthresh: int = 7, close: int = 0) -> ThumbTissueDetector:
    detector = ThumbTissueDetector(
        patch_size,
        patch_level,
        features_level,
        TissueTransforms(
            PureBlackToPureWhite(),
            RgbToHsv(),
            MedianBlur(mthresh=mthresh),
            ThresholdFixed(sthresh=sthresh),
            MorphologicalClosing(close),
            ToMask(),
        ),
    )
    return detector


@register_tissue_detector("clam_segmentation_otsu")
def clam_segmentation_otsu(patch_size: int, patch_level: int, features_level: int, close: int = 0) -> ThumbTissueDetector:
    detector = ThumbTissueDetector(
        patch_size,
        patch_level,
        features_level,
        TissueTransforms(
            PureBlackToPureWhite(),
            RgbToHsv(),
            MedianBlur(mthresh=7),
            ThresholdOTSU(sthresh_up=255),
            MorphologicalClosing(close),
            ToMask(),
        ),
    )
    return detector


@register_tissue_detector("otsu_hs_segmentation")
def otsu_hs_segmentation(patch_size: int, patch_level: int, features_level: int) -> ThumbTissueDetector:
    detector = ThumbTissueDetector(
        patch_size,
        patch_level,
        features_level,
        TissueTransforms(
            PureBlackToPureWhite(),
            RgbToHsv(),
            OTSU_H_S_Mask(),
            ToMask(),
        ),
    )
    return detector


@register_tissue_detector("per_patch_canny_segmentation")
def per_patch_canny_segmentation(patch_size: int, patch_level: int) -> PatchesTissueDetector:
    detector = PatchesTissueDetector(
        patch_size,
        patch_level,
        TissueTransforms(
            CannyEdgeDetector(),
            GreaterThan(threshold=0.02)
        ),        
    )
    return detector


@register_tissue_detector("per_patch_canny_ranker")
def per_patch_canny_ranker(patch_size: int, patch_level: int) -> PatchesTissueDetector:
    detector = PatchesTissueDetector(
        patch_size,
        patch_level,
        TissueTransforms(
            CannyEdgeDetector()
        ),        
    )
    return detector