from abc import ABCMeta, abstractmethod

import cv2
import numpy as np


from histokit.utils.filters import pool2d

"""
OpenCV functions that use used in the transforms:
- cv2.cvtColor: Converts an image from one color space to another.
- cv2.threshold: Applies a fixed-level threshold to each array element.
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
- cv2.medianBlur: Blurs an image using the median filter.
- cv2.morphologyEx: Performs advanced morphological transformations.
- cv2.Canny: Finds edges in an image using the Canny algorithm.
"""


class TissueTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, img: np.ndarray) -> np.ndarray:
        pass


class TissueTransforms(TissueTransform):
    def __init__(self, *args: TissueTransform) -> None:
        super().__init__()
        self.transforms = args

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            img = transform(img)
        return img


class RgbToHsv(TissueTransform):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


class Rgb2Grey(TissueTransform):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class PureBlackToPureWhite(TissueTransform):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        # image is assumed to be RGB
        channels = (img[:, :, 0] == 0, img[:, :, 1] == 0, img[:, :, 2] == 0)
        mask = np.expand_dims(np.logical_and.reduce(channels), axis=-1)
        removed = np.where(mask, [255, 255, 255], img)
        return np.array(removed, dtype=np.uint8)


class OTSU_H_S_Mask(TissueTransform):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        # image is assumed to be HSV
        _, mask_h = cv2.threshold(
            img[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, mask_s = cv2.threshold(
            img[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        np_mask = np.logical_and(mask_h, mask_s)
        return np_mask


class GreyScaleMask(TissueTransform):
    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__()
        self.threshold = threshold

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # image is assumed to be Grey
        return np.less_equal(img, self.threshold)


class MedianBlur(TissueTransform):
    def __init__(self, mthresh: int):
        self.mthresh = mthresh

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(img[:, :, 1], self.mthresh)


class MorphologicalClosing(TissueTransform):
    def __init__(self, close: int = 0):
        self.close = close

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.int8)
        if self.close > 0:
            kernel = np.ones((self.close, self.close), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img


class ThresholdOTSU(TissueTransform):
    def __init__(self, sthresh_up: int = 255):
        self.sthresh_up = sthresh_up

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _, img_otsu = cv2.threshold(
            img, 0, self.sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
        return img_otsu


class ThresholdFixed(TissueTransform):
    def __init__(self, sthresh: int = 20, sthresh_up: int = 255):
        self.sthresh = sthresh
        self.sthresh_up = sthresh_up

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _, img_thresh = cv2.threshold(
            img, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY
        )
        return img_thresh


class ToMask(TissueTransform):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.bool)


class CannyEdgeTheshold(TissueTransform):
    def __init__(self, theshold: float = 2.0) -> None:
        self.theshold = theshold

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        canny_edges = cv2.Canny(img_gray, 40, 100)  # note - hardcoded?
        max_edge = np.max(canny_edges)

        # early out
        if max_edge == 0 or img.size == 0:
            return np.array(False)

        edges = (np.sum(canny_edges / max_edge) / img.size) * 100
        return np.array(edges >= self.theshold)


class MaxPoolDownSample(TissueTransform):
    def __init__(self, features_level: int = 6, labels_level: int = 9) -> None:
        self.is_identity = labels_level == features_level
        kernel_size = 2 ** (labels_level - features_level)
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # slide_h, slide_w = img.shape
        # height, width = (slide_h - self.kernel_size) // self.kernel_size + 1, (slide_w - self.kernel_size) // self.kernel_size + 1
        if self.is_identity:
            return img
        out = pool2d(img, self.kernel_size, self.stride, pool_mode="max")
        return out
