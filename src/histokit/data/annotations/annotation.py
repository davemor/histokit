from typing import List, Dict

import cv2
import numpy as np

from histokit.utils.geometry import PointF, Shape

annotation_types = ["Dot", "Polygon", "Spline", "Rectangle"]


class Annotation:
    def __init__(
        self, name: str, annotation_type: str, label: str, vertices: List[PointF]
    ):
        assert annotation_type in annotation_types
        self.name = name
        self.type = annotation_type
        self.label = label
        self.coordinates = vertices

    def draw(self, image: np.array, labels: Dict[str, int], factor: float):
        fill_colour = labels[self.label]
        vertices = np.array(self.coordinates) / factor
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(image, [vertices], (fill_colour))

    def __str__(self) -> np.str:
        return f"name: {self.name}, type: {self.type}, label: {self.label}"


class AnnotationSet:
    def __init__(
        self,
        annotations: List[Annotation],
        labels: Dict[str, int],
        labels_order: List[str],
        fill_label: str,
    ) -> None:
        assert all(label in labels for label in labels_order), (
            "All labels in labels_order must be keys in labels dictionary."
        )

        annotation_labels = {a.label for a in annotations}
        missing_from_order = annotation_labels - set(labels_order)
        if missing_from_order:
            raise ValueError(
                "These annotation labels are missing from labels_order: "
                f"{sorted(missing_from_order)}"
            )

        missing_from_labels = annotation_labels - set(labels)
        if missing_from_labels:
            raise ValueError(
                "These annotation labels are missing from labels: "
                f"{sorted(missing_from_labels)}"
            )

        if fill_label not in labels:
            raise ValueError(f"fill_label '{fill_label}' is not in labels.")

        self.annotations = annotations
        self.labels = labels
        self.labels_order = labels_order
        self.fill_label = fill_label

    def render(self, shape: Shape, factor: float) -> np.array:
        assert len(shape) == 2, "Annotations must be rendered onto a 2D array."
        annotations = sorted(
            self.annotations, key=lambda a: self.labels_order.index(a.label)
        )
        image = np.full(shape, self.labels[self.fill_label], dtype=float)
        for a in annotations:
            a.draw(image, self.labels, factor)
        return image.astype("int")

    def __str__(self) -> np.str:
        rtn = f"labels: {self.labels}" + f"\nfill_label: {self.fill_label}"
        rtn += "\nannotations:"
        for annot in self.annotations:
            rtn += "\n\t" + str(annot)
        return rtn
