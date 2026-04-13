from typing import List, Dict

import cv2
import numpy as np
from shapely.geometry import Polygon, box

from histokit.dataset.schema import AnnotationSchema
from histokit.utils.geometry import PointF, Shape

annotation_types = ["Dot", "Polygon", "Spline", "Rectangle"]


class AnnotationRegion:
    def __init__(
        self, name: str, annotation_type: str, label: str, vertices: List[PointF]
    ):
        assert annotation_type in annotation_types
        self.name = name
        self.type = annotation_type
        self.label = label
        self.coordinates = vertices

    def draw(self, image: np.ndarray, labels: Dict[str, int], factor: float):
        fill_colour = labels[self.label]
        vertices = np.array(self.coordinates) / factor
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(image, [vertices], (fill_colour))

    @property
    def geometry(self) -> Polygon:
        coords = [(p[0], p[1]) for p in self.coordinates]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly


class AnnotationSet:
    def __init__(
        self,
        annotations: List[AnnotationRegion],
        schema: AnnotationSchema,
    ) -> None:
        self.annotations: List[AnnotationRegion] = annotations
        self.label_map: Dict[str, int] = schema.label_map
        self.labels_order: List[str] = schema.label_order
        self.fill_label: str = schema.fill_label

    def __repr__(self) -> str:
        return f"AnnotationSet(num_annotations={len(self.annotations)}, labels={list(self.label_map.keys())})"

    def __len__(self) -> int:
        return len(self.annotations)

    def render(self, shape: Shape, factor: float) -> np.ndarray:
        assert len(shape) == 2, "Annotations must be rendered onto a 2D array."
        annotations = sorted(
            self.annotations, key=lambda a: self.labels_order.index(a.label)
        )
        image = np.full(shape, self.label_map[self.fill_label], dtype=float)
        for a in annotations:
            a.draw(image, self.label_map, factor)
        return image.astype("int")
    
    def bounding_box(self) -> Polygon:
        assert len(self.annotations) > 0, "Cannot compute bounding box of empty annotation set."
        geometries = [a.geometry for a in self.annotations]
        union = geometries[0]
        for g in geometries[1:]:
            union = union.union(g)
        return box(*union.bounds)
    
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.bounding_box().bounds
    
    def get_fill_index(self) -> int:
        return self.label_map[self.fill_label]

    def render_as_grid(self, patch_size: int, patch_level: int) -> np.ndarray:
        factor = patch_size * (2 ** patch_level)
        shape = (int(self.bounds[3] / factor) + 1, int(self.bounds[2] / factor) + 1)
        return self.render(Shape(*shape), factor)

    def render_to_grid(self, shape: Shape, patch_size: int, patch_level: int) -> np.ndarray:
        factor = patch_size * (2 ** patch_level)
        return self.render(shape, factor)