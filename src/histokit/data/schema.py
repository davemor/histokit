from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class SlideSchema:
    kind: str
    label_schema: dict[str, int]


@dataclass
class AnnotationSchema:
    kind: str
    label_map: dict[str, int]
    cutout_label: str
    fill_label: str
    label_order: list[str]


@dataclass
class DatasetSchema:
    slides: SlideSchema
    annotations: AnnotationSchema

    @classmethod
    def from_json(cls, json_path: Path) -> "DatasetSchema":
        with open(json_path, "r") as file:
            schema_dict = json.load(file)
        return cls.from_dict(schema_dict)

    @classmethod
    def from_dict(cls, d: dict[str, dict[str, Any]]) -> "DatasetSchema":
        d = d.copy()

        slide_data = d["slides"].copy()
        annotation_data = d["annotations"].copy()

        if "type" in slide_data:
            slide_data["kind"] = slide_data.pop("type")
        if "labellings" in slide_data:
            slide_data["label_schema"] = slide_data.pop("labellings")

        if "type" in annotation_data:
            annotation_data["kind"] = annotation_data.pop("type")
        if "labels" in annotation_data:
            annotation_data["label_map"] = annotation_data.pop("labels")
        if "cutout" in annotation_data:
            annotation_data["cutout_label"] = annotation_data.pop("cutout")
        if "fill" in annotation_data:
            annotation_data["fill_label"] = annotation_data.pop("fill")
        if "order" in annotation_data:
            annotation_data["label_order"] = annotation_data.pop("order")

        slides = SlideSchema(**slide_data)
        annotations = AnnotationSchema(**annotation_data)
        return cls(slides=slides, annotations=annotations)
