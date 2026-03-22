from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlideSchema:
    kind: str
    label_schema: dict


@dataclass
class AnnotationSchema:
    kind: str
    label_map: dict
    cutout_label: str
    fill_label: str
    label_order: list[str]


@dataclass
class DatasetSchema:
    slides: SlideSchema
    annotations: AnnotationSchema

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSchema":
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

        d["slides"] = SlideSchema(**slide_data)
        d["annotations"] = AnnotationSchema(**annotation_data)
        return cls(**d)
