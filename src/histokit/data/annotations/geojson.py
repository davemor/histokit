from __future__ import annotations

import json
import re
from pathlib import Path

from histokit.data.annotations.annotation import Annotation
from histokit.data.annotations.registry import register_annotation
from histokit.data.schema import AnnotationSchema


def json_load(filepath: Path, **kwargs) -> dict:
    with open(filepath, "r") as f:
        return json.load(f, **kwargs)


def base_shape(coord_list: list) -> list[tuple[float, float]]:
    return [(float(c[0]), float(c[1])) for c in coord_list]


def gjson_polygon(
    polygon: list,
    label: str,
    cutout_label: str,
) -> list[Annotation]:
    polygon_vertices = [base_shape(ring) for ring in polygon]

    outer_polygon = Annotation(label, "Polygon", label, polygon_vertices[0])
    annotations = [outer_polygon]

    for poly in polygon_vertices[1:]:
        inner_polygon = Annotation(label, "Polygon", cutout_label, poly)
        annotations.append(inner_polygon)

    return annotations


def standardise_label(input_string: str) -> str:
    lowercased_string = input_string.lower()
    return re.sub(r"[^a-z0-9]", "_", lowercased_string)


def annotation_from_feature(
    feature: dict,
    labels: dict[str, int],
    cutout_label: str,
) -> list[Annotation]:
    geometry = feature["geometry"]
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]
    properties = feature["properties"]

    if "classification" not in properties:
        raise ValueError("Unlabelled annotation.")

    classification = properties["classification"]

    print(classification)

    label = standardise_label(classification["name"])

    if label not in labels:
        raise ValueError(f"Unknown annotation group: {label}")

    if geometry_type == "Polygon":
        return gjson_polygon(coordinates, label, cutout_label)

    if geometry_type == "MultiPolygon":
        annotations: list[Annotation] = []
        for polygon in coordinates:
            annotations.extend(gjson_polygon(polygon, label, cutout_label))
        return annotations

    raise ValueError(f"Unknown geometry type encountered: {geometry_type}")


@register_annotation("geojson")
def load_annotations_geojson(
    json_path: Path,
    schema: AnnotationSchema,
) -> list[Annotation]:
    file_in = json_load(json_path)
    features = file_in["features"]

    annotations: list[Annotation] = []
    for feature in features:
        annotations.extend(
            annotation_from_feature(feature, schema.label_map, schema.cutout_label)
        )

    return annotations
