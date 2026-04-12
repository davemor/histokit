from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET

from histokit.io.annotations.registry import register_annotation
from histokit.dataset.schema import AnnotationSchema
from histokit.utils.geometry import PointF

from .annotation import AnnotationRegion


def annotation_from_tag(
    tag: ET.Element, group_labels: Dict[str, int]
) -> AnnotationRegion:
    """Parse a single ASAP XML annotation tag."""

    name = tag.attrib["Name"]
    group = tag.attrib["PartOfGroup"]
    annotation_type = tag.attrib["Type"]
    coordinate_tags = tag.find("Coordinates")

    if group not in group_labels:
        raise ValueError(f"Unknown annotation group encountered: {group}")

    vertices = [
        PointF(float(c.attrib["X"]), float(c.attrib["Y"])) for c in coordinate_tags
    ]

    # Keep the semantic label name, do not map it to its integer id here.
    return AnnotationRegion(name, annotation_type, group, vertices)


@register_annotation("asap-xml", extensions=[".xml"])
def load_annotations_asapxml(
    xml_file_path: Path, schema: AnnotationSchema
) -> List[AnnotationRegion]:
    """Load ASAP XML annotations as a list of Annotation objects."""

    if not xml_file_path.is_file():
        return []

    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    tags = root.find("Annotations")

    if tags is None:
        return []

    annotations = [annotation_from_tag(tag, schema.label_map) for tag in tags]
    return annotations
