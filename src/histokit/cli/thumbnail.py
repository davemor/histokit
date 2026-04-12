from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
from PIL import Image

from histokit.io.slides.registry import get_slide_cls_for_path
from histokit.io.annotations.registry import get_annotation_loader_for_path
from histokit.io.annotations.annotation import AnnotationSet
from histokit.dataset.schema import DatasetSchema
from histokit.utils.geometry import Shape
import histokit.io.slides  # noqa: F401 — triggers slide backend registration
import histokit.io.annotations  # noqa: F401 — triggers annotation loader registration


def overlay_annotations(
    thumb: np.ndarray,
    annotation_set: AnnotationSet,
    slide_width: int,
    slide_height: int,
    alpha: float = 0.3,
) -> np.ndarray:
    """Render annotations and blend them onto the thumbnail."""
    thumb_h, thumb_w = thumb.shape[:2]
    factor_x = slide_width / thumb_w
    factor_y = slide_height / thumb_h
    factor = (factor_x + factor_y) / 2

    mask = annotation_set.render(Shape(thumb_h, thumb_w), factor)

    fill_val = annotation_set.label_map[annotation_set.fill_label]
    label_values = sorted(annotation_set.label_map.values())
    cmap = plt.get_cmap("tab10", len(label_values))
    colours = {v: (np.array(cmap(i)[:3]) * 255).astype(np.uint8) for i, v in enumerate(label_values)}

    overlay = thumb.copy()
    for label_val, colour in colours.items():
        if label_val == fill_val:
            continue
        region = mask == label_val
        if region.any():
            overlay[region] = colour

    blended = (thumb.astype(float) * (1 - alpha) + overlay.astype(float) * alpha).astype(np.uint8)
    return blended


def thumbnail(
    slide_path: Annotated[Path, typer.Argument(help="Path to the whole-slide image.")],
    level: Annotated[
        int,
        typer.Argument(min=0, max=9, help="Magnification level (0-9) for the thumbnail."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for the thumbnail image."),
    ] = None,
    annotations: Annotated[
        Optional[Path],
        typer.Option("--annotations", "-a", help="Path to annotation file to overlay."),
    ] = None,
    labels: Annotated[
        Optional[Path],
        typer.Option("--labels", "-l", help="Path to dataset labels.json schema file."),
    ] = None,
) -> None:
    """Generate a thumbnail of a slide at a given magnification level."""
    if output is None:
        output = Path(f"{slide_path.stem}_thumb_L{level}.png")

    slide_cls = get_slide_cls_for_path(slide_path)
    with slide_cls(slide_path) as slide:
        thumb = slide.get_thumbnail(level)
        slide_width, slide_height = slide.dimensions[0]

    if annotations is not None:
        if labels is None:
            raise typer.BadParameter("--labels is required when --annotations is provided.")
        schema = DatasetSchema.from_json(labels).annotations
        load_annotation = get_annotation_loader_for_path(annotations)
        annotation_regions = load_annotation(annotations, schema)
        annotation_set = AnnotationSet(annotation_regions, schema)
        thumb = overlay_annotations(thumb, annotation_set, slide_width, slide_height)

    Image.fromarray(thumb).save(output)
    typer.echo(f"Saved thumbnail to {output}")
