from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

from histokit.data.slides.registry import get_slide_cls_for_path
import histokit.data.slides  # noqa: F401 — triggers slide backend registration


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
) -> None:
    """Generate a thumbnail of a slide at a given magnification level."""
    if output is None:
        output = Path(f"{slide_path.stem}_thumb_L{level}.png")

    slide_cls = get_slide_cls_for_path(slide_path)
    with slide_cls(slide_path) as slide:
        thumb = slide.get_thumbnail(level)

    Image.fromarray(thumb).save(output)
    typer.echo(f"Saved thumbnail to {output}")
