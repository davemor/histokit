

from pathlib import Path
from typing import Optional
import typer
from PIL import Image
from histokit.io.slides.registry import get_slide_cls_for_path
import histokit.io.slides  # noqa: F401 — triggers slide backend registration
from histokit.dataset import Dataset


def patchmap(
	patch_size: int = typer.Argument(..., help="Patch size in pixels at the given level."),
	patch_level: int = typer.Argument(..., help="Pyramid level at which patch size is defined."),
	pixels_per_patch: int = typer.Argument(..., help="Number of pixels per patch in the output image."),
	slide_path: Optional[Path] = typer.Option(None, "--slide", help="Path to a single whole-slide image."),
	output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for the patchmap image (single slide mode or output directory for dataset mode)."),
	index: Optional[Path] = typer.Option(None, "--index", help="Path to dataset index.csv (enables dataset mode)."),
	schema: Optional[Path] = typer.Option(None, "--schema", help="Path to dataset labels.json schema file (dataset mode)."),
	slides_dir: Optional[Path] = typer.Option(None, "--slides-dir", help="Optional slides directory for dataset mode."),
	white_background: bool = typer.Option(True, help="Replace pure black pixels with white."),
) -> None:
    """Generate patchmaps for a single slide or for all slides in a dataset."""
    if index and schema:
        # Dataset mode
        ds = Dataset.from_index(index, schema, slides_dir=slides_dir)
        if output is not None:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None
        for sample in ds.samples():
            slide_path = sample.slide_path
            if output_dir is not None:
                out_path = output_dir / f"{Path(slide_path).stem}_patchmap_L{patch_level}_P{patch_size}_S{pixels_per_patch}.png"
            else:
                out_path = slide_path.with_suffix("").parent / f"{Path(slide_path).stem}_patchmap_L{patch_level}_P{patch_size}_S{pixels_per_patch}.png"
            try:
                with sample.open_slide() as slide:
                    patch_grid = slide.size_in_patches(patch_size, patch_level)
                    out_width = patch_grid.width * pixels_per_patch
                    out_height = patch_grid.height * pixels_per_patch
                    thumb = slide.get_thumbnail_for_size(
                        width=out_width,
                        height=out_height,
                        white_background=white_background,
                    )
                im = Image.fromarray(thumb)
                im.save(out_path)
                typer.echo(f"Saved patchmap to {out_path}")
            except Exception as e:
                typer.echo(f"Failed for {slide_path}: {e}", err=True)
    elif slide_path:
        # Single slide mode
        if output is None:
            output = Path(f"{slide_path.stem}_patchmap_L{patch_level}_P{patch_size}_S{pixels_per_patch}.png")
        slide_cls = get_slide_cls_for_path(slide_path)
        with slide_cls(slide_path) as slide:
            patch_grid = slide.size_in_patches(patch_size, patch_level)
            out_width = patch_grid.width * pixels_per_patch
            out_height = patch_grid.height * pixels_per_patch
            thumb = slide.get_thumbnail_for_size(
                width=out_width,
                height=out_height,
                white_background=white_background,
            )
        im = Image.fromarray(thumb)
        im.save(output)
        typer.echo(f"Saved patchmap to {output}")
    else:
        typer.echo("You must provide either --slide for single slide mode or --index and --schema for dataset mode.", err=True)

