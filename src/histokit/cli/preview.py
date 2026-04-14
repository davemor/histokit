"""histokit preview — run a pipeline on a single sample and save diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .helpers import load_pipeline, parse_set_overrides, load_dataset


def preview(
    pipeline: Annotated[
        str,
        typer.Argument(help="Pipeline reference (module.path:attribute)."),
    ],
    index: Annotated[
        Path,
        typer.Option("--index", help="Path to dataset index CSV."),
    ],
    labels: Annotated[
        Path,
        typer.Option("--labels", help="Path to dataset labels JSON."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for preview artifacts."),
    ],
    sample: Annotated[
        Optional[str],
        typer.Option("--sample", help="Sample ID to preview. Defaults to first sample."),
    ] = None,
    set: Annotated[
        Optional[list[str]],
        typer.Option("--set", help="Parameter override (key=value). Repeatable."),
    ] = None,
    width: Annotated[
        int,
        typer.Option("--width", help="Thumbnail width in pixels."),
    ] = 1024,
) -> None:
    """Preview a pipeline on a single sample with diagnostic images."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from histokit.patchset.patchset import PatchSet

    pipe = load_pipeline(pipeline)
    params = parse_set_overrides(set)
    dataset = load_dataset(index, labels)

    # Find the target sample
    all_samples = list(dataset.samples())
    if sample is not None:
        matches = [s for s in all_samples if s.id == sample]
        if not matches:
            available = ", ".join(s.id for s in all_samples)
            typer.echo(f"Sample '{sample}' not found. Available: {available}")
            raise typer.Exit(1)
        target = matches[0]
    else:
        target = all_samples[0]

    typer.echo(f"Pipeline: {pipe.name}")
    typer.echo(f"Sample:   {target.id}")
    typer.echo(f"Output:   {output}\n")

    # Run pipeline on just this sample
    from histokit.pipelines.runtime import RuntimeContext

    runtime = RuntimeContext(
        params=params,
        pipeline_name=pipe.name,
        dataset_summary={"num_samples": 1},
    )
    result = pipe._run_one(target, runtime)

    output.mkdir(parents=True, exist_ok=True)

    # Save the PatchSet
    if isinstance(result, PatchSet):
        patchset = result
        patchset.save(output / "patchset")
        typer.echo(f"PatchSet: {len(patchset.frame)} patches")

        if "keep" in patchset.frame.columns:
            n_kept = int(patchset.frame["keep"].sum())
            typer.echo(f"Kept:     {n_kept}")

        desc = patchset.describe()
        if not desc.empty:
            typer.echo("\nLabel counts:")
            for col in desc.columns:
                typer.echo(f"  {col}: {int(desc[col].iloc[0])}")

    # Generate thumbnail
    with target.open_slide() as slide:
        slide_w, slide_h = slide.dimensions[0]
        height = max(1, round(width * slide_h / slide_w))
        thumb = slide.get_thumbnail_for_size(width, height)

    Image.fromarray(thumb).save(output / "thumbnail.png")
    typer.echo(f"\nSaved thumbnail.png ({width}x{height})")

    # Generate patch overlay if we have a PatchSet
    if isinstance(result, PatchSet) and target.annotation_schema is not None:
        _save_overlay(result, thumb, width, height, output)
        typer.echo("Saved overlay.png")


def _save_overlay(
    patchset: "PatchSet",
    thumb: "np.ndarray",
    width: int,
    height: int,
    output: Path,
) -> None:
    """Render the patch overlay to a file (non-interactive)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from histokit.viz.patchloc import (
        blend_overlay,
        build_label_colours,
        build_legend_handles,
        build_title,
        draw_patch_boxes,
    )

    context = patchset.contexts[0]
    sample = context.sample
    schema = sample.annotation_schema
    if schema is None:
        return

    frame = patchset.frame
    grid_w = int(frame["column"].max()) + 1
    grid_h = int(frame["row"].max()) + 1
    label_map = schema.label_map
    fill_val = label_map[schema.fill_label]
    val_to_name = {v: k for k, v in label_map.items()}

    _, label_colours = build_label_colours(label_map, fill_val, show_fill=False)
    blended, valid_frame = blend_overlay(
        thumb, frame, grid_h, grid_w, width, height,
        fill_val, label_colours, show_only_keep=True, alpha=0.45,
    )

    fig_h = height / 100
    fig_w = width / 100
    top_pad = 0.6
    bottom_pad = 0.9
    total_h = fig_h + top_pad + bottom_pad
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, total_h))
    fig.subplots_adjust(
        top=1.0 - top_pad / total_h,
        bottom=bottom_pad / total_h,
        left=0.0,
        right=1.0,
    )

    ax.imshow(blended)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(1.0)

    present_label_vals = set(
        valid_frame["annotation_label"].astype(int).unique()
    )
    title = build_title(sample, val_to_name, present_label_vals)
    ax.set_title(title, fontsize=18, pad=12)
    draw_patch_boxes(ax, frame, grid_h, grid_w, width, height, box_alpha=0.75)

    legend_handles = build_legend_handles(
        label_colours, val_to_name, present_label_vals,
        fill_val, show_fill=False, show_only_keep=True, frame=frame,
    )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=max(1, len(legend_handles)),
            fontsize=18,
            frameon=True,
            bbox_to_anchor=(0.5, 0.005),
        )

    fig.savefig(output / "overlay.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
