"""histokit export — export patch images from a saved PatchSet."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .helpers import load_dataset


def export_cmd(
    patchset_dir: Annotated[
        Path,
        typer.Argument(help="Path to saved PatchSet directory (contains frame.parquet)."),
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
        typer.Option("--output", "-o", help="Output directory for exported patch images."),
    ],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite the output directory if it exists."),
    ] = False,
) -> None:
    """Export patch images from a saved PatchSet to label-name directories."""
    from histokit.patchset.patchset import PatchSet

    if not patchset_dir.exists():
        typer.echo(f"PatchSet directory not found: {patchset_dir}")
        raise typer.Exit(1)

    if output.exists() and not overwrite:
        typer.echo(f"Output directory already exists: {output}")
        typer.echo("Use --overwrite to replace it.")
        raise typer.Exit(1)

    dataset = load_dataset(index, labels)
    patchset = PatchSet.load(patchset_dir, dataset)

    n_patches = len(patchset.frame)
    n_kept = int(patchset.frame["keep"].sum()) if "keep" in patchset.frame.columns else n_patches
    typer.echo(f"PatchSet: {n_patches} patches ({n_kept} kept)")
    typer.echo(f"Output:   {output}\n")

    patchset.export(output)

    typer.echo(f"Done. Exported patches to {output}")
