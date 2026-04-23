"""histokit run — execute a pipeline on a dataset and save the PatchSet."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .helpers import load_pipeline, parse_set_overrides, load_dataset


def run(
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
        typer.Option("--output", "-o", help="Output directory for saved PatchSet."),
    ],
    set: Annotated[
        Optional[list[str]],
        typer.Option("--set", help="Parameter override (key=value). Repeatable."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite the output directory if it exists."),
    ] = False,
) -> None:
    """Run a pipeline on a dataset and save the resulting PatchSet."""
    from histokit.patchset.patchset import combine_patchsets

    if output.exists() and not overwrite:
        typer.echo(f"Output directory already exists: {output}")
        typer.echo("Use --overwrite to replace it.")
        raise typer.Exit(1)

    pipe = load_pipeline(pipeline)
    params = parse_set_overrides(set)
    dataset = load_dataset(index, labels)

    n_samples = len(dataset.index)
    typer.echo(f"Pipeline: {pipe.name}")
    typer.echo(f"Dataset:  {n_samples} sample(s)")
    typer.echo(f"Output:   {output}\n")

    results = pipe.run(dataset, **params)

    print(f"Pipeline completed. Got {len(results)} patchsets.")

    # Save per-sample patchsets
    output.mkdir(parents=True, exist_ok=True)
    for patchset in results:
        print('Processing patchset for sample:', patchset.contexts[0].sample.id if patchset.contexts else 'unknown')
        sample_id = patchset.contexts[0].sample.id if patchset.contexts else "unknown"
        patchset_dir = output / f"{sample_id}.patchset"
        patchset.save(patchset_dir)

    # Combine and save the full patchset as before
    patchset = combine_patchsets(results)
    patchset.save(output / "full.patchset")

    n_kept = int(patchset.frame["keep"].sum()) if "keep" in patchset.frame.columns else len(patchset.frame)
    typer.echo(f"\nDone. {len(patchset.frame)} patches ({n_kept} kept), saved to {output}")

    desc = patchset.describe()
    if not desc.empty:
        typer.echo("\nLabel counts:")
        for col in desc.columns:
            typer.echo(f"  {col}: {int(desc[col].iloc[0])}")
