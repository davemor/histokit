"""histokit plan — inspect a pipeline's stages and resolved parameters."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from .helpers import load_pipeline, parse_set_overrides, describe_stage


def plan(
    pipeline: Annotated[
        str,
        typer.Argument(help="Pipeline reference (module.path:attribute)."),
    ],
    set: Annotated[
        Optional[list[str]],
        typer.Option("--set", help="Parameter override (key=value). Repeatable."),
    ] = None,
) -> None:
    """Show pipeline stages and resolved parameters."""
    pipe = load_pipeline(pipeline)
    params = parse_set_overrides(set)

    typer.echo(f"Pipeline: {pipe.name}\n")
    typer.echo("Stages:")
    for i, stage in enumerate(pipe.stages, 1):
        desc = describe_stage(stage, params)
        typer.echo(f"  {i}. {desc}")

    if params:
        typer.echo("\nParameter overrides:")
        for k, v in params.items():
            typer.echo(f"  {k} = {v!r}")
