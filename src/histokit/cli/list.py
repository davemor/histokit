"""histokit list — show available built-in pipelines."""

from __future__ import annotations

import importlib
import pkgutil

import typer

from histokit.pipelines.pipeline import Pipeline


def list_pipelines() -> None:
    """List available built-in pipelines."""
    package_name = "histokit.pipelines.presets"
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        typer.echo("No built-in presets found.")
        raise typer.Exit(1)

    typer.echo("Available pipelines:\n")
    for importer, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        if module_name.startswith("_"):
            continue
        full_module = f"{package_name}.{module_name}"
        try:
            mod = importlib.import_module(full_module)
        except Exception:
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, Pipeline):
                ref = f"{full_module}:{attr_name}"
                n_stages = len(obj.stages)
                typer.echo(f"  {ref}  ({n_stages} stages)")
