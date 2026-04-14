"""Shared helpers for CLI commands."""

from __future__ import annotations

import importlib
from dataclasses import fields
from pathlib import Path
from typing import Any

from histokit.pipelines.params import Param
from histokit.pipelines.pipeline import Pipeline


def load_pipeline(ref: str) -> Pipeline:
    """Import a Pipeline from a ``module.path:attribute`` reference.

    Examples::

        load_pipeline("histokit.pipelines.presets.basic:pipeline")
        load_pipeline("myproject.custom:experiment_a")
    """
    if ":" not in ref:
        raise ValueError(
            f"Pipeline reference must be 'module.path:attribute', got '{ref}'"
        )
    module_path, attr_name = ref.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Cannot import module '{module_path}': {exc}") from exc
    try:
        obj = getattr(module, attr_name)
    except AttributeError:
        raise ValueError(
            f"Module '{module_path}' has no attribute '{attr_name}'"
        ) from None
    if not isinstance(obj, Pipeline):
        raise TypeError(
            f"'{ref}' resolved to {type(obj).__name__}, expected Pipeline"
        )
    return obj


def parse_set_overrides(values: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--set key=value`` strings into a params dict.

    Performs basic type coercion: booleans, ints, floats, else str.
    """
    if not values:
        return {}
    params: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}': expected key=value")
        key, raw = item.split("=", 1)
        params[key] = _coerce(raw)
    return params


def _coerce(raw: str) -> Any:
    """Best-effort coercion of a CLI string to a Python value."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def load_dataset(
    index: Path,
    labels: Path,
    slides_dir: str = "slides",
    annotations_dir: str = "annotations",
):
    """Load a Dataset from index CSV and labels JSON."""
    from histokit.dataset.dataset import Dataset

    return Dataset.from_index(
        index,
        labels,
        slides_dir=slides_dir,
        annotations_dir=annotations_dir,
    )


def describe_stage(stage, params: dict[str, Any] | None = None) -> str:
    """Return a human-readable string for a resolved or unresolved stage.

    If *params* is given, Param fields are resolved against it so the user
    sees the effective value.
    """
    stage_name = type(stage).__name__
    parts: list[str] = []
    for f in fields(stage):
        val = getattr(stage, f.name)
        if isinstance(val, Param):
            resolved = params.get(val.name, val.default) if params else val.default
            parts.append(f"{f.name}={resolved!r}")
        else:
            parts.append(f"{f.name}={val!r}")
    return f"{stage_name}({', '.join(parts)})"
