from pathlib import Path
from typing import List

from .slide import SlideBase

registry: dict[str, type[SlideBase]] = {}
ext_registry: dict[str, type[SlideBase]] = {}


def register_slide(name: str, extensions: List[str]):
    def decorator(cls: type["SlideBase"]):
        if name in registry:
            raise ValueError(f"Slide type '{name}' already registered")
        registry[name] = cls
        for ext in extensions:
            key = ext if ext.startswith(".") else f".{ext}"
            ext_registry[key] = cls
        return cls

    return decorator


def get_slide_cls(name: str) -> type["SlideBase"]:
    return registry[name]


def get_slide_cls_for_path(path: Path) -> type["SlideBase"]:
    suffixes = path.suffixes
    # Try the full compound suffix first (e.g. ".ome.tiff"), then each trailing suffix
    for i in range(len(suffixes)):
        key = "".join(suffixes[i:])
        if key in ext_registry:
            return ext_registry[key]
    raise ValueError(f"No slide backend registered for '{path.name}'")
