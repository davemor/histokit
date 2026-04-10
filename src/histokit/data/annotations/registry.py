from pathlib import Path
from typing import Callable, List

from histokit.data.annotations.annotation import AnnotationRegion
from histokit.data.schema import AnnotationSchema

AnnotationLoader = Callable[[Path, AnnotationSchema], list[AnnotationRegion]]

registry: dict[str, AnnotationLoader] = {}
ext_registry: dict[str, AnnotationLoader] = {}


def register_annotation(name: str, extensions: List[str] | None = None):
    def decorator(func: AnnotationLoader):
        if name in registry:
            raise ValueError(f"Annotation loader '{name}' already registered")
        registry[name] = func
        for ext in extensions or []:
            key = ext if ext.startswith(".") else f".{ext}"
            ext_registry[key] = func
        return func

    return decorator


def get_annotation_loader(name: str) -> AnnotationLoader:
    return registry[name]


def get_annotation_loader_for_path(path: Path) -> AnnotationLoader:
    suffixes = path.suffixes
    for i in range(len(suffixes)):
        key = "".join(suffixes[i:])
        if key in ext_registry:
            return ext_registry[key]
    raise ValueError(f"No annotation loader registered for '{path.name}'")
