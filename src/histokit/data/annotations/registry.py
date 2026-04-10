from pathlib import Path
from typing import Callable

from histokit.data.annotations.annotation import AnnotationRegion
from histokit.data.schema import AnnotationSchema

AnnotationLoader = Callable[[Path, AnnotationSchema], list[AnnotationRegion]]

registry: dict[str, AnnotationLoader] = {}


def register_annotation(name: str):
    def decorator(func: AnnotationLoader):
        if name in registry:
            raise ValueError(f"Slide type '{name}' already registered")
        registry[name] = func

    return decorator


def get_annotation_loader(name: str) -> AnnotationLoader:
    return registry[name]
