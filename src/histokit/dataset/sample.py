from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from histokit.dataset.schema import AnnotationSchema, SlideSchema

if TYPE_CHECKING:
    from histokit.io.annotations.annotation import AnnotationSet


@dataclass(frozen=True)
class Sample:
    id: str
    slide_path: Path
    slide_schema: SlideSchema
    annotation_path: Path | None = None
    annotation_schema: AnnotationSchema | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def make_slide(self):
        from histokit.io.slides.registry import get_slide_cls

        slide_cls = get_slide_cls(self.slide_schema.kind)
        return slide_cls(self.slide_path)

    @contextmanager
    def open_slide(self):
        slide = self.make_slide()
        with slide:
            yield slide

    def make_annotations(self) -> AnnotationSet | None:
        from histokit.io.annotations.annotation import AnnotationSet
        from histokit.io.annotations.registry import get_annotation_loader

        if self.annotation_path is None:
            return None

        if self.annotation_schema is None:
            raise ValueError(
                "annotation_schema must be provided when annotation_path is set."
            )

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"File not found: {self.annotation_path}")

        load_annotations = get_annotation_loader(self.annotation_schema.kind)
        annotations = load_annotations(
            self.annotation_path,
            self.annotation_schema,
        )

        return AnnotationSet(
            annotations=annotations,
            schema=self.annotation_schema,
        )
