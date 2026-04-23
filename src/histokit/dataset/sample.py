from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from histokit.dataset.schema import AnnotationSchema, SlideSchema
from histokit.io.slides.registry import get_slide_cls, get_slide_cls_for_path, is_slide_extension_supported, is_slide_format_supported
from histokit.io.slides.slide import SlideBase


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

    def make_slide(self) -> SlideBase:
        if is_slide_format_supported(self.slide_schema.kind):
            slide_cls = get_slide_cls(self.slide_schema.kind)
        elif is_slide_extension_supported(self.slide_path.suffix):
            slide_cls = get_slide_cls_for_path(self.slide_path)
        else:
            raise ValueError(f"No slide backend registered for format '{self.slide_schema.kind}' or extension '{self.slide_path.suffix}'")
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
