from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from histokit.dataset.sample import Sample
from histokit.dataset.schema import DatasetSchema


class Dataset:
    def __init__(
        self, index: pd.DataFrame, schema: DatasetSchema, data_root: Path, 
        annotations_root: Path = Path(),
        slide_root: Path = Path()
    ) -> None:
        self.index = index
        self.schema = schema
        self.data_root = data_root
        self.annotations_root = annotations_root
        self.slide_root = slide_root


    @property
    def slide_kind(self) -> str:
        return self.schema.slides.kind

    @property
    def annotation_kind(self) -> str:
        return self.schema.annotations.kind

    def sample_from_row(self, row: pd.Series) -> Sample:
        slide_path = self.data_root / row["slide"]

        annotation_path = None
        if "annotation" in row and pd.notna(row["annotation"]):
            annotation_path = self.data_root / row["annotation"]

        metadata = {
            col: row[col]
            for col in self.index.columns
            if col not in {"slide", "annotation"}
        }

        # the id is just the file name without any extensions
        sample_id = metadata.get("id", Path(row["slide"]).name.split(".", 1)[0])

        return Sample(
            id=str(sample_id),
            slide_path=slide_path,
            slide_schema=self.schema.slides,
            annotation_path=annotation_path,
            annotation_schema=self.schema.annotations if annotation_path else None,
            metadata=metadata,
        )

    def samples(self):
        for _, row in self.index.iterrows():
            yield self.sample_from_row(row)

    @classmethod
    def from_index(
        cls, 
        index_csv_path: Path | str, 
        schema_path: Path | str,
        slides_dir: Path | str | None = None, 
        annotations_dir: Path | str | None = None,
    ) -> "Dataset":
        index_csv_path = Path(index_csv_path)
        schema_path = Path(schema_path)

        index = pd.read_csv(index_csv_path)

        # append the slide_dir to the slide paths in the index if provided
        if slides_dir is not None:
            slides_dir = Path(slides_dir)
            index["slide"] = index["slide"].apply(lambda x: str(slides_dir / x))

        # append the annotations_dir to the annotation paths in the index if provided
        if annotations_dir is not None:
            annotations_dir = Path(annotations_dir)
            index["annotation"] = index["annotation"].apply(lambda x: str(annotations_dir / x) if pd.notna(x) else x)

        with open(schema_path, "r") as file:
            schema_dict = json.load(file)

        schema = DatasetSchema.from_dict(schema_dict)

        # All paths in the index are relative to the index.csv parent directory.
        data_root = index_csv_path.parent

        return cls(index, schema, data_root)
