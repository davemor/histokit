import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .manifest import PatchSetManifest
from .context import PatchContext

if TYPE_CHECKING:
    from histokit.dataset.dataset import Dataset


class PatchSet:
    def __init__(
        self,
        frame: pd.DataFrame,  # columns: x, y, labels, context_id, keep - also any additional columns based on quantities extracted in the pipeline
        contexts: list[PatchContext],
        manifest: PatchSetManifest,
    ) -> None:
        self.frame = frame
        self.contexts = contexts
        self.manifest = manifest

    def __repr__(self) -> str:
        return f"PatchSet(num_patches={len(self.frame)}, contexts={self.contexts}, manifest={self.manifest})"
    
    def describe(self) -> pd.DataFrame:
        annotation_schema = None

        # find the first annotation schema from the contexts to get the label map for counting annotation labels - assumes all contexts in the patchset have the same annotation schema if they have annotations at all
        for ctx in self.contexts:
            if ctx.sample.annotation_schema is not None:
                annotation_schema = ctx.sample.annotation_schema
                break

        if annotation_schema is None:
            return pd.DataFrame()

        label_map = annotation_schema.label_map
        counts = {
            label: int((self.frame["annotation_label"] == value).sum())
            for label, value in label_map.items()
        }
        return pd.DataFrame([counts])

    def save(self, path: Path | str) -> None:
        """Save a PatchSet to a directory.

        Creates ``path/`` containing ``frame.parquet`` and ``manifest.json``.
        The manifest records the PatchSetManifest fields as well as enough
        context information (sample id, level, patch_size) to reconstruct the
        PatchSet later via :meth:`load`.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.frame.to_parquet(path / "frame.parquet", index=False)

        contexts_data = [
            {
                "sample_id": ctx.sample.id,
                "level": ctx.level,
                "patch_size": ctx.patch_size,
            }
            for ctx in self.contexts
        ]

        manifest_data = {
            "manifest": asdict(self.manifest),
            "contexts": contexts_data,
        }

        with open(path / "manifest.json", "w") as f:
            json.dump(manifest_data, f, indent=2)

    @classmethod
    def load(cls, path: Path | str, dataset: "Dataset") -> "PatchSet":
        """Load a PatchSet from a directory previously created by :meth:`save`.

        Args:
            path: Directory containing ``frame.parquet`` and ``manifest.json``.
            dataset: The Dataset used to reconstruct full Sample objects from
                the saved sample ids.
        """
        path = Path(path)

        frame = pd.read_parquet(path / "frame.parquet")

        with open(path / "manifest.json", "r") as f:
            manifest_data = json.load(f)

        manifest = PatchSetManifest(**manifest_data["manifest"])

        sample_lookup = {s.id: s for s in dataset.samples()}
        contexts = []
        for ctx_data in manifest_data["contexts"]:
            sample = sample_lookup[ctx_data["sample_id"]]
            contexts.append(
                PatchContext(
                    sample=sample,
                    level=ctx_data["level"],
                    patch_size=ctx_data["patch_size"],
                )
            )

        return cls(frame=frame, contexts=contexts, manifest=manifest)

    def export(self, path: Path | str, only_kept: bool = True) -> None:
        """Export patch images to a directory with label-name subdirectories.

        The output structure is compatible with ``torchvision.datasets.ImageFolder``::

            path/
            ├── provenance.json
            ├── label_a/
            │   ├── sample_row_col.png
            │   └── ...
            └── label_b/
                └── ...

        A ``provenance.json`` file records how the export was produced:
        manifest info, per-context extraction parameters, label mapping,
        and patch counts per label.

        Args:
            path: Root directory to write patch images into.
            only_kept: If True (default) and a ``keep`` column exists, only
                export patches where ``keep`` is True.
        """
        from histokit.io.slides.slide import Region

        path = Path(path)

        # Build label value -> name mapping from the annotation schema
        val_to_name: dict[int, str] = {}
        annotation_schema_dict: dict = {}
        for ctx in self.contexts:
            if ctx.sample.annotation_schema is not None:
                schema = ctx.sample.annotation_schema
                val_to_name = {v: k for k, v in schema.label_map.items()}
                annotation_schema_dict = {
                    "kind": schema.kind,
                    "label_map": schema.label_map,
                    "fill_label": schema.fill_label,
                    "cutout_label": schema.cutout_label,
                    "label_order": schema.label_order,
                }
                break

        frame = self.frame
        if only_kept and "keep" in frame.columns:
            frame = frame[frame["keep"] == True]

        for context_id, group in frame.groupby("context_id"):
            ctx = self.contexts[int(context_id)]
            sample = ctx.sample
            level = ctx.level
            patch_size = ctx.patch_size

            with sample.open_slide() as slide:
                downsample = float(slide.level_downsamples()[level])

                for _, row in group.iterrows():
                    r = int(row["row"])
                    c = int(row["column"])

                    x0 = int(c * patch_size * downsample)
                    y0 = int(r * patch_size * downsample)
                    region = Region.patch(x0, y0, patch_size, level)
                    img = slide.read_region(region).convert("RGB")

                    label_val = row.get("annotation_label")
                    if pd.notna(label_val):
                        label_name = val_to_name.get(int(label_val), str(int(label_val)))
                    else:
                        label_name = "unlabelled"

                    label_dir = path / label_name
                    label_dir.mkdir(parents=True, exist_ok=True)

                    filename = f"{sample.id}_{r}_{c}.png"
                    img.save(label_dir / filename)

        # Count patches per label in the exported set
        label_counts: dict[str, int] = {}
        if "annotation_label" in frame.columns:
            for val, count in frame["annotation_label"].value_counts().items():
                name = val_to_name.get(int(val), str(int(val))) if pd.notna(val) else "unlabelled"
                label_counts[name] = int(count)
            n_unlabelled = int(frame["annotation_label"].isna().sum())
            if n_unlabelled > 0:
                label_counts["unlabelled"] = n_unlabelled

        provenance = {
            "manifest": asdict(self.manifest),
            "num_patches": len(frame),
            "num_samples": len(self.contexts),
            "label_counts": label_counts,
            "annotation_schema": annotation_schema_dict,
            "only_kept": only_kept,
            "contexts": [
                {
                    "sample_id": ctx.sample.id,
                    "slide_path": str(ctx.sample.slide_path),
                    "level": ctx.level,
                    "patch_size": ctx.patch_size,
                    "metadata": ctx.sample.metadata,
                }
                for ctx in self.contexts
            ],
        }

        with open(path / "provenance.json", "w") as f:
            json.dump(provenance, f, indent=2)
    



def combine_patchsets(patchsets: list[PatchSet]) -> PatchSet:
    """Combine multiple PatchSets into a single PatchSet.

    Each input PatchSet's context_id values are remapped so they remain unique
    in the combined result.
    """
    if not patchsets:
        raise ValueError("Cannot combine an empty list of PatchSets")

    if len(patchsets) == 1:
        return patchsets[0]

    all_contexts: list[PatchContext] = []
    frames: list[pd.DataFrame] = []
    context_offset = 0

    for ps in patchsets:
        f = ps.frame.copy()
        if "context_id" in f.columns:
            f["context_id"] = f["context_id"] + context_offset
        frames.append(f)
        all_contexts.extend(ps.contexts)
        context_offset += len(ps.contexts)

    combined_frame = pd.concat(frames, ignore_index=True)

    return PatchSet(
        frame=combined_frame,
        contexts=all_contexts,
        manifest=patchsets[0].manifest,
    )