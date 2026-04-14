"""Tests for histokit.cli (helpers, list, plan, run, preview, export)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from histokit.cli.main import app
from histokit.cli.helpers import (
    _coerce,
    describe_stage,
    load_pipeline,
    parse_set_overrides,
)
from histokit.dataset.sample import Sample
from histokit.dataset.schema import AnnotationSchema, SlideSchema
from histokit.patchset.context import PatchContext
from histokit.patchset.manifest import PatchSetManifest
from histokit.patchset.patchset import PatchSet
from histokit.pipelines.params import Param, param
from histokit.pipelines.pipeline import Pipeline
from histokit.pipelines.stages import FilterPatches, Grid

from conftest import needs_cervical_mini, CERVICAL_MINI

slow = pytest.mark.slow

runner = CliRunner()


# ── helpers: _coerce ─────────────────────────────────────────────────────────

class TestCoerce:
    def test_true_variants(self):
        assert _coerce("true") is True
        assert _coerce("True") is True
        assert _coerce("yes") is True

    def test_false_variants(self):
        assert _coerce("false") is False
        assert _coerce("False") is False
        assert _coerce("no") is False

    def test_none(self):
        assert _coerce("none") is None
        assert _coerce("None") is None

    def test_int(self):
        assert _coerce("42") == 42
        assert isinstance(_coerce("42"), int)

    def test_float(self):
        assert _coerce("3.14") == pytest.approx(3.14)
        assert isinstance(_coerce("3.14"), float)

    def test_string_fallback(self):
        assert _coerce("hello") == "hello"
        assert _coerce("") == ""


# ── helpers: parse_set_overrides ─────────────────────────────────────────────

class TestParseSetOverrides:
    def test_none_returns_empty(self):
        assert parse_set_overrides(None) == {}

    def test_empty_list_returns_empty(self):
        assert parse_set_overrides([]) == {}

    def test_single_int(self):
        assert parse_set_overrides(["level=2"]) == {"level": 2}

    def test_multiple_types(self):
        result = parse_set_overrides(["level=2", "thresh=0.05", "flag=true", "name=foo"])
        assert result == {"level": 2, "thresh": pytest.approx(0.05), "flag": True, "name": "foo"}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="expected key=value"):
            parse_set_overrides(["bad"])

    def test_value_with_equals(self):
        result = parse_set_overrides(["expr=a=b"])
        assert result == {"expr": "a=b"}


# ── helpers: load_pipeline ───────────────────────────────────────────────────

class TestLoadPipeline:
    def test_load_basic_preset(self):
        pipe = load_pipeline("histokit.pipelines.presets.basic:pipeline")
        assert isinstance(pipe, Pipeline)
        assert len(pipe.stages) > 0

    def test_load_research_preset(self):
        pipe = load_pipeline("histokit.pipelines.presets.research:pipeline")
        assert isinstance(pipe, Pipeline)

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="module.path:attribute"):
            load_pipeline("histokit.pipelines.presets.basic")

    def test_bad_module_raises(self):
        with pytest.raises(ValueError, match="Cannot import module"):
            load_pipeline("nonexistent.module:pipeline")

    def test_bad_attribute_raises(self):
        with pytest.raises(ValueError, match="no attribute"):
            load_pipeline("histokit.pipelines.presets.basic:nonexistent")

    def test_non_pipeline_raises(self):
        with pytest.raises(TypeError, match="expected Pipeline"):
            load_pipeline("histokit.pipelines.params:Param")


# ── helpers: describe_stage ──────────────────────────────────────────────────

class TestDescribeStage:
    def test_plain_values(self):
        g = Grid(level=1, patch_size=256)
        desc = describe_stage(g)
        assert "Grid" in desc
        assert "level=1" in desc
        assert "patch_size=256" in desc

    def test_param_defaults(self):
        g = Grid(level=param("level", 2), patch_size=256)
        desc = describe_stage(g)
        assert "level=2" in desc

    def test_param_with_overrides(self):
        g = Grid(level=param("level", 2), patch_size=param("ps", 512))
        desc = describe_stage(g, params={"level": 0, "ps": 128})
        assert "level=0" in desc
        assert "patch_size=128" in desc


# ── CLI: histokit --help ─────────────────────────────────────────────────────

class TestAppHelp:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("list", "plan", "run", "preview", "export"):
            assert cmd in result.output


# ── CLI: histokit list ───────────────────────────────────────────────────────

class TestListCommand:
    def test_lists_built_in_pipelines(self):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Available pipelines" in result.output
        assert "histokit.pipelines.presets.basic:pipeline" in result.output
        assert "histokit.pipelines.presets.research:pipeline" in result.output
        assert "stages" in result.output


# ── CLI: histokit plan ───────────────────────────────────────────────────────

class TestPlanCommand:
    def test_plan_basic(self):
        result = runner.invoke(app, ["plan", "histokit.pipelines.presets.basic:pipeline"])
        assert result.exit_code == 0
        assert "Pipeline:" in result.output
        assert "Stages:" in result.output
        assert "Grid" in result.output
        assert "FilterPatches" in result.output

    def test_plan_with_overrides(self):
        result = runner.invoke(app, [
            "plan",
            "histokit.pipelines.presets.basic:pipeline",
            "--set", "level=0",
            "--set", "patch_size=512",
        ])
        assert result.exit_code == 0
        assert "Parameter overrides" in result.output
        assert "level = 0" in result.output

    def test_plan_bad_ref(self):
        result = runner.invoke(app, ["plan", "bad.module:nope"])
        assert result.exit_code != 0


# ── Fixtures for run / preview / export ──────────────────────────────────────

def _make_sample(sample_id: str = "test_sample") -> Sample:
    return Sample(
        id=sample_id,
        slide_path=Path("/tmp/slide.tiff"),
        slide_schema=SlideSchema(kind="tiffslide", label_schema={}),
        annotation_schema=AnnotationSchema(
            kind="geojson",
            label_map={"background": 0, "normal": 1, "tumor": 2},
            cutout_label="normal",
            fill_label="normal",
            label_order=["background", "tumor", "normal"],
        ),
        metadata={"category": "test"},
    )


def _make_patchset(n: int = 10) -> PatchSet:
    frame = pd.DataFrame({
        "row": [i // 5 for i in range(n)],
        "column": [i % 5 for i in range(n)],
        "context_id": [0] * n,
        "annotation_label": [i % 3 for i in range(n)],
        "tissue_score": [float(i) / n for i in range(n)],
        "keep": [i >= 3 for i in range(n)],
    })
    ctx = PatchContext(sample=_make_sample(), level=1, patch_size=256)
    manifest = PatchSetManifest(created_at="2025-01-01T00:00:00Z")
    return PatchSet(frame=frame, contexts=[ctx], manifest=manifest)


# ── CLI: histokit run ────────────────────────────────────────────────────────

class TestRunCommand:
    def test_missing_index(self):
        result = runner.invoke(app, [
            "run",
            "histokit.pipelines.presets.basic:pipeline",
            "--index", "/nonexistent/index.csv",
            "--labels", "/nonexistent/labels.json",
            "--output", "/tmp/out",
        ])
        assert result.exit_code != 0

    def test_output_exists_no_overwrite(self, tmp_path):
        out = tmp_path / "existing"
        out.mkdir()
        result = runner.invoke(app, [
            "run",
            "histokit.pipelines.presets.basic:pipeline",
            "--index", "/nonexistent/index.csv",
            "--labels", "/nonexistent/labels.json",
            "--output", str(out),
        ])
        assert result.exit_code != 0
        assert "already exists" in result.output

    @needs_cervical_mini
    @slow
    def test_run_basic_pipeline(self, tmp_path):
        out = tmp_path / "run_out"
        result = runner.invoke(app, [
            "run",
            "histokit.pipelines.presets.basic:pipeline",
            "--index", str(CERVICAL_MINI / "index.csv"),
            "--labels", str(CERVICAL_MINI / "labels.json"),
            "--output", str(out),
        ])
        assert result.exit_code == 0
        assert "Done." in result.output
        assert (out / "frame.parquet").exists()
        assert (out / "manifest.json").exists()

    @needs_cervical_mini
    @slow
    def test_run_with_overwrite(self, tmp_path):
        out = tmp_path / "run_ow"
        out.mkdir()
        (out / "old_file.txt").write_text("stale")
        result = runner.invoke(app, [
            "run",
            "histokit.pipelines.presets.basic:pipeline",
            "--index", str(CERVICAL_MINI / "index.csv"),
            "--labels", str(CERVICAL_MINI / "labels.json"),
            "--output", str(out),
            "--overwrite",
        ])
        assert result.exit_code == 0
        assert "Done." in result.output

    @needs_cervical_mini
    @slow
    def test_run_with_overrides(self, tmp_path):
        out = tmp_path / "run_ov"
        result = runner.invoke(app, [
            "run",
            "histokit.pipelines.presets.basic:pipeline",
            "--index", str(CERVICAL_MINI / "index.csv"),
            "--labels", str(CERVICAL_MINI / "labels.json"),
            "--output", str(out),
            "--set", "patch_size=512",
        ])
        assert result.exit_code == 0
        assert "Done." in result.output


# ── CLI: histokit preview ────────────────────────────────────────────────────

class TestPreviewCommand:
    def test_bad_sample_id(self, tmp_path):
        """--sample with a non-existent ID should fail gracefully."""
        result = runner.invoke(app, [
            "preview",
            "histokit.pipelines.presets.basic:pipeline",
            "--index", str(CERVICAL_MINI / "index.csv"),
            "--labels", str(CERVICAL_MINI / "labels.json"),
            "--output", str(tmp_path / "pv"),
            "--sample", "NONEXISTENT_SAMPLE",
        ])
        # should fail with sample-not-found or data-not-available skip
        if "not found" in result.output.lower() or result.exit_code != 0:
            pass  # expected
        else:
            pytest.fail("Expected error for non-existent sample")

    @needs_cervical_mini
    @slow
    def test_preview_first_sample(self, tmp_path):
        out = tmp_path / "preview_out"
        result = runner.invoke(app, [
            "preview",
            "histokit.pipelines.presets.basic:pipeline",
            "--index", str(CERVICAL_MINI / "index.csv"),
            "--labels", str(CERVICAL_MINI / "labels.json"),
            "--output", str(out),
        ])
        assert result.exit_code == 0
        assert "Pipeline:" in result.output
        assert "Sample:" in result.output
        assert (out / "thumbnail.png").exists()

    @needs_cervical_mini
    @slow
    def test_preview_specific_sample(self, tmp_path):
        # Discover actual sample IDs from the dataset
        from histokit.dataset.dataset import Dataset

        ds = Dataset.from_index(
            CERVICAL_MINI / "index.csv",
            CERVICAL_MINI / "labels.json",
            slides_dir="slides",
            annotations_dir="annotations",
        )
        sample_ids = [s.id for s in ds.samples()]
        target = sample_ids[-1]

        out = tmp_path / "preview_specific"
        result = runner.invoke(app, [
            "preview",
            "histokit.pipelines.presets.basic:pipeline",
            "--index", str(CERVICAL_MINI / "index.csv"),
            "--labels", str(CERVICAL_MINI / "labels.json"),
            "--output", str(out),
            "--sample", target,
        ])
        assert result.exit_code == 0
        assert target in result.output


# ── CLI: histokit export ─────────────────────────────────────────────────────

class TestExportCommand:
    def test_missing_patchset_dir(self, tmp_path):
        result = runner.invoke(app, [
            "export",
            str(tmp_path / "nonexistent"),
            "--index", "/tmp/i.csv",
            "--labels", "/tmp/l.json",
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_output_exists_no_overwrite(self, tmp_path):
        ps_dir = tmp_path / "ps"
        ps_dir.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        result = runner.invoke(app, [
            "export",
            str(ps_dir),
            "--index", "/tmp/i.csv",
            "--labels", "/tmp/l.json",
            "--output", str(out),
        ])
        assert result.exit_code != 0
        assert "already exists" in result.output

    @needs_cervical_mini
    @slow
    def test_export_roundtrip(self, tmp_path):
        """Run a pipeline, save, then export patches."""
        # First run a pipeline to produce a patchset
        run_out = tmp_path / "run_out"
        run_result = runner.invoke(app, [
            "run",
            "histokit.pipelines.presets.basic:pipeline",
            "--index", str(CERVICAL_MINI / "index.csv"),
            "--labels", str(CERVICAL_MINI / "labels.json"),
            "--output", str(run_out),
        ])
        assert run_result.exit_code == 0

        # Then export
        export_out = tmp_path / "export_out"
        result = runner.invoke(app, [
            "export",
            str(run_out),
            "--index", str(CERVICAL_MINI / "index.csv"),
            "--labels", str(CERVICAL_MINI / "labels.json"),
            "--output", str(export_out),
        ])
        assert result.exit_code == 0
        assert "Done." in result.output
        assert (export_out / "provenance.json").exists()
