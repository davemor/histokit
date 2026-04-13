"""Tests for histokit.pipelines (params, stages, pipeline composition, runtime)."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd
import pytest

from histokit.pipelines.params import Param, param, resolve_value
from histokit.pipelines.runtime import RuntimeContext
from histokit.pipelines.stage import Stage
from histokit.pipelines.pipeline import Pipeline
from histokit.pipelines.stages import Grid, TissueMask, AssignLabels, FilterPatches
from histokit.pipelines.model import PatchCandidates, REQUIRED_GRID_COLUMNS
from histokit.patchset.context import PatchContext
from histokit.patchset.patchset import PatchSet
from conftest import needs_cervical_mini

slow = pytest.mark.slow


# ── Param / resolve_value ────────────────────────────────────────────────────

class TestParam:
    def test_param_factory(self):
        p = param("level", 2)
        assert isinstance(p, Param)
        assert p.name == "level"
        assert p.default == 2

    def test_resolve_from_runtime(self):
        p = param("level", 2)
        rt = RuntimeContext(params={"level": 5})
        assert p.resolve(rt) == 5

    def test_resolve_default(self):
        p = param("level", 2)
        rt = RuntimeContext(params={})
        assert p.resolve(rt) == 2

    def test_resolve_value_plain(self):
        rt = RuntimeContext()
        assert resolve_value(42, rt) == 42

    def test_resolve_value_param(self):
        rt = RuntimeContext(params={"x": 99})
        assert resolve_value(param("x", 0), rt) == 99


# ── RuntimeContext ───────────────────────────────────────────────────────────

class TestRuntimeContext:
    def test_defaults(self):
        rt = RuntimeContext()
        assert rt.params == {}
        assert rt.pipeline_name == "pipeline"
        assert rt.dataset_summary == {}
        assert isinstance(rt.created_at, str)

    def test_with_params(self):
        rt = RuntimeContext(params={"a": 1}, pipeline_name="test")
        assert rt.params["a"] == 1
        assert rt.pipeline_name == "test"


# ── PatchCandidates ──────────────────────────────────────────────────────────

class TestPatchCandidates:
    def _make_candidates(self, n: int = 6) -> PatchCandidates:
        from histokit.dataset.sample import Sample
        from histokit.dataset.schema import SlideSchema
        from pathlib import Path

        sample = Sample(
            id="s",
            slide_path=Path("/tmp/s.tiff"),
            slide_schema=SlideSchema(kind="tiffslide", label_schema={}),
        )
        ctx = PatchContext(sample=sample, level=1, patch_size=256)
        frame = pd.DataFrame({
            "row": list(range(n)),
            "column": [0] * n,
            "context_idx": [0] * n,
        })
        return PatchCandidates.from_grid(frame, [ctx])

    def test_from_grid(self):
        c = self._make_candidates()
        assert len(c.frame) == 6
        assert len(c.contexts) == 1

    def test_from_grid_missing_columns(self):
        from pathlib import Path
        from histokit.dataset.sample import Sample
        from histokit.dataset.schema import SlideSchema

        frame = pd.DataFrame({"row": [0], "column": [0]})  # missing context_idx
        sample = Sample(id="s", slide_path=Path("/tmp/s.tiff"), slide_schema=SlideSchema(kind="tiffslide", label_schema={}))
        ctx = PatchContext(sample=sample, level=1, patch_size=256)
        with pytest.raises(ValueError, match="missing required columns"):
            PatchCandidates.from_grid(frame, [ctx])

    def test_copy(self):
        c = self._make_candidates()
        c2 = c.copy()
        c2.frame["row"] = 999
        assert c.frame["row"].iloc[0] != 999

    def test_require_columns(self):
        c = self._make_candidates()
        c.require_columns("row", "column")  # should not raise
        with pytest.raises(ValueError, match="missing"):
            c.require_columns("nonexistent_col")

    def test_groupby_context(self):
        c = self._make_candidates()
        groups = list(c.groupby_context())
        assert len(groups) == 1
        idx, ctx, df = groups[0]
        assert idx == 0
        assert len(df) == 6


# ── Stage composition ────────────────────────────────────────────────────────

class TestStageComposition:
    def test_rshift_creates_pipeline(self):
        g = Grid(level=1, patch_size=256)
        f = FilterPatches()
        pipe = g >> f
        assert isinstance(pipe, Pipeline)
        assert len(pipe.stages) == 2

    def test_pipeline_rshift(self):
        g = Grid(level=1, patch_size=256)
        t = TissueMask()
        f = FilterPatches()
        pipe = g >> t >> f
        assert isinstance(pipe, Pipeline)
        assert len(pipe.stages) == 3


# ── Stage resolve ────────────────────────────────────────────────────────────

class TestGridResolve:
    def test_resolve_with_defaults(self):
        g = Grid(level=param("level", 2), patch_size=param("ps", 512))
        rt = RuntimeContext(params={})
        resolved = g.resolve(rt)
        assert resolved.level == 2
        assert resolved.patch_size == 512
        assert resolved.stride == 512  # defaults to patch_size

    def test_resolve_with_params(self):
        g = Grid(level=param("level", 2), patch_size=param("ps", 512))
        rt = RuntimeContext(params={"level": 0, "ps": 128})
        resolved = g.resolve(rt)
        assert resolved.level == 0
        assert resolved.patch_size == 128

    def test_resolve_stride_none_defaults_to_patch_size(self):
        g = Grid(level=1, patch_size=256, stride=None)
        rt = RuntimeContext()
        resolved = g.resolve(rt)
        assert resolved.stride == 256


class TestFilterPatchesResolve:
    def test_resolve(self):
        f = FilterPatches(
            tissue_threshold=param("thresh", 0.05),
            drop_background=True,
        )
        rt = RuntimeContext(params={"thresh": 0.1})
        resolved = f.resolve(rt)
        assert resolved.tissue_threshold == 0.1
        assert resolved.drop_background is True


# ── FilterPatches.run ────────────────────────────────────────────────────────

class TestFilterPatchesRun:
    def test_produces_patchset(self):
        from pathlib import Path
        from histokit.dataset.sample import Sample
        from histokit.dataset.schema import SlideSchema

        sample = Sample(
            id="s",
            slide_path=Path("/tmp/s.tiff"),
            slide_schema=SlideSchema(kind="tiffslide", label_schema={}),
        )
        ctx = PatchContext(sample=sample, level=1, patch_size=256)
        frame = pd.DataFrame({
            "row": [0, 1, 2],
            "column": [0, 0, 0],
            "context_idx": [0, 0, 0],
            "tissue_score": [0.0, 0.5, 0.9],
            "annotation_label": [1, 2, 1],
        })
        candidates = PatchCandidates(frame=frame, contexts=[ctx])
        rt = RuntimeContext()

        f = FilterPatches(tissue_threshold=0.1, drop_background=True)
        result = f.run(candidates, rt)

        assert isinstance(result, PatchSet)
        assert "keep" in result.frame.columns
        assert "context_id" in result.frame.columns
        # patch at tissue_score=0.0 should be dropped
        assert result.frame["keep"].sum() == 2


# ── Integration: Pipeline.run with real data ─────────────────────────────────

@needs_cervical_mini
@slow
class TestPipelineIntegration:
    def test_basic_pipeline_single_sample(self, cervical_mini_dataset):
        """Run the full basic preset pipeline on cervical_mini data."""
        from histokit.pipelines.presets.basic import pipeline

        results = pipeline.run(
            cervical_mini_dataset,
            level=1,
            patch_size=256,
            tissue_method="per_patch_canny_ranker",
        )
        assert len(results) == 3  # 3 samples
        for ps in results:
            assert isinstance(ps, PatchSet)
            assert len(ps.frame) > 0
            assert "keep" in ps.frame.columns
            assert "annotation_label" in ps.frame.columns
            assert "tissue_score" in ps.frame.columns
            assert len(ps.contexts) == 1

    def test_pipeline_patch_size_override(self, cervical_mini_dataset):
        """Verify param overrides propagate through the pipeline."""
        from histokit.pipelines.presets.basic import pipeline

        results = pipeline.run(
            cervical_mini_dataset,
            level=1,
            patch_size=512,
            tissue_method="per_patch_canny_ranker",
        )

        for ps in results:
            assert ps.contexts[0].patch_size == 512
