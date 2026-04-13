"""Tests for histokit.utils (geometry, convert, filters)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from histokit.utils.geometry import Point, PointF, Address, Size, Shape
from histokit.utils.convert import invert, pil_to_np, np_to_pil, to_frame_with_locations
from histokit.utils.filters import compute_padding, pool2d, PoolMode


# ── Geometry types ───────────────────────────────────────────────────────────

class TestPoint:
    def test_fields(self):
        p = Point(3, 7)
        assert p.x == 3
        assert p.y == 7

    def test_tuple_access(self):
        p = Point(1, 2)
        assert p[0] == 1
        assert p[1] == 2


class TestPointF:
    def test_float_values(self):
        p = PointF(1.5, 2.5)
        assert p.x == 1.5
        assert p.y == 2.5


class TestAddress:
    def test_fields(self):
        a = Address(row=4, col=5)
        assert a.row == 4
        assert a.col == 5


class TestSize:
    def test_fields(self):
        s = Size(width=640, height=480)
        assert s.width == 640
        assert s.height == 480

    def test_as_shape(self):
        s = Size(640, 480)
        sh = s.as_shape()
        assert sh == Shape(480, 640)


class TestShape:
    def test_fields(self):
        s = Shape(num_rows=10, num_cols=20)
        assert s.num_rows == 10
        assert s.num_cols == 20

    def test_as_size(self):
        s = Shape(10, 20)
        assert s.as_size() == Size(20, 10)


# ── Convert utilities ────────────────────────────────────────────────────────

class TestInvert:
    def test_basic(self):
        assert invert({"a": 1, "b": 2}) == {1: "a", 2: "b"}

    def test_empty(self):
        assert invert({}) == {}


class TestPilNpConversion:
    def test_pil_to_np(self):
        img = Image.new("RGB", (10, 10), color=(128, 64, 32))
        arr = pil_to_np(img)
        assert arr.shape == (10, 10, 3)
        assert arr[0, 0, 0] == 128

    def test_np_to_pil_uint8(self):
        arr = np.zeros((5, 5, 3), dtype=np.uint8)
        arr[..., 0] = 200
        img = np_to_pil(arr)
        assert img.size == (5, 5)

    def test_np_to_pil_bool(self):
        arr = np.array([[True, False], [False, True]])
        img = np_to_pil(arr)
        px = np.asarray(img)
        assert px[0, 0] == 255
        assert px[0, 1] == 0

    def test_np_to_pil_float(self):
        arr = np.array([[0.0, 0.5], [1.0, 0.25]])
        img = np_to_pil(arr)
        px = np.asarray(img)
        assert px[0, 0] == 0
        assert px[1, 0] == 255

    def test_roundtrip(self):
        arr = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img = np_to_pil(arr)
        arr2 = pil_to_np(img)
        np.testing.assert_array_equal(arr, arr2)


class TestToFrameWithLocations:
    def test_basic(self):
        arr = np.array([[10, 20], [30, 40]])
        df = to_frame_with_locations(arr, "val")
        assert len(df) == 4
        assert set(df.columns) == {"row", "column", "val"}
        assert df.loc[0, "row"] == 0
        assert df.loc[0, "column"] == 0
        assert df.loc[0, "val"] == 10

    def test_custom_column_name(self):
        arr = np.ones((2, 3))
        df = to_frame_with_locations(arr, "tissue_score")
        assert "tissue_score" in df.columns
        assert len(df) == 6


# ── Filters ──────────────────────────────────────────────────────────────────

class TestComputePadding:
    def test_no_padding_needed(self):
        pad = compute_padding((10, 10), kernel_size=2, stride=2)
        assert pad == (0, 0)

    def test_padding_needed(self):
        pad_h, pad_w = compute_padding((11, 11), kernel_size=2, stride=2)
        assert isinstance(pad_h, int)
        assert isinstance(pad_w, int)


class TestPool2d:
    def test_max_pool(self):
        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]], dtype=float)
        result = pool2d(arr, kernel_size=2, stride=2, pool_mode=PoolMode.MAX)
        assert result.shape == (2, 2)
        assert result[0, 0] == 6
        assert result[0, 1] == 8
        assert result[1, 0] == 14
        assert result[1, 1] == 16

    def test_avg_pool(self):
        arr = np.ones((4, 4), dtype=float) * 4.0
        result = pool2d(arr, kernel_size=2, stride=2, pool_mode=PoolMode.AVG)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, 4.0)

    def test_min_pool(self):
        arr = np.array([[1, 2], [3, 4]], dtype=float)
        result = pool2d(arr, kernel_size=2, stride=2, pool_mode=PoolMode.MIN)
        assert result.shape == (1, 1)
        assert result[0, 0] == 1

    def test_with_padding(self):
        arr = np.ones((3, 3), dtype=float) * 5.0
        result = pool2d(arr, kernel_size=2, stride=2, padding=1, pool_mode=PoolMode.MAX)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_invalid_pool_mode(self):
        arr = np.ones((4, 4), dtype=float)
        with pytest.raises(ValueError, match="Unknown pool_mode"):
            pool2d(arr, kernel_size=2, stride=2, pool_mode="bad")
