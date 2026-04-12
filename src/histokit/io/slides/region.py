from typing import NamedTuple

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from histokit.utils.geometry import Point, Size


class Region(NamedTuple):
    level: int
    location: Point
    size: Size

    @classmethod
    def patch(cls, x, y, size, level):
        location = Point(x, y)
        size = Size(size, size)
        return cls(level, location, size)

    @classmethod
    def make(cls, x, y, width, height, level):
        location = Point(x, y)
        size = Size(width, height)
        return cls(level, location, size)

    @property
    def x(self) -> float:
        return self.location.x

    @property
    def y(self) -> float:
        return self.location.y

    @property
    def width(self) -> float:
        return self.size.width

    @property
    def height(self) -> float:
        return self.size.height

    def to_level0_geometry(self, downsample: float) -> BaseGeometry:
        """
        Convert this region to a level-0 Shapely rectangle.
        `downsample` should be the slide downsample factor for self.level.
        """
        x0 = self.x * downsample
        y0 = self.y * downsample
        w0 = self.width * downsample
        h0 = self.height * downsample
        return box(x0, y0, x0 + w0, y0 + h0)

    def area_level0(self, downsample: float) -> float:
        return self.width * self.height * downsample * downsample
