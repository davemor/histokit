from pathlib import Path
from typing import List

from PIL import Image
from tiffslide import TiffSlide

from histokit.io.slides.registry import register_slide
from histokit.io.slides.slide import Region, SlideBase
from histokit.utils.geometry import Size


@register_slide("tiffslide", extensions=[".ome.tiff", ".ome.tif"])
class TiffSlideSlide(SlideBase):
    def __init__(self, slide_path: Path) -> None:
        super().__init__(slide_path)

    def open(self) -> None:
        self.tsr = TiffSlide(str(self.path))

    def close(self) -> None:
        self.tsr.close()

    @property
    def dimensions(self) -> List[Size]:
        return [Size(*dim) for dim in self.tsr.level_dimensions]

    def level_downsamples(self) -> List[float]:
        return list(self.tsr.level_downsamples)

    def read_region(self, region: Region) -> Image.Image:
        image = self.tsr.read_region(region.location, region.level, region.size)
        return image

    def read_regions(self, regions: List[Region]) -> list[Image.Image]:
        region_images = [self.read_region(region) for region in regions]
        return region_images
