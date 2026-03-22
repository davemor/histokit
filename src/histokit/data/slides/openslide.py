from pathlib import Path
from typing import List

from PIL import Image
from openslide import open_slide

from histokit.data.slides.registry import register_slide
from histokit.data.slides.slide import Region, SlideBase
from histokit.utils.geometry import Size


@register_slide("openslide")
class OpenSlideSlide(SlideBase):
    def __init__(self, slide_path: Path):
        super().__init__()
        self.slide_path = slide_path

    def open(self) -> None:
        self.osr = open_slide(str(self.path))

    def close(self) -> None:
        self.osr.close()

    @property
    def path(self) -> Path:
        return self.slide_path

    @property
    def dimensions(self) -> List[Size]:
        return [Size(*dim) for dim in self.osr.level_dimensions]

    @property
    def level_downsamples(self):
        return self.osr.level_downsamples

    def read_region(self, region: Region) -> Image.Image:
        image = self.osr.read_region(region.location, region.level, region.size)
        return image

    def read_regions(self, regions: List[Region]) -> list[Image.Image]:
        # TODO: this call could be parallelised
        # though pytorch loaders will do this for us
        region_images = [self.read_region(region) for region in regions]
        return region_images
