# Registry for TissueDetector factory functions
from collections.abc import Callable

from histokit.pipelines.segmentation.detector import TissueDetector


TissueDetectorFactory = Callable[..., "TissueDetector"]

tissue_detector_registry: dict[str, TissueDetectorFactory] = {}

def register_tissue_detector(name: str) -> Callable[[TissueDetectorFactory], TissueDetectorFactory]:
    """Decorator to register a function that returns a TissueDetector subclass.

    The returned detector's ``name`` property is automatically set to *name*.
    """
    def decorator(func: TissueDetectorFactory) -> TissueDetectorFactory:
        def wrapper(*args: object, **kwargs: object) -> TissueDetector:
            detector = func(*args, **kwargs)
            detector.name = name
            return detector
        tissue_detector_registry[name] = wrapper
        return wrapper
    return decorator


def get_tissue_detector(name: str) -> TissueDetectorFactory:
    """Get a registered TissueDetector factory function by name."""
    if name not in tissue_detector_registry:
        raise ValueError(f"TissueDetector '{name}' not found in registry. Available: {list(tissue_detector_registry.keys())}")
    return tissue_detector_registry[name]