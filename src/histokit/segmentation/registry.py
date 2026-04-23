# Registry for Detector factory functions
from collections.abc import Callable

from histokit.segmentation.detector import Detector


DetectorFactory = Callable[..., "Detector"]

detector_registry: dict[str, DetectorFactory] = {}

def register_detector(name: str) -> Callable[[DetectorFactory], DetectorFactory]:
    """Decorator to register a function that returns a Detector subclass.

    The returned detector's ``name`` property is automatically set to *name*.
    """
    def decorator(func: DetectorFactory) -> DetectorFactory:
        def wrapper(*args: object, **kwargs: object) -> Detector:
            detector = func(*args, **kwargs)
            detector.name = name
            return detector
        detector_registry[name] = wrapper
        return wrapper
    return decorator


def get_detector(name: str) -> DetectorFactory:
    """Get a registered Detector factory function by name."""
    if name not in detector_registry:
        raise ValueError(f"Detector '{name}' not found in registry. Available: {list(detector_registry.keys())}")
    return detector_registry[name]