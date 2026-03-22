from .slide import SlideBase

registry = {}


def register_slide(name: str):
    def decorator(cls: type["SlideBase"]):
        if name in registry:
            raise ValueError(f"Slide type '{name}' already registered")
        registry[name] = cls

    return decorator


def get_slide_cls(name: str) -> type["SlideBase"]:
    return registry[name]
