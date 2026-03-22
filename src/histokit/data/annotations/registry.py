registry = {}


def register_annotation(name: str):
    def decorator(func):
        if name in registry:
            raise ValueError(f"Slide type '{name}' already registered")
        registry[name] = func

    return decorator


def get_annotation_loader(name: str):
    return registry[name]
