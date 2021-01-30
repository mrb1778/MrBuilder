from typing import Callable

import functools


def register_layer_wrapper(registry, name: str, *aliases):
    def decorator_register_layer(func: Callable):
        registry[name.lower()] = func
        for alias in aliases:
            registry[alias.lower()] = func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    return decorator_register_layer
