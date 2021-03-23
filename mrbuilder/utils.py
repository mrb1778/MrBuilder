import inspect
from typing import Iterable

import re


def get_params(fun) -> dict:
    sig = inspect.signature(fun)
    return {p.name: p.default
            for p in sig.parameters.values()
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default != inspect.Parameter.empty}


def remove_keys(items: dict, keys: Iterable[str]):
    for key in keys:
        if key in items:
            del items[key]


def getattr_ignore_case(obj, attr: str):
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)


def repeat_if_single(value, num_times=2):
    if is_single(value):
        return [value] * num_times
    else:
        return value


def is_single(value):
    return not isinstance(value, (list, tuple))


_camel_snake_regex_name_letters = re.compile('(.)([A-Z][a-z]+)')
_camel_snake_regex_name_numbers = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake(name):
    name = re.sub(_camel_snake_regex_name_letters, r'\1_\2', name)
    return re.sub(_camel_snake_regex_name_numbers, r'\1_\2', name).lower()
