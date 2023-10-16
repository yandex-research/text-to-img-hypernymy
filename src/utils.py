"""Utility functions."""

from typing import Generator


def flatten_list_generator(items: list) -> Generator:
    """Generates flattened list."""
    for item in items:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def flatten_list(items: list) -> list:
    """Flattens nested lists."""
    return list(flatten_list_generator(items))


def unique_list(items: list) -> list:
    """Makes a unique list."""
    return list(set(items))
