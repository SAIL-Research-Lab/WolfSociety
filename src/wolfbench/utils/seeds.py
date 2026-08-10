"""Deterministic RNG helpers."""
from __future__ import annotations
import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def split_rng(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Spawn n independent child generators from a parent."""
    seeds = rng.integers(0, 2**31 - 1, size=n)
    return [np.random.default_rng(int(s)) for s in seeds]
