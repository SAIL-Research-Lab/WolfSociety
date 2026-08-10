"""Lightweight logging helper."""
from __future__ import annotations
import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "wolfbench", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        root = logging.getLogger("wolfbench")
        root.handlers = [h]
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
