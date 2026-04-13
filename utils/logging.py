from __future__ import annotations

import logging
from pathlib import Path

from . import paths

LOGGER_NAME = "hallway_avatar_gen"
_INITIALIZED = False


def _configure_logger(log_path: Path) -> logging.Logger:
    global _INITIALIZED
    logger = logging.getLogger(LOGGER_NAME)
    if _INITIALIZED:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _INITIALIZED = True
    return logger


def get_logger(name: str | None = None, configured_cache_dir: str = "") -> logging.Logger:
    base_logger = _configure_logger(paths.log_file_path(configured_cache_dir))
    if not name:
        return base_logger
    return base_logger.getChild(name)
