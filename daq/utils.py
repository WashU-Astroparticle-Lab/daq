# -*- coding: utf-8 -*-
"""Backward-compatible utility API.

This module is retained for compatibility. New code should import configuration
from :mod:`daq.config` and date helpers from :mod:`daq.time_utils`.
"""

from __future__ import annotations

import warnings
from typing import Optional

from .config import (
    get_data_folder as _get_data_folder,
    get_presto_address as _get_presto_address,
    get_presto_port as _get_presto_port,
    get_settings,
)
from .time_utils import get_date_str, get_date_str_with_time

_settings = get_settings()
PRESTO_ADDRESS = _settings.presto_address
PRESTO_PORT = _settings.presto_port
DATA_FOLDER = str(_settings.data_folder)


def _warn_deprecated(name: str, replacement: str) -> None:
    """Emit deprecation warning for compatibility wrappers."""
    warnings.warn(
        f"daq.utils.{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def get_presto_address() -> str:
    """Return configured Presto address (compat wrapper)."""
    _warn_deprecated("get_presto_address", "daq.config.get_presto_address")
    return _get_presto_address()


def get_presto_port() -> Optional[int]:
    """Return configured Presto port (compat wrapper)."""
    _warn_deprecated("get_presto_port", "daq.config.get_presto_port")
    return _get_presto_port()


def get_data_folder() -> str:
    """Return configured data folder path (compat wrapper)."""
    _warn_deprecated("get_data_folder", "daq.config.get_data_folder")
    return _get_data_folder()


__all__ = [
    "PRESTO_ADDRESS",
    "PRESTO_PORT",
    "DATA_FOLDER",
    "get_date_str",
    "get_date_str_with_time",
    "get_presto_address",
    "get_presto_port",
    "get_data_folder",
]
