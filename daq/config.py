# -*- coding: utf-8 -*-
"""Runtime configuration for DAQ."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


_DEFAULT_DATA_FOLDER = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True, slots=True)
class Settings:
    """Container for runtime settings."""

    presto_address: str
    presto_port: Optional[int]
    data_folder: Path
    mongodb_uri: str
    mongodb_db_name: str
    mongodb_collection_name: str
    fgen_resource: Optional[str]
    led_resource: Optional[str]
    visa_backend: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with package defaults."""
        return cls(
            presto_address=os.getenv("DAQ_PRESTO_ADDRESS", "172.23.20.29"),
            presto_port=_parse_optional_int(
                os.getenv("DAQ_PRESTO_PORT"),
                env_var="DAQ_PRESTO_PORT",
            ),
            data_folder=_parse_data_folder(os.getenv("DAQ_DATA_FOLDER")),
            mongodb_uri=os.getenv("DAQ_MONGODB_URI", "mongodb://localhost:27017"),
            mongodb_db_name=os.getenv("DAQ_MONGODB_DB_NAME", "WashU_Astroparticle_Detector"),
            mongodb_collection_name=os.getenv("DAQ_MONGODB_COLLECTION_NAME", "measurement"),
            fgen_resource=_parse_optional_str(os.getenv("DAQ_FGEN_RESOURCE")),
            led_resource=_parse_optional_str(os.getenv("DAQ_LED_RESOURCE")),
            visa_backend=os.getenv("DAQ_VISA_BACKEND", ""),
        )


def _parse_optional_int(value: Optional[str], *, env_var: str) -> Optional[int]:
    """Parse an optional integer env var."""
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer or empty, got {value!r}.") from exc


def _parse_optional_str(value: Optional[str]) -> Optional[str]:
    """Return a stripped string, or ``None`` when unset or blank."""
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _parse_data_folder(value: Optional[str]) -> Path:
    """Parse data folder from env var or return package default."""
    if value is None or value.strip() == "":
        return _DEFAULT_DATA_FOLDER
    return Path(value).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings.from_env()


def reload_settings() -> Settings:
    """Clear cached settings and reload from environment variables."""
    get_settings.cache_clear()
    return get_settings()


def get_presto_address() -> str:
    """Return the configured Presto device address."""
    return get_settings().presto_address


def get_presto_port() -> Optional[int]:
    """Return the configured Presto device port."""
    return get_settings().presto_port


def get_data_folder() -> str:
    """Return the configured data folder path as a string."""
    return str(get_settings().data_folder)


def get_mongodb_uri() -> str:
    """Return the configured MongoDB URI."""
    return get_settings().mongodb_uri


def get_mongodb_db_name() -> str:
    """Return the configured MongoDB database name."""
    return get_settings().mongodb_db_name


def get_mongodb_collection_name() -> str:
    """Return the configured MongoDB collection name."""
    return get_settings().mongodb_collection_name


def get_fgen_resource() -> Optional[str]:
    """Return the configured function-generator VISA resource, or ``None`` to autodiscover."""
    return get_settings().fgen_resource


def get_led_resource() -> Optional[str]:
    """Return the configured LED-driver VISA resource, or ``None`` to autodiscover."""
    return get_settings().led_resource


def get_visa_backend() -> str:
    """Return the configured PyVISA backend (``""`` for NI-VISA, ``"@py"`` for pyvisa-py)."""
    return get_settings().visa_backend
