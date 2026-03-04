# -*- coding: utf-8 -*-
"""Datetime helper functions."""

from datetime import date, datetime


def get_date_str() -> str:
    """Return today's date as YYYYMMDD."""
    today = date.today()
    return f"{today.year}{today.month:02d}{today.day:02d}"


def get_date_str_with_time() -> str:
    """Return current local date-time as YYYYMMDD_HHMMSS."""
    now = datetime.now()
    return (
        f"{now.year}{now.month:02d}{now.day:02d}_"
        f"{now.hour:02d}{now.minute:02d}{now.second:02d}"
    )
