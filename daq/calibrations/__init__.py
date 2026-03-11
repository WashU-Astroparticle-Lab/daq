# -*- coding: utf-8 -*-
"""
Power calibration utilities.

Translates between DAC full-scale amplitude (``amp``) and calibrated output
power in dBm using packaged calibration grid data.

The preferred calibration asset is a NumPy ``.npz`` archive containing the
calibration grids (``f_grid``, ``a_grid``, ``Z``). A legacy cloudpickle-based
asset is still supported as a fallback for compatibility, but the interpolator
is always rebuilt from raw arrays at runtime.
"""

from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt

_DATA_PATH = Path(__file__).parent / "power_calibration.npz"
_LEGACY_PICKLE_PATH = Path(__file__).parent / "power_cal_interpolator.pkl"

FloatArray = Union[float, npt.NDArray[np.floating]]


@lru_cache(maxsize=1)
def _load_calibration():
    """Load calibration grids and build a scipy interpolator.

    The preferred data source is a packaged ``.npz`` archive with ``f_grid``
    (GHz), ``a_grid`` (full-scale amplitude), and ``Z`` (power in dBm). For
    backward compatibility, a legacy cloudpickle-based calibration asset is
    used only if the stable archive is unavailable.

    :returns: ``(interpolator, f_grid, a_grid)``
    """
    from scipy.interpolate import RegularGridInterpolator

    if _DATA_PATH.exists():
        with np.load(_DATA_PATH) as data:
            f_grid = np.asarray(data["f_grid"], dtype=np.float64)
            a_grid = np.asarray(data["a_grid"], dtype=np.float64)
            Z = np.asarray(data["Z"], dtype=np.float64)
    else:
        try:
            import cloudpickle

            with _LEGACY_PICKLE_PATH.open("rb") as f:
                fn = cloudpickle.load(f)
        except Exception as exc:
            raise RuntimeError(
                "Calibration data could not be loaded. "
                f"Expected '{_DATA_PATH.name}' or a compatible legacy "
                f"'{_LEGACY_PICKLE_PATH.name}'."
            ) from exc

        g = fn.__globals__
        f_grid = np.asarray(g["f_grid"], dtype=np.float64)
        a_grid = np.asarray(g["a_grid"], dtype=np.float64)
        Z = np.asarray(g["Z"], dtype=np.float64)

    interp = RegularGridInterpolator(
        (f_grid, a_grid), Z, method="linear", bounds_error=True
    )
    return interp, f_grid, a_grid


def amp_to_power_dbm(freq_ghz: float, amp: FloatArray) -> FloatArray:
    """Convert DAC full-scale amplitude to calibrated output power in dBm.

    :param freq_ghz: Carrier frequency in GHz.
    :param amp: DAC amplitude (fraction of full scale), scalar or array.
    :returns: Output power in dBm, same shape as *amp*.
    :raises ValueError: If *freq_ghz* or any *amp* value is outside the
        calibration grid.
    """
    interp, _, _ = _load_calibration()
    amp = np.asarray(amp, dtype=np.float64)
    scalar = amp.ndim == 0
    amp = np.atleast_1d(amp)
    pts = np.column_stack([np.full_like(amp, freq_ghz), amp])
    result = interp(pts)
    return float(result[0]) if scalar else result


def amp_to_power_dbm_hz(freq_hz: float, amp: FloatArray) -> FloatArray:
    """Convert DAC full-scale amplitude to calibrated output power in dBm.

    Convenience wrapper that accepts frequency in Hz (converted to GHz
    internally).

    :param freq_hz: Carrier frequency in Hz.
    :param amp: DAC amplitude (fraction of full scale), scalar or array.
    :returns: Output power in dBm, same shape as *amp*.
    :raises ValueError: If frequency or any *amp* value is outside the
        calibration grid.
    """
    return amp_to_power_dbm(freq_hz * 1e-9, amp)


def power_dbm_to_amp(freq_ghz: float, power_dbm: float) -> float:
    """Convert a desired output power in dBm back to DAC amplitude.

    Uses root-finding (Brent's method) assuming that at a given frequency,
    amplitude and power are monotonically related.

    :param freq_ghz: Carrier frequency in GHz.
    :param power_dbm: Desired output power in dBm.
    :returns: DAC amplitude (fraction of full scale).
    :raises ValueError: If the requested power is outside the calibrated range.
    """
    from scipy.optimize import brentq

    _, _, a_grid = _load_calibration()
    amp_lo, amp_hi = float(a_grid[0]), float(a_grid[-1])
    p_lo = amp_to_power_dbm(freq_ghz, amp_lo)
    p_hi = amp_to_power_dbm(freq_ghz, amp_hi)

    # Ensure bracket covers the target
    if power_dbm < min(p_lo, p_hi) or power_dbm > max(p_lo, p_hi):
        raise ValueError(
            f"Requested power {power_dbm:.1f} dBm is outside the calibrated range "
            f"[{min(p_lo, p_hi):.1f}, {max(p_lo, p_hi):.1f}] dBm at {freq_ghz:.4f} GHz."
        )

    return float(
        brentq(lambda a: amp_to_power_dbm(freq_ghz, a) - power_dbm, amp_lo, amp_hi)
    )
