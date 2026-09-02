# -*- coding: utf-8 -*-
"""Power calibration utilities.

``amp`` is the DAC voltage amplitude as a fraction of full scale.  At a fixed
frequency, output power therefore follows

``power_dbm = power_dbm_at_amp_1 + 20 * log10(amp)``.

The packaged calibration contains the measured full-scale power versus
frequency.  Keeping amplitude scaling analytic avoids interpolating sparse or
noise-floor-limited power measurements on an inappropriate linear-amplitude
grid.
"""

from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt

_DATA_PATH = Path(__file__).parent / "power_calibration.npz"

FloatArray = Union[float, npt.NDArray[np.floating]]


@lru_cache(maxsize=1)
def _load_calibration() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
    """Load and validate the packaged full-scale calibration."""
    try:
        with np.load(_DATA_PATH) as data:
            frequency_ghz = np.asarray(data["frequency_ghz"], dtype=np.float64)
            power_dbm_at_amp_1 = np.asarray(data["power_dbm_at_amp_1"], dtype=np.float64)
            amp_min = float(data["amp_min"])
            amp_max = float(data["amp_max"])
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load a valid power calibration from '{_DATA_PATH}'."
        ) from exc

    if frequency_ghz.ndim != 1 or power_dbm_at_amp_1.ndim != 1:
        raise RuntimeError("Calibration frequency and power arrays must be one-dimensional.")
    if frequency_ghz.size < 2 or frequency_ghz.shape != power_dbm_at_amp_1.shape:
        raise RuntimeError("Calibration frequency and power arrays have incompatible sizes.")
    if not np.all(np.isfinite(frequency_ghz)) or not np.all(np.isfinite(power_dbm_at_amp_1)):
        raise RuntimeError("Calibration frequency and power arrays must be finite.")
    if not np.all(np.diff(frequency_ghz) > 0.0):
        raise RuntimeError("Calibration frequencies must be strictly increasing.")
    if not (0.0 < amp_min < amp_max <= 1.0):
        raise RuntimeError("Calibration amplitude limits must satisfy 0 < min < max <= 1.")

    return frequency_ghz, power_dbm_at_amp_1, amp_min, amp_max


def _full_scale_power_dbm(freq_ghz: float) -> float:
    """Interpolate measured full-scale power at *freq_ghz*."""
    frequency_ghz, power_dbm_at_amp_1, _, _ = _load_calibration()
    if not np.isfinite(freq_ghz):
        raise ValueError("Frequency must be finite.")
    if freq_ghz < frequency_ghz[0] or freq_ghz > frequency_ghz[-1]:
        raise ValueError(
            f"Frequency {freq_ghz:.4f} GHz is outside the calibrated range "
            f"[{frequency_ghz[0]:.4f}, {frequency_ghz[-1]:.4f}] GHz."
        )
    return float(np.interp(freq_ghz, frequency_ghz, power_dbm_at_amp_1))


def amp_to_power_dbm(freq_ghz: float, amp: FloatArray) -> FloatArray:
    """Convert DAC voltage amplitude to calibrated output power in dBm.

    :param freq_ghz: Carrier frequency in GHz.
    :param amp: DAC voltage amplitude (fraction of full scale), scalar or array.
    :returns: Output power in dBm, with the same shape as *amp*.
    :raises ValueError: If frequency or amplitude is outside the calibrated range.
    """
    _, _, amp_min, amp_max = _load_calibration()
    amps = np.asarray(amp, dtype=np.float64)
    if not np.all(np.isfinite(amps)):
        raise ValueError("Amplitude must be finite.")
    if np.any(amps < amp_min) or np.any(amps > amp_max):
        raise ValueError(f"Amplitude is outside the calibrated range [{amp_min:g}, {amp_max:g}].")

    result = _full_scale_power_dbm(freq_ghz) + 20.0 * np.log10(amps)
    return float(result) if result.ndim == 0 else result


def amp_to_power_dbm_hz(freq_hz: float, amp: FloatArray) -> FloatArray:
    """Convert DAC voltage amplitude to output power, accepting frequency in Hz."""
    return amp_to_power_dbm(freq_hz * 1e-9, amp)


def power_dbm_to_amp(freq_ghz: float, power_dbm: float) -> float:
    """Convert calibrated output power in dBm to DAC voltage amplitude."""
    _, _, amp_min, amp_max = _load_calibration()
    if not np.isfinite(power_dbm):
        raise ValueError("Power must be finite.")

    full_scale_power = _full_scale_power_dbm(freq_ghz)
    power_min = full_scale_power + 20.0 * np.log10(amp_min)
    power_max = full_scale_power + 20.0 * np.log10(amp_max)
    if power_dbm < power_min or power_dbm > power_max:
        raise ValueError(
            f"Requested power {power_dbm:.1f} dBm is outside the calibrated range "
            f"[{power_min:.1f}, {power_max:.1f}] dBm at {freq_ghz:.4f} GHz."
        )
    amp = 10.0 ** ((power_dbm - full_scale_power) / 20.0)
    return float(np.clip(amp, amp_min, amp_max))
