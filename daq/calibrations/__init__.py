# -*- coding: utf-8 -*-
"""Power calibration utilities.

``amp`` is the DAC voltage amplitude as a fraction of full scale, so at a fixed frequency the
output power follows

``power_dbm = power_dbm_at_amp_1 + 20 * log10(|amp|)``.

The unfiltered October 2026 asset covers 0.1--9 GHz at 251 measured frequencies.
Its power reference is the analyzer input through the unfiltered cable path, without cable-loss
correction. It is a single-port calibration, not a calibration of every possible DAC setting.

Frequency interpolation is allowed only across measured intervals marked safe by the builder.
Intervals touching DAC-switch uncertainty windows or adjacent model-level jumps over 1 dB
are rejected. Exact measured frequencies remain available. This conservative policy
avoids inventing exact switch boundaries from a 5 MHz software scan.

Most low-band nodes were measured only at amplitude 0.1. Other amplitudes there return a model
estimate with CalibrationWarning. At swept nodes, conversions outside the empirical supported
amplitude range also warn. This range is not an absolute accuracy specification.
See docs/power_calibration_update.md for provenance, limitations and validation.

"""

import warnings
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

_DATA_PATH = Path(__file__).parent / "power_calibration.npz"

FloatArray = Union[float, npt.NDArray[np.floating]]
_Calibration = Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64],
]


class CalibrationWarning(UserWarning):
    """A conversion fell outside the region the calibration measurements verify."""


@lru_cache(maxsize=1)
def _load_calibration() -> _Calibration:
    """Load and validate the packaged calibration.

    :returns: ``(frequency_ghz, power_dbm_at_amp_1, amp_floor, interpolation_allowed, amp_ceiling)``; the
        first three arrays are
        indexed by calibrated frequency.  ``amp_floor`` is the smallest amplitude at which the
        measurements at that frequency still follow the ``20 log10(amp)`` law within the
        builder's tolerance.
    :raises RuntimeError: If the asset is missing, corrupt or internally inconsistent.
    """
    try:
        with np.load(_DATA_PATH, allow_pickle=False) as data:
            if data["schema_version"].shape != () or data["schema_version"].item() != 3:
                raise ValueError("Expected unfiltered schema version 3; rebuild the asset.")
            frequency_ghz = np.asarray(data["frequency_ghz"], dtype=np.float64)
            power_dbm_at_amp_1 = np.asarray(data["power_dbm_at_amp_1"], dtype=np.float64)
            amp_floor = np.asarray(data["amp_floor"], dtype=np.float64)
            interpolation_allowed = np.asarray(data["interpolation_allowed"])
            amp_ceiling = np.asarray(data["amp_ceiling"], dtype=np.float64)
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Could not load a valid power calibration from '{_DATA_PATH}'."
        ) from exc

    if frequency_ghz.ndim != 1 or frequency_ghz.size < 2:
        raise RuntimeError(
            "Calibration frequencies must form a one-dimensional array of at least two points."
        )
    if power_dbm_at_amp_1.shape != frequency_ghz.shape or amp_floor.shape != frequency_ghz.shape or amp_ceiling.shape != frequency_ghz.shape:
        raise RuntimeError("Calibration arrays must hold one value per calibrated frequency.")
    if not (
        np.all(np.isfinite(frequency_ghz))
        and np.all(np.isfinite(power_dbm_at_amp_1))
        and np.all(np.isfinite(amp_floor))
        and np.all(np.isfinite(amp_ceiling))
    ):
        raise RuntimeError("Calibration arrays must be finite.")
    if np.any(frequency_ghz <= 0.0) or not np.all(np.diff(frequency_ghz) > 0.0):
        raise RuntimeError("Calibration frequencies must be positive and strictly increasing.")
    if np.any(amp_floor <= 0.0) or np.any(amp_floor > 1.0):
        raise RuntimeError("Calibration amplitude floors must lie in (0, 1].")
    if np.any(amp_ceiling < amp_floor) or np.any(amp_ceiling > 1.0):
        raise RuntimeError("Calibration amplitude ceilings must lie between the floor and 1.")

    if interpolation_allowed.shape != (frequency_ghz.size - 1,) or (
        interpolation_allowed.dtype != np.dtype(bool)
    ):
        raise RuntimeError("Calibration must contain one boolean interpolation flag per interval.")

    return frequency_ghz, power_dbm_at_amp_1, amp_floor, interpolation_allowed, amp_ceiling


def _bracket(freq_ghz: float) -> Tuple[int, int]:
    """Return the indices of the calibrated frequencies bracketing *freq_ghz*.

    Both indices are equal when *freq_ghz* coincides with a calibrated frequency.

    :raises ValueError: If *freq_ghz* is non-finite or outside a supported calibration interval.
    """
    frequency_ghz, _, _, allowed, _ = _load_calibration()
    if not np.isfinite(freq_ghz):
        raise ValueError("Frequency must be finite.")
    # Absorb only floating-point roundoff from Hz/GHz conversion (a few ULPs), not
    # a physical frequency offset. This preserves exact nodes beside excluded intervals.
    nearest = int(np.argmin(np.abs(frequency_ghz - freq_ghz)))
    tolerance = 4 * np.finfo(float).eps * max(1.0, abs(freq_ghz))
    if abs(freq_ghz - frequency_ghz[nearest]) <= tolerance:
        return nearest, nearest
    if freq_ghz < frequency_ghz[0] or freq_ghz > frequency_ghz[-1]:
        raise ValueError(
            f"Frequency {freq_ghz:.6g} GHz is outside the calibrated range "
            f"[{frequency_ghz[0]:.6g}, {frequency_ghz[-1]:.6g}] GHz."
        )
    hi = int(np.searchsorted(frequency_ghz, freq_ghz))
    lo = hi - 1
    if not allowed[lo]:
        raise ValueError(
            f"Frequency {freq_ghz:.6g} GHz lies in an unverified interpolation interval "
            f"({frequency_ghz[lo]:.6g}, {frequency_ghz[hi]:.6g}) GHz: a DAC-switch "
            "uncertainty window or a measured level jump. Use a measured frequency or "
            "acquire calibration for the actual DAC configuration; do not blend across it."
        )
    return lo, hi


def _full_scale_power_dbm(freq_ghz: float) -> float:
    """Interpolate the model full-scale level (inferred at single-anchor nodes)."""
    frequency_ghz, powers, _, _, _ = _load_calibration()
    lo, hi = _bracket(freq_ghz)
    if lo == hi:
        return float(powers[lo])
    weight = (freq_ghz - frequency_ghz[lo]) / (frequency_ghz[hi] - frequency_ghz[lo])
    return float(powers[lo] + weight * (powers[hi] - powers[lo]))


def min_verified_amp(freq_ghz: float) -> float:
    """Lower bound of the empirical amplitude range; also check ``verified_amp_range``.

    A lower bound alone does not imply support at all higher amplitudes: most low-band
    nodes have an upper bound of 0.1 as well.
    Between two calibrated frequencies the larger of their floors is used, so the answer is
    conservative.  Below this empirical amplitude floor, fit residuals or acquisition review flags
    prevent verification. The conversions extrapolate the law and warn with
    :class:`CalibrationWarning`. Between measured frequencies the floor is itself an estimate,
    not verification at the requested frequency.

    :param freq_ghz: Carrier frequency in GHz, within the calibrated band.
    :returns: Amplitude as a fraction of full scale.
    :raises ValueError: If *freq_ghz* is non-finite or outside a supported calibration interval.
    """
    return verified_amp_range(freq_ghz)[0]


def verified_amp_range(freq_ghz: float) -> Tuple[float, float]:
    """Return empirical amplitude bounds, not an absolute accuracy guarantee.

    Most low-band nodes return (0.1, 0.1): only this amplitude was measured.
    Sparse checks at 0.1 and 1 GHz support the model over 0.01--1;
    gap/high-band nodes use the reviewed sweep floor through amplitude 1.
    Between nodes, return the intersection of their bounds. This is a conservative
    estimate, not validation at an unmeasured frequency.
    """
    _, _, floor, _, ceiling = _load_calibration()
    lo, hi = _bracket(freq_ghz)
    return float(max(floor[lo], floor[hi])), float(min(ceiling[lo], ceiling[hi]))


def _check_amp(amp: FloatArray) -> npt.NDArray[np.float64]:
    """Return ``|amp|`` as a float array after validating the hardware range."""
    amps = np.abs(np.asarray(amp, dtype=np.float64))
    if not np.all(np.isfinite(amps)):
        raise ValueError("Amplitude must be finite.")
    if np.any(amps <= 0.0) or np.any(amps > 1.0):
        raise ValueError("Amplitude must satisfy 0 < |amp| <= 1 (fraction of DAC full scale).")
    return amps


def _warn_below_floor(freq_ghz: float, amps: npt.NDArray[np.float64]) -> None:
    """Warn outside either empirical amplitude bound (legacy internal name)."""
    floor, ceiling = verified_amp_range(freq_ghz)
    tolerance = 8 * np.finfo(float).eps
    outside = (amps < floor * (1-tolerance)) | (amps > ceiling * (1+tolerance))
    if np.any(outside):
        warnings.warn(
            f"Amplitude at {freq_ghz:.4f} GHz is outside the empirically supported "
            f"range [{floor:.6g}, {ceiling:.6g}]. "
            + ("Only amplitude 0.1 was directly measured at a supporting frequency. " if floor == ceiling else "")
            + "The returned 20 dB/decade model estimate is unverified at this amplitude.",
            CalibrationWarning,
            stacklevel=3,
        )


def amp_to_power_dbm(freq_ghz: float, amp: FloatArray) -> FloatArray:
    """Convert DAC voltage amplitude to analyzer-input power for the calibration setup.

    :param freq_ghz: Carrier frequency in GHz, within the calibrated band.
    :param amp: DAC voltage amplitude as a fraction of full scale, scalar or array.  The sign is
        a phase flip on the Presto, so only ``|amp|`` enters.
    :returns: Output power in dBm: a float for scalar input, otherwise an array of the same
        shape as *amp*.
    :raises ValueError: If the frequency is non-finite or outside a supported calibration interval, or any
        amplitude is non-finite, zero or above full scale.
    :warns CalibrationWarning: If any amplitude is outside :func:`verified_amp_range` at that
        frequency; the value is still returned.
    """
    amps = _check_amp(amp)
    full_scale_power = _full_scale_power_dbm(freq_ghz)
    _warn_below_floor(freq_ghz, amps)
    result = full_scale_power + 20.0 * np.log10(amps)
    return float(result) if result.ndim == 0 else result


def amp_to_power_dbm_hz(freq_hz: float, amp: FloatArray) -> FloatArray:
    """Convert DAC voltage amplitude to output power, accepting the frequency in Hz.

    :param freq_hz: Carrier frequency in Hz, within the calibrated band.
    :param amp: DAC voltage amplitude as a fraction of full scale, scalar or array.
    :returns: Output power in dBm, as :func:`amp_to_power_dbm`.
    :raises ValueError: As :func:`amp_to_power_dbm`.
    :warns CalibrationWarning: As :func:`amp_to_power_dbm`.
    """
    return amp_to_power_dbm(freq_hz / 1e9, amp)


def power_dbm_to_amp(freq_ghz: float, power_dbm: float) -> float:
    """Convert a desired output power in dBm to the DAC voltage amplitude that produces it.

    :param freq_ghz: Carrier frequency in GHz, within the calibrated band.
    :param power_dbm: Desired output power in dBm, at most the full-scale power at that
        frequency.
    :returns: DAC voltage amplitude as a fraction of full scale, in ``(0, 1]``.
    :raises ValueError: If the frequency is non-finite or outside a supported calibration interval, or the
        power is non-finite or above what full-scale drive delivers.
    :warns CalibrationWarning: If the resulting amplitude is outside :func:`verified_amp_range` at
        that frequency; the value is still returned.
    """
    if not np.isfinite(power_dbm):
        raise ValueError("Power must be finite.")
    full_scale_power = _full_scale_power_dbm(freq_ghz)
    if power_dbm > full_scale_power:
        raise ValueError(
            f"Requested power {power_dbm:.1f} dBm exceeds the full-scale output "
            f"{full_scale_power:.1f} dBm at {freq_ghz:.4f} GHz."
        )
    amp = min(10.0 ** ((power_dbm - full_scale_power) / 20.0), 1.0)
    if amp == 0.0:
        raise ValueError("Requested power underflows the representable positive amplitude range.")
    _warn_below_floor(freq_ghz, np.asarray(amp, dtype=np.float64))
    return float(amp)
