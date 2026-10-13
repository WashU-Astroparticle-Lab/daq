# -*- coding: utf-8 -*-
"""Power calibration and nominal fridge-loss utilities.

``amp`` is the DAC voltage amplitude as a fraction of full scale, so at a fixed frequency the
output power follows

``power_dbm = power_dbm_at_amp_1 + 20 * log10(|amp|)``.

The packaged calibration (``power_calibration.npz``, built by
``scripts/build_power_calibration.py`` from the spectrum-analyzer sweeps committed under
``source_data/``) stores, per calibrated frequency, the measured full-scale power and the
smallest amplitude at which the measurements still follow that law.  Below that floor the
analyzer's channel-power reading flattens onto its own noise (about -45 to -55 dBm depending on
the session) while the DAC keeps scaling, so the data there verify nothing.  Conversions below
the floor are still returned -- the law is what a linear DAC does -- but they raise a
:class:`CalibrationWarning` so the caller knows the number is an extrapolation.

Frequency dependence is linear interpolation between the calibrated frequencies.  The Presto
switches DAC mode at several frequencies inside the calibrated band and the measured full-scale
power steps by up to 5.5 dB across those switch points, so interpolating across one is a known
limitation; see the Calibrations section of ``CLAUDE.md``.
"""

import warnings
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

_DATA_PATH = Path(__file__).parent / "power_calibration.npz"
_FRIDGE_LOSS_PATH = Path(__file__).parent / "source_data" / "Empty_Line_port_2_-_2026-06-15.s2p"

FloatArray = Union[float, npt.NDArray[np.floating]]
_Calibration = Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]


class CalibrationWarning(UserWarning):
    """A conversion fell outside the region the calibration measurements verify."""


@lru_cache(maxsize=1)
def _load_fridge_loss() -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read the packaged port-2 measurement (Hz, S-parameters in dB/angle)."""
    try:
        with _FRIDGE_LOSS_PATH.open(encoding="utf-8") as stream:
            if stream.readline().strip().upper() != "# HZ S DB R 50":
                raise ValueError("Expected the packaged Hz/dB Touchstone format.")
            data = np.loadtxt(stream, comments="!", ndmin=2)
        if data.shape[0] < 2 or data.shape[1] != 9:
            raise ValueError("Expected frequency and four dB/angle pairs per row.")
        frequency_hz = data[:, 0]
        # The file labels its pairs S11, S21, S12, S22; loss is minus S21 in dB.
        loss_db = -data[:, 3]
        if not (
            np.all(np.isfinite(frequency_hz))
            and np.all(np.isfinite(loss_db))
            and np.all(frequency_hz > 0.0)
            and np.all(np.diff(frequency_hz) > 0.0)
        ):
            raise ValueError("Expected finite losses and positive, increasing frequencies.")
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load a valid fridge-loss calibration from '{_FRIDGE_LOSS_PATH}'."
        ) from exc
    return frequency_hz, loss_db


@lru_cache(maxsize=1)
def _smoothed_fridge_loss() -> npt.NDArray[np.float64]:
    """Smooth the uniformly spaced measurement in dB, including polynomial edge fits."""
    from scipy.signal import savgol_filter

    _, loss_db = _load_fridge_loss()
    return savgol_filter(loss_db, window_length=21, polyorder=3, mode="interp")


def nominal_fridge_loss(freq_hz: FloatArray, *, smooth: bool = True) -> FloatArray:
    """Return nominal fridge insertion loss in dB at a frequency in Hz.

    Smooths ``-S21_dB`` from the empty-line port-2 measurement dated 2026-06-15 with a
    21-point cubic Savitzky-Golay filter in dB, then linearly interpolates. The window
    spans about 200 MHz; edges use polynomial fits (``mode="interp"``). Smoothing is
    applied to the full calibration before evaluating the requested frequencies.
    This is the loss of the measured path, with no additional correction
    for the input/output lines. Subtract it from source power in dBm to apply that loss.

    :param freq_hz: Frequency in Hz, scalar or array, from 1e5 to 1e10 inclusive.
    :param smooth: Apply smoothing by default. Set to ``False`` to interpolate the
        original measured dB values exactly, without smoothing.
    :returns: Loss in dB (positive for attenuation), a float for scalar input or an
        array with the input shape.
    :raises ValueError: If any frequency is non-finite or outside the measured band.
        No extrapolation is performed.
    :raises RuntimeError: If the packaged measurement is missing or invalid.
    """
    frequencies = np.asarray(freq_hz, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Frequency must be finite.")
    measured_freqs, loss_db = _load_fridge_loss()
    if np.any((frequencies < measured_freqs[0]) | (frequencies > measured_freqs[-1])):
        raise ValueError(
            f"Frequency must be within the measured fridge-loss range "
            f"[{measured_freqs[0]:g}, {measured_freqs[-1]:g}] Hz."
        )
    if smooth:
        loss_db = _smoothed_fridge_loss()
    result = np.interp(frequencies, measured_freqs, loss_db)
    return float(result) if frequencies.ndim == 0 else result


@lru_cache(maxsize=1)
def _load_calibration() -> _Calibration:
    """Load and validate the packaged calibration.

    :returns: ``(frequency_ghz, power_dbm_at_amp_1, amp_floor)``, three one-dimensional arrays
        indexed by calibrated frequency.  ``amp_floor`` is the smallest amplitude at which the
        measurements at that frequency still follow the ``20 log10(amp)`` law within the
        builder's tolerance.
    :raises RuntimeError: If the asset is missing, corrupt or internally inconsistent.
    """
    try:
        with np.load(_DATA_PATH) as data:
            frequency_ghz = np.asarray(data["frequency_ghz"], dtype=np.float64)
            power_dbm_at_amp_1 = np.asarray(data["power_dbm_at_amp_1"], dtype=np.float64)
            amp_floor = np.asarray(data["amp_floor"], dtype=np.float64)
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Could not load a valid power calibration from '{_DATA_PATH}'."
        ) from exc

    if frequency_ghz.ndim != 1 or frequency_ghz.size < 2:
        raise RuntimeError(
            "Calibration frequencies must form a one-dimensional array of at least two points."
        )
    if power_dbm_at_amp_1.shape != frequency_ghz.shape or amp_floor.shape != frequency_ghz.shape:
        raise RuntimeError("Calibration arrays must hold one value per calibrated frequency.")
    if not (
        np.all(np.isfinite(frequency_ghz))
        and np.all(np.isfinite(power_dbm_at_amp_1))
        and np.all(np.isfinite(amp_floor))
    ):
        raise RuntimeError("Calibration arrays must be finite.")
    if not np.all(np.diff(frequency_ghz) > 0.0):
        raise RuntimeError("Calibration frequencies must be strictly increasing.")
    if np.any(amp_floor <= 0.0) or np.any(amp_floor > 1.0):
        raise RuntimeError("Calibration amplitude floors must lie in (0, 1].")

    return frequency_ghz, power_dbm_at_amp_1, amp_floor


def _bracket(freq_ghz: float) -> Tuple[int, int]:
    """Return the indices of the calibrated frequencies bracketing *freq_ghz*.

    Both indices are equal when *freq_ghz* coincides with a calibrated frequency.

    :raises ValueError: If *freq_ghz* is non-finite or outside the calibrated band.
    """
    frequency_ghz, _, _ = _load_calibration()
    if not np.isfinite(freq_ghz):
        raise ValueError("Frequency must be finite.")
    if freq_ghz < frequency_ghz[0] or freq_ghz > frequency_ghz[-1]:
        raise ValueError(
            f"Frequency {freq_ghz:.4f} GHz is outside the calibrated range "
            f"[{frequency_ghz[0]:.4f}, {frequency_ghz[-1]:.4f}] GHz."
        )
    lo = int(np.searchsorted(frequency_ghz, freq_ghz, side="right")) - 1
    lo = max(lo, 0)
    hi = lo if freq_ghz == frequency_ghz[lo] else min(lo + 1, frequency_ghz.size - 1)
    return lo, hi


def _full_scale_power_dbm(freq_ghz: float) -> float:
    """Interpolate the measured full-scale power at *freq_ghz*."""
    frequency_ghz, power_dbm_at_amp_1, _ = _load_calibration()
    _bracket(freq_ghz)
    return float(np.interp(freq_ghz, frequency_ghz, power_dbm_at_amp_1))


def min_verified_amp(freq_ghz: float) -> float:
    """Smallest amplitude at which the calibration data verify the ``20 log10(amp)`` law.

    Between two calibrated frequencies the larger of their floors is used, so the answer is
    conservative.  Below this amplitude the measured power sits on the analyzer floor and the
    conversions extrapolate the law; they warn with :class:`CalibrationWarning`.

    :param freq_ghz: Carrier frequency in GHz, within the calibrated band.
    :returns: Amplitude as a fraction of full scale.
    :raises ValueError: If *freq_ghz* is non-finite or outside the calibrated band.
    """
    _, _, amp_floor = _load_calibration()
    lo, hi = _bracket(freq_ghz)
    return float(max(amp_floor[lo], amp_floor[hi]))


def _check_amp(amp: FloatArray) -> npt.NDArray[np.float64]:
    """Return ``|amp|`` as a float array after validating the hardware range."""
    amps = np.abs(np.asarray(amp, dtype=np.float64))
    if not np.all(np.isfinite(amps)):
        raise ValueError("Amplitude must be finite.")
    if np.any(amps <= 0.0) or np.any(amps > 1.0):
        raise ValueError("Amplitude must satisfy 0 < |amp| <= 1 (fraction of DAC full scale).")
    return amps


def _warn_below_floor(freq_ghz: float, amps: npt.NDArray[np.float64]) -> None:
    """Emit :class:`CalibrationWarning` if any amplitude lies below the verified floor."""
    floor = min_verified_amp(freq_ghz)
    below = amps < floor
    if np.any(below):
        warnings.warn(
            f"amp {float(np.min(amps[below])):.3g} at {freq_ghz:.4f} GHz is below {floor:.3g}, "
            "the smallest amplitude down to which the calibration measurements confirm the "
            "20 dB/decade law (the analyzer floor hides the tone below it); the conversion "
            "extrapolates that law and is unverified.",
            CalibrationWarning,
            stacklevel=3,
        )


def amp_to_power_dbm(freq_ghz: float, amp: FloatArray) -> FloatArray:
    """Convert DAC voltage amplitude to calibrated output power in dBm.

    :param freq_ghz: Carrier frequency in GHz, within the calibrated band.
    :param amp: DAC voltage amplitude as a fraction of full scale, scalar or array.  The sign is
        a phase flip on the Presto, so only ``|amp|`` enters.
    :returns: Output power in dBm: a float for scalar input, otherwise an array of the same
        shape as *amp*.
    :raises ValueError: If the frequency is non-finite or outside the calibrated band, or any
        amplitude is non-finite, zero or above full scale.
    :warns CalibrationWarning: If any amplitude is below :func:`min_verified_amp` at that
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
    return amp_to_power_dbm(freq_hz * 1e-9, amp)


def power_dbm_to_amp(freq_ghz: float, power_dbm: float) -> float:
    """Convert a desired output power in dBm to the DAC voltage amplitude that produces it.

    :param freq_ghz: Carrier frequency in GHz, within the calibrated band.
    :param power_dbm: Desired output power in dBm, at most the full-scale power at that
        frequency.
    :returns: DAC voltage amplitude as a fraction of full scale, in ``(0, 1]``.
    :raises ValueError: If the frequency is non-finite or outside the calibrated band, or the
        power is non-finite or above what full-scale drive delivers.
    :warns CalibrationWarning: If the resulting amplitude is below :func:`min_verified_amp` at
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
    _warn_below_floor(freq_ghz, np.asarray(amp, dtype=np.float64))
    return float(amp)
