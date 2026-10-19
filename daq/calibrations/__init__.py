# -*- coding: utf-8 -*-
"""Power conversions for the analyzer-input reference plane and nominal fridge loss.

0.1–2.1 GHz uses the full original amplitude curves corrected by a per-frequency
filter offset. These are estimates of unfiltered power, not direct unfiltered sweeps.
Above 2.1 GHz the direct unfiltered sweep model retains the 20 dB/decade law.
Low-band interpolation is linear in dBm versus log amplitude, preserving supported
measured curve shape. Frequency interpolation never crosses stored exclusions.
See docs/power_calibration_update.md for correction assumptions and provenance.
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
_FRIDGE_INPUT_LOSS_PATH = Path(__file__).parent / "source_data" / "fridge_loss_to_mc.csv"

FloatArray = Union[float, npt.NDArray[np.floating]]
_Calibration = Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]


class CalibrationWarning(UserWarning):
    """A conversion has a calibration-provenance or amplitude-support limitation."""


class FilterOffsetWarning(CalibrationWarning):
    """Power uses a filter-offset-corrected estimate of the unfiltered low band."""


@lru_cache(maxsize=2)
def _load_fridge_loss(
    include_output: bool = False,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read the digitized input loss or the original cold loopback measurement."""
    path = _FRIDGE_LOSS_PATH if include_output else _FRIDGE_INPUT_LOSS_PATH
    try:
        if include_output:
            with path.open(encoding="utf-8") as stream:
                if stream.readline().strip().upper() != "# HZ S DB R 50":
                    raise ValueError("Expected the packaged Hz/dB Touchstone format.")
                data = np.loadtxt(stream, comments="!", ndmin=2)
            if data.shape[0] < 2 or data.shape[1] != 9:
                raise ValueError("Expected frequency and four dB/angle pairs per row.")
            # The file labels its pairs S11, S21, S12, S22; loss is minus S21 in dB.
            loss_db = -data[:, 3]
        else:
            data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
            if data.shape[0] < 2 or data.shape[1] != 2:
                raise ValueError("Expected frequency in Hz and positive loss in dB per row.")
            loss_db = data[:, 1]
        frequency_hz = data[:, 0]
        if not (
            np.all(np.isfinite(frequency_hz))
            and np.all(np.isfinite(loss_db))
            and np.all(frequency_hz > 0.0)
            and np.all(np.diff(frequency_hz) > 0.0)
        ):
            raise ValueError("Expected finite losses and positive, increasing frequencies.")
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load a valid fridge-loss calibration from '{path}'."
        ) from exc
    return frequency_hz, loss_db


@lru_cache(maxsize=2)
def _smoothed_fridge_loss(include_output: bool = False) -> npt.NDArray[np.float64]:
    """Smooth the uniformly spaced measurement in dB, including polynomial edge fits."""
    from scipy.signal import savgol_filter

    _, loss_db = _load_fridge_loss(include_output)
    # Both windows span ~200 MHz: ~50 MHz input spacing, ~10 MHz loopback spacing.
    window_length = 21 if include_output else 5
    return savgol_filter(loss_db, window_length=window_length, polyorder=3, mode="interp")


def nominal_fridge_loss(
    freq_hz: FloatArray, *, smooth: bool = True, include_output: bool = False
) -> FloatArray:
    """Return nominal fridge insertion loss in dB at a frequency in Hz.

    By default, use the room-temperature measurement of the input line down to the
    mixing chamber (MC), digitized from the red trace in the supplied screenshot.
    Set ``include_output=True`` for the original cold empty-line port-2 loopback
    measurement dated 2026-06-15, which includes the output line's contribution.
    These are separate measurements at different temperatures, not an additive
    estimate of the output-line loss.

    Apply a cubic Savitzky-Golay filter in dB, then linearly interpolate. The window
    spans about 200 MHz (5 points for the digitized input trace, 21 for the loopback);
    edges use polynomial fits (``mode="interp"``). Smoothing is applied to the full
    selected calibration before evaluating the requested frequencies. Subtract the
    default loss from source power in dBm to estimate power at MC. The screenshot
    calibration is approximate, not the original numerical instrument data.

    :param freq_hz: Frequency in Hz, scalar or array, from 1e5 to 1e10 inclusive.
    :param smooth: Apply smoothing by default. Set to ``False`` to interpolate the
        selected source's dB values exactly, without smoothing.
    :param include_output: Use the cold loopback including the output line. Defaults
        to ``False`` for the room-temperature input-line measurement down to MC.
    :returns: Loss in dB (positive for attenuation), a float for scalar input or an
        array with the input shape.
    :raises ValueError: If any frequency is non-finite or outside the measured band.
        No extrapolation is performed.
    :raises RuntimeError: If the packaged measurement is missing or invalid.
    """
    frequencies = np.asarray(freq_hz, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Frequency must be finite.")
    measured_freqs, loss_db = _load_fridge_loss(include_output)
    if np.any((frequencies < measured_freqs[0]) | (frequencies > measured_freqs[-1])):
        raise ValueError(
            f"Frequency must be within the measured fridge-loss range "
            f"[{measured_freqs[0]:g}, {measured_freqs[-1]:g}] Hz."
        )
    if smooth:
        loss_db = _smoothed_fridge_loss(include_output)
    result = np.interp(frequencies, measured_freqs, loss_db)
    return float(result) if frequencies.ndim == 0 else result


@lru_cache(maxsize=1)
def _load_calibration() -> _Calibration:
    """Load frequencies, full-scale powers, bounds, exclusions and tabulated curves."""
    try:
        with np.load(_DATA_PATH, allow_pickle=False) as data:
            if data["schema_version"].shape != () or data["schema_version"].item() != 4:
                raise ValueError(
                    "Expected filter-offset curve schema version 4; rebuild the asset."
                )
            frequency_ghz = np.asarray(data["frequency_ghz"], dtype=np.float64)
            power_dbm_at_amp_1 = np.asarray(data["power_dbm_at_amp_1"], dtype=np.float64)
            amp_floor = np.asarray(data["amp_floor"], dtype=np.float64)
            interpolation_allowed = np.asarray(data["interpolation_allowed"])
            amp_ceiling = np.asarray(data["amp_ceiling"], dtype=np.float64)
            grid = np.asarray(data["curve_log_amplitude"], dtype=np.float64)
            curves = np.asarray(data["curve_power_dbm"], dtype=np.float64)
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Could not load a valid power calibration from '{_DATA_PATH}'."
        ) from exc

    if frequency_ghz.ndim != 1 or frequency_ghz.size < 2:
        raise RuntimeError(
            "Calibration frequencies must form a one-dimensional array of at least two points."
        )
    if (
        power_dbm_at_amp_1.shape != frequency_ghz.shape
        or amp_floor.shape != frequency_ghz.shape
        or amp_ceiling.shape != frequency_ghz.shape
    ):
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

    if (
        grid.ndim != 1
        or len(grid) < 2
        or not np.isfinite(grid).all()
        or not np.all(np.diff(grid) > 0)
        or grid[-1] != 0
        or curves.shape != (len(frequency_ghz), len(grid))
        or not np.isfinite(curves).all()
        or np.any(np.log10(amp_floor) < grid[0])
        or np.any(amp_ceiling != 1)
        or not np.array_equal(curves[:, -1], power_dbm_at_amp_1)
    ):
        raise RuntimeError("Invalid calibration curve grid, power matrix or endpoint metadata.")
    for row, floor in zip(curves, amp_floor):
        x = np.unique(np.r_[np.log10(floor), grid[grid >= np.log10(floor)]])
        if len(x) < 2 or not np.all(np.diff(np.interp(x, grid, row)) > 0):
            raise RuntimeError(
                "Supported calibration curves must be strictly monotonic for inversion."
            )
    return (
        frequency_ghz,
        power_dbm_at_amp_1,
        amp_floor,
        interpolation_allowed,
        amp_ceiling,
        grid,
        curves,
    )


def _bracket(freq_ghz: float) -> Tuple[int, int]:
    """Return the indices of the calibrated frequencies bracketing *freq_ghz*.

    Both indices are equal when *freq_ghz* coincides with a calibrated frequency.

    :raises ValueError: If *freq_ghz* is non-finite or outside a supported calibration interval.
    """
    frequency_ghz, _, _, allowed, _, _, _ = _load_calibration()
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


def _frequency_curve(freq_ghz: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a monotonic supported curve; use the common amplitude range off-grid."""
    f, _, floors, _, _, grid, curves = _load_calibration()
    lo, hi = _bracket(freq_ghz)
    floor = float(max(floors[lo], floors[hi]))
    x = np.unique(np.r_[np.log10(floor), grid[grid >= np.log10(floor)]])
    lower = np.interp(x, grid, curves[lo])
    if lo == hi:
        return x, lower
    weight = (freq_ghz - f[lo]) / (f[hi] - f[lo])
    return x, lower + weight * (np.interp(x, grid, curves[hi]) - lower)


def _full_scale_power_dbm(freq_ghz: float) -> float:
    """Return the corrected or directly calibrated full-scale power."""
    return float(_frequency_curve(freq_ghz)[1][-1])


def min_verified_amp(freq_ghz: float) -> float:
    """Return the lower empirical support bound; low-band support is conditional on correction.

    This compatibility name does not mean direct unfiltered validation. See
    :func:`verified_amp_range` and :class:`FilterOffsetWarning`.

    :param freq_ghz: Carrier frequency in GHz.
    :returns: Dimensionless amplitude floor.
    """
    return verified_amp_range(freq_ghz)[0]


def verified_amp_range(freq_ghz: float) -> Tuple[float, float]:
    """Return source-quality amplitude bounds, not an absolute accuracy guarantee.

    At 0.1–2.1 GHz these bounds apply to the corrected filtered curve and assume an
    amplitude-independent offset. They do not claim a direct unfiltered amplitude sweep.
    Elsewhere they apply to the direct sweep model. Off-grid bounds are the intersection
    of endpoint bounds. Below the floor, conversions warn and extrapolate at 20 dB/decade.

    :param freq_ghz: Carrier frequency in GHz.
    :returns: Lower and upper dimensionless amplitude bounds.
    """
    _, _, floors, _, ceilings, _, _ = _load_calibration()
    lo, hi = _bracket(freq_ghz)
    return float(max(floors[lo], floors[hi])), float(min(ceilings[lo], ceilings[hi]))


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
    lo, _ = _bracket(freq_ghz)
    if _load_calibration()[0][lo] <= 2.1:
        warnings.warn(
            "0.1–2.1 GHz power is calculated by filter offset from the original filtered "
            "amplitude curves. It is an unfiltered estimate assuming amplitude-independent "
            "loss; offsets may also include path/session drift.",
            FilterOffsetWarning,
            stacklevel=3,
        )
    tolerance = 8 * np.finfo(float).eps
    outside = (amps < floor * (1 - tolerance)) | (amps > ceiling * (1 + tolerance))
    if np.any(outside):
        warnings.warn(
            f"Amplitude at {freq_ghz:.4f} GHz is outside the empirically supported "
            f"range [{floor:.6g}, {ceiling:.6g}]. "
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
        frequency; the value is still returned. Low-band calls also emit FilterOffsetWarning.
    """
    amps = _check_amp(amp)
    x, powers = _frequency_curve(freq_ghz)
    _warn_below_floor(freq_ghz, amps)
    log_amp = np.log10(amps)
    result = np.asarray(np.interp(log_amp, x, powers))
    result = np.where(log_amp < x[0], powers[0] + 20 * (log_amp - x[0]), result)
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
        that frequency; the value is still returned. Low-band calls also emit FilterOffsetWarning.
    """
    if not np.isfinite(power_dbm):
        raise ValueError("Power must be finite.")
    full_scale_power = _full_scale_power_dbm(freq_ghz)
    if power_dbm > full_scale_power:
        raise ValueError(
            f"Requested power {power_dbm:.1f} dBm exceeds the full-scale output "
            f"{full_scale_power:.1f} dBm at {freq_ghz:.4f} GHz."
        )
    x, powers = _frequency_curve(freq_ghz)
    log_amp = (
        x[0] + (power_dbm - powers[0]) / 20
        if power_dbm < powers[0]
        else float(np.interp(power_dbm, powers, x))
    )
    amp = min(10.0**log_amp, 1.0)
    if amp == 0.0:
        raise ValueError("Requested power underflows the representable positive amplitude range.")
    _warn_below_floor(freq_ghz, np.asarray(amp, dtype=np.float64))
    return float(amp)
