# -*- coding: utf-8 -*-
"""Noise analysis utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from daq.measurements.sweep import Sweep


def compute_psd(
    data: npt.ArrayLike,
    fs: float,
    welch: bool = False,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    detrend: str | bool = "constant",
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the Power Spectral Density of a real-valued time series.

    By default the bare periodogram (direct FFT, no windowing or detrending) is
    used.  Setting ``welch=True`` switches to Welch's method
    (:func:`scipy.signal.welch`), which averages the periodograms of overlapping
    windowed segments to reduce variance at the cost of frequency resolution.

    For 1-D input the result is a single PSD.  For 2-D input each row is
    treated as an independent time series and the result has shape
    ``(nrows, nfreqs)``.

    :param data: Real-valued time series data. 1-D or 2-D array where each row
        is a separate time series. Complex input is not supported.
    :param fs: The sampling frequency in Hz.
    :param welch: When ``True``, use Welch's method instead of the bare
        periodogram. Defaults to ``False``.
    :param nperseg: Length of each Welch segment. Only used when ``welch`` is
        ``True``; defaults to scipy's choice (256 or the series length).
    :param noverlap: Number of points to overlap between Welch segments. Only
        used when ``welch`` is ``True``; defaults to ``nperseg // 2``.
    :param window: Window passed to :func:`scipy.signal.welch`. Only used when
        ``welch`` is ``True``. Defaults to ``"hann"``.
    :param detrend: Detrending applied to each Welch segment. Only used when
        ``welch`` is ``True``. Defaults to ``"constant"``.
    :returns: ``(f, psd)`` — sample frequencies in Hz and power spectral density in
        (units of *data*)²/Hz.
    :raises TypeError: If *data* is complex.
    :raises ValueError: If *data* is empty or has more than 2 dimensions.
    """
    data = np.asarray(data, dtype=np.float64)
    if np.iscomplexobj(data):
        raise TypeError("compute_psd only supports real-valued input; got complex data")
    if data.ndim == 0 or data.shape[-1] == 0:
        raise ValueError("data must be a non-empty 1-D or 2-D array")
    if data.ndim > 2:
        raise ValueError(f"data must be 1-D or 2-D, got {data.ndim}-D")

    if welch:
        from scipy.signal import welch as _welch

        f, psd = _welch(
            data,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            axis=-1,
        )
        return f, psd

    N = data.shape[-1]

    # Compute the FFT for real-valued input along the last axis
    fft_values = np.fft.rfft(data, axis=-1)

    # Compute the frequency bins
    f = np.fft.rfftfreq(N, d=1 / fs)

    # Compute the raw PSD: |FFT|^2 / (fs * N), avoiding unnecessary sqrt from np.abs
    psd = (fft_values.real**2 + fft_values.imag**2) / (fs * N)

    # Convert to a one-sided spectrum.
    # Multiply by 2 to conserve energy (dropped negative frequencies),
    # but do NOT multiply the DC component (0 Hz) or the Nyquist frequency.
    if N % 2 == 0:
        # Even length: DC is index 0, Nyquist is the last index
        psd[..., 1:-1] *= 2
    else:
        # Odd length: DC is index 0, no Nyquist bin in rfft
        psd[..., 1:] *= 2

    return f, psd


def from_elec_to_reson(ts: npt.NDArray[np.complexfloating], sw: Sweep) -> tuple[
    npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Transform raw I/Q data from the electronic basis to the resonator basis.

    Uses the fitted sweep parameters to remove environmental effects and
    project the complex time-stream onto the resonator coordinate system.

    :param ts: Complex time-stream data (USB output of a :class:`TimeStream`
        measurement).
    :param sw: A :class:`~daq.measurements.sweep.Sweep` instance whose
        ``fit_results`` provide the calibration parameters.
    :returns: ``(tsz, rad, arc)`` where *tsz* is the complex resonator-basis
        coordinate, *rad* is the dissipation (radial) response, and *arc* is
        the frequency (arc-length) response.
    """
    fr_fit_idx = np.argmin(np.abs(sw.freq_arr - sw.fit_results["fr"]))
    tsz = (ts / sw.fit_results["environmental_term"][fr_fit_idx] - 1) / np.exp(
        1j * sw.fit_results["phi0"]
    ) + 1

    q_ratio = sw.fit_results["Qc_dia_corr"] ** 2 / sw.fit_results["Qi_dia_corr"]
    rad = tsz.real / q_ratio
    arc = tsz.imag / (-2 * q_ratio)

    return tsz, rad, arc


def remove_correlated_noise(
    on_res: npt.NDArray[np.complexfloating],
    off_res: npt.NDArray[np.complexfloating],
    fs: float,
    min_t_s: Optional[float] = None,
    max_t_s: Optional[float] = None,
    return_r_rho: bool = False,
) -> (
    tuple[npt.NDArray[np.complexfloating], float, float]
    | tuple[
        npt.NDArray[np.complexfloating],
        float,
        float,
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]
):
    """Remove correlated electronics noise using an off-resonance reference tone.

    Implements the cleaning procedure of Eqn 7.44–7.45 in Wen (2025).
    Decomposition is performed in the gain / arc-length basis:
    ``r = |z|`` (gain) and ``rho = angle(z) * mean(|z|)`` (arc length).
    For each component, a cleaning coefficient ``x = Cov(D, S) / Var(S)``
    is computed and the scaled off-resonance signal is subtracted.

    :param on_res: Complex 1-D time series from the on-resonance tone.
    :param off_res: Complex 1-D time series from the off-resonance tone (same length).
    :param fs: Sampling frequency in Hz.
    :param min_t_s: If given, start time (seconds) of the window used to compute
        the cleaning coefficients. Subtraction is still applied to the full array.
    :param max_t_s: If given, end time (seconds) of the window.
    :param return_r_rho: When ``True``, also return the mean-subtracted cleaned
        ``r`` and ``rho`` arrays (useful for computing PSDs).
    :returns: ``(cleaned, x_r, x_rho)`` by default, or
        ``(cleaned, x_r, x_rho, cleaned_r, cleaned_rho)`` when *return_r_rho* is
        ``True``.
    :raises TypeError: If inputs are not complex.
    :raises ValueError: If inputs are not 1-D or have different lengths.
    """
    on_res = np.asarray(on_res)
    off_res = np.asarray(off_res)

    if not np.iscomplexobj(on_res) or not np.iscomplexobj(off_res):
        raise TypeError("on_res and off_res must be complex arrays")
    if on_res.ndim != 1 or off_res.ndim != 1:
        raise ValueError("on_res and off_res must be 1-D arrays")
    if on_res.shape[0] != off_res.shape[0]:
        raise ValueError("on_res and off_res must have the same length")
    if on_res.shape[0] == 0:
        raise ValueError("on_res and off_res must be non-empty")

    N = on_res.shape[0]

    # --- Decompose into gain (r) and arc-length (rho) ---
    r_on = np.abs(on_res)
    r_off = np.abs(off_res)
    mean_r_on = np.mean(r_on)
    mean_r_off = np.mean(r_off)
    theta_on = np.angle(on_res)
    rho_on = theta_on * mean_r_on
    rho_off = np.angle(off_res) * mean_r_off

    # --- Mean-subtract ---
    mean_theta_on = np.mean(theta_on)
    r_on_c = r_on - mean_r_on
    r_off_c = r_off - mean_r_off
    mean_rho_on = np.mean(rho_on)
    mean_rho_off = np.mean(rho_off)
    rho_on_c = rho_on - mean_rho_on
    rho_off_c = rho_off - mean_rho_off

    # --- Time mask for cleaning coefficient computation ---
    mask = np.ones(N, dtype=bool)
    if min_t_s is not None or max_t_s is not None:
        t = np.arange(N) / fs
        if min_t_s is not None:
            mask &= t >= min_t_s
        if max_t_s is not None:
            mask &= t <= max_t_s

    # --- Cleaning coefficients (Eqn 7.45) ---
    if mask.sum() < 2:
        raise ValueError(
            "Time window selects fewer than 2 samples; cannot compute cleaning coefficients"
        )

    def _cleaning_coeff(d: npt.NDArray[np.floating], s: npt.NDArray[np.floating]) -> float:
        C = np.cov(d[mask], s[mask])
        var_s = C[1, 1]
        if var_s == 0.0 or not np.isfinite(var_s):
            return 0.0
        return float(C[0, 1] / var_s)

    x_r = _cleaning_coeff(r_on_c, r_off_c)
    x_rho = _cleaning_coeff(rho_on_c, rho_off_c)

    # --- Clean full arrays (Eqn 7.44) ---
    cleaned_r = r_on_c - x_r * r_off_c
    cleaned_rho = rho_on_c - x_rho * rho_off_c

    # --- Reconstruct complex signal ---
    cleaned_abs = cleaned_r + mean_r_on
    cleaned_theta = cleaned_rho / mean_r_on + mean_theta_on
    cleaned = cleaned_abs * np.exp(1j * cleaned_theta)

    if return_r_rho:
        return cleaned, x_r, x_rho, cleaned_r, cleaned_rho
    return cleaned, x_r, x_rho
