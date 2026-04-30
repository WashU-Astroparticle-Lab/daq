# -*- coding: utf-8 -*-
"""Noise analysis utilities."""

import numpy as np
import numpy.typing as npt


def compute_psd(
    data: npt.ArrayLike, fs: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the Power Spectral Density using the Periodogram method (Direct FFT).

    No windowing or detrending is applied (bare periodogram).

    For 1-D input the result is a single PSD.  For 2-D input each row is
    treated as an independent time series and the result has shape
    ``(nrows, nfreqs)``.

    :param data: Real-valued time series data. 1-D or 2-D array where each row
        is a separate time series. Complex input is not supported.
    :param fs: The sampling frequency in Hz.
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

    N = data.shape[-1]

    # Compute the FFT for real-valued input along the last axis
    fft_values = np.fft.rfft(data, axis=-1)

    # Compute the frequency bins
    f = np.fft.rfftfreq(N, d=1 / fs)

    # Compute the raw PSD: |FFT|^2 / (fs * N), avoiding unnecessary sqrt from np.abs
    psd = (fft_values.real ** 2 + fft_values.imag ** 2) / (fs * N)

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
