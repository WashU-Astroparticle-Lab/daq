# -*- coding: utf-8 -*-
"""Folding utilities for periodically-driven time streams."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from daq.measurements.timestream import TimeStream


def _as_complex(
    data: Union["TimeStream", npt.ArrayLike],
    tone: int = 0,
) -> npt.NDArray[np.complexfloating]:
    """Coerce a time stream or raw array into a 1-D complex array.

    :param data: A :class:`~daq.measurements.timestream.TimeStream` (its ``signal`` is used)
        or an array of complex samples.
    :param tone: Which tone to take when *data* is a time stream or a 2-D array.
    :raises ValueError: If the array has more than two dimensions.
    :returns: A 1-D complex array of samples.

    """
    if hasattr(data, "signal"):
        array = np.asarray(data.signal)
    else:
        array = np.asarray(data)
    if array.ndim == 2:
        return array[:, tone]
    if array.ndim == 1:
        return array
    raise ValueError(f"Expected a 1-D or 2-D array of samples, got shape {array.shape}")


def fold_timestream(
    data: Union["TimeStream", npt.ArrayLike],
    fs: float,
    *,
    period_s: Optional[float] = None,
    n_periods: Optional[int] = None,
    tone: int = 0,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Block-average a periodically-driven time stream into a single drive period.

    Used for the sawtooth-biased QC trace: the bias generator repeats a ramp at a fixed rate
    while the time stream records continuously, so averaging the record in blocks of one ramp
    period beats the uncorrelated noise down by ``sqrt(n_periods)`` and leaves the device's
    response to one sweep of the gate voltage.

    Specify the period either directly with *period_s* or, equivalently, by how many whole
    periods the record spans with *n_periods*. Samples left over after the last whole period
    are dropped.

    The input must already be trimmed of acquisition start-up junk;
    :class:`~daq.measurements.timestream.TimeStream` does this itself via ``discard_start_ms``,
    so its in-memory ``signal`` can be passed straight in.

    :param data: A time stream, or an array of complex samples.
    :param fs: Sample rate in hertz (``TimeStream.df``).
    :param period_s: Drive period in seconds, e.g. ``1 / ramp_freq_hz``.
    :param n_periods: Number of whole drive periods the record spans.
    :param tone: Which tone to fold, for a multi-tone time stream.
    :raises ValueError: If neither or both of *period_s* and *n_periods* are given, or if the
        record is too short to contain one whole period.
    :returns: ``(time_ms, avg_iq)`` where ``time_ms`` is the time axis of one period in
        milliseconds and ``avg_iq`` has shape ``(2, n_samples)`` holding the averaged I and Q.

    """
    if (period_s is None) == (n_periods is None):
        raise ValueError("Specify exactly one of period_s or n_periods")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    signal = _as_complex(data, tone=tone)
    n_samples = signal.shape[0]

    if period_s is not None:
        if period_s <= 0:
            raise ValueError(f"period_s must be positive, got {period_s}")
        sample_window = int(round(period_s * fs))
        n_periods = n_samples // sample_window if sample_window else 0
    else:
        if n_periods <= 0:
            raise ValueError(f"n_periods must be positive, got {n_periods}")
        sample_window = n_samples // n_periods

    if sample_window < 1:
        raise ValueError(
            f"One drive period is shorter than one sample at fs={fs} Hz; "
            "increase the sample rate or slow the drive."
        )
    if not n_periods:
        raise ValueError(
            f"The record holds {n_samples} samples, fewer than the {sample_window} samples "
            "in one drive period."
        )

    blocks = signal[: n_periods * sample_window].reshape(n_periods, sample_window)
    avg_iq = np.vstack((blocks.real.mean(axis=0), blocks.imag.mean(axis=0)))
    time_ms = np.arange(sample_window) / fs * 1e3

    return time_ms, avg_iq
