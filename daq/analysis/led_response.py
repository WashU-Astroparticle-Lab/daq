# -*- coding: utf-8 -*-
"""Response of a gate-ramped device array to a periodic train of LED flashes.

The flashes are software-started (:class:`~daq.measurements.led_pulsed_ramp.LEDPulsedRamp`),
so their phase in each record is unknown until read off the data. This module owns that step
and what follows from it: locate the flash comb on a marker tone (a KID), average the marker's
pulse shape, and fold events from the other tones onto the comb to get a rate against time
since the flash. Every function takes plain arrays, so it applies to one file or to a night of
them; nothing here reads a manifest.

Imports only numpy (and :mod:`daq.analysis.folding`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "Comb",
    "find_pulse_comb",
    "average_pulse",
    "fold_events",
    "folded_event_rate",
    "folded_fraction",
]


@dataclass
class Comb:
    """A periodic flash train located in one record."""

    period_s: float
    phase_s: float
    """Time of the first flash **onset** at or after the record start, in seconds."""
    peak_s: float
    """Time of the folded profile's extremum within the period, in seconds."""
    times: npt.NDArray[np.floating]
    """Flash onsets inside the record: ``phase_s + k * period_s``."""
    profile_t: npt.NDArray[np.floating]
    """Time within one period, seconds, for *profile*."""
    profile: npt.NDArray[np.floating]
    """Folded (mean over flashes) marker trace, baseline (median) removed."""
    peak: float
    """Folded profile extremum, signed."""
    noise: float
    """Per-sample noise of the marker, from first differences."""
    n_flashes: int
    snr_folded: float
    snr_single: float

    @property
    def detected(self) -> bool:
        """Whether the folded profile stands out; see :func:`find_pulse_comb`."""
        return bool(self.snr_folded >= 5.0)


def _fold_mean(
    x: npt.NDArray[np.floating], fs: float, period_s: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    per = int(round(period_s * fs))
    n = x.size // per
    if n < 1:
        raise ValueError("The record is shorter than one period")
    folded = x[: n * per].reshape(n, per).mean(axis=0)
    return np.arange(per) / fs, folded, n


def find_pulse_comb(
    x: npt.ArrayLike,
    fs: float,
    period_s: float,
    *,
    onset_fraction: float = 0.5,
    record_start_s: float = 0.0,
    remove_period_s: Optional[float] = None,
) -> Comb:
    """Locate a periodic pulse train of known period in a marker trace.

    Folds *x* at *period_s* (one mean profile over all complete periods), takes the extremum
    of the baseline-subtracted profile as the pulse, and walks back from it to where the
    profile first exceeds *onset_fraction* of the extremum -- the flash **onset**, which is
    what the other tones' events are timed against. The single-flash SNR compares the
    extremum to the per-sample noise (first differences, insensitive to the slow gate
    structure); the folded SNR is that times ``sqrt(n_flashes)``, and the reference for
    :attr:`Comb.detected`.

    The period is taken as given (the DC2200's own timebase; verify it once with
    ``qpd.reconstruction.estimate_fold_period`` if in doubt). Only the phase is measured,
    which is all the fold needs -- and it is measured on the folded profile, so a marker whose
    single flashes are below the noise (SNR ~3 at 198 mA on this device) still yields it.

    :param x: Real marker trace (a projection, ``dtheta``, ...), 1-D.
    :param fs: Sample rate in hertz (the tuned ``df``).
    :param period_s: Flash period in seconds.
    :param onset_fraction: Fraction of the extremum defining the onset on the leading edge.
    :param record_start_s: Time of ``x[0]`` in the record's own axis (e.g. the discarded
        start), added to every returned time.
    :param remove_period_s: A periodicity to subtract before folding -- the gate period,
        ``1 / ramp_freq_hz``. A marker read out beside a ramped array carries the ramp's
        pickup, and when the flash period is a whole number of gate periods that pickup folds
        coherently at *every* multiple of the gate period, so the finder locks onto it
        wherever the flash is weaker than the ripple. The mean fold at this period is tiled
        and subtracted; the flash, incommensurate with the gate at the sample level only
        through its jitter, survives while the ripple does not.
    :returns: A :class:`Comb`.

    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite 1-D array")
    if not 0.0 < onset_fraction <= 1.0:
        raise ValueError("onset_fraction must be in (0, 1]")
    if remove_period_s is not None:
        if not (np.isfinite(remove_period_s) and 0 < remove_period_s < period_s):
            raise ValueError("remove_period_s must be positive and shorter than period_s")
        _, ripple, n_gate = _fold_mean(x, fs, remove_period_s)
        ripple -= ripple.mean()
        x = x - np.tile(ripple, n_gate + 1)[: x.size]
    t, folded, n = _fold_mean(x, fs, period_s)
    profile = folded - np.median(folded)
    k = int(np.argmax(np.abs(profile)))
    peak = float(profile[k])
    noise = float(np.std(np.diff(x)) / np.sqrt(2.0))
    # Leading edge: walk back from the extremum while the profile stays above the fraction.
    threshold = onset_fraction * abs(peak)
    j = k
    while j > 0 and abs(profile[j - 1]) >= threshold and np.sign(profile[j - 1]) == np.sign(peak):
        j -= 1
    # If the pulse wraps around the period boundary the walk stops at 0; accept that.
    phase = float(t[j]) + record_start_s
    n_total = int(np.floor((x.size / fs - t[j]) / period_s)) + (1 if t[j] < x.size / fs else 0)
    times = phase + period_s * np.arange(max(n_total, 0))
    times = times[times < record_start_s + x.size / fs]
    snr_single = abs(peak) / noise if noise > 0 else np.inf
    return Comb(
        period_s=float(period_s),
        phase_s=phase,
        peak_s=float(t[k]) + record_start_s,
        times=times,
        profile_t=t,
        profile=profile,
        peak=peak,
        noise=noise,
        n_flashes=n,
        snr_folded=float(snr_single * np.sqrt(n)),
        snr_single=float(snr_single),
    )


def average_pulse(
    x: npt.ArrayLike,
    fs: float,
    times: npt.ArrayLike,
    *,
    pre_s: float,
    post_s: float,
    record_start_s: float = 0.0,
    subtract_baseline: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stack windows of *x* around each flash time and average them.

    :param x: Real trace, 1-D (``dtheta`` or ``dx`` for a KID).
    :param fs: Sample rate in hertz.
    :param times: Flash onsets in the record's axis, e.g. ``Comb.times``.
    :param pre_s: Window before each onset, seconds.
    :param post_s: Window after each onset, seconds.
    :param record_start_s: Time of ``x[0]`` in the record's axis.
    :param subtract_baseline: Remove each window's pre-onset mean before stacking.
    :returns: ``(t, mean, std, n)`` -- time from onset, the mean and per-sample std across the
        ``n`` windows that fit inside the record.

    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n_pre = int(round(pre_s * fs))
    n_post = int(round(post_s * fs))
    if n_pre < 0 or n_post <= 0:
        raise ValueError("pre_s must be >= 0 and post_s > 0")
    windows = []
    for t0 in np.asarray(times, dtype=np.float64).ravel():
        i = int(round((t0 - record_start_s) * fs))
        if i - n_pre < 0 or i + n_post > x.size:
            continue
        w = x[i - n_pre : i + n_post].copy()
        if subtract_baseline and n_pre > 0:
            w -= w[:n_pre].mean()
        windows.append(w)
    if not windows:
        raise ValueError("No flash window fits inside the record")
    stack = np.array(windows)
    t = (np.arange(-n_pre, n_post)) / fs
    return (
        t,
        stack.mean(axis=0),
        stack.std(axis=0, ddof=1) if len(windows) > 1 else np.zeros(t.size),
        len(windows),
    )


def fold_events(
    event_times: npt.ArrayLike,
    phase_s: float,
    period_s: float,
    record_s: float,
) -> Tuple[np.ndarray, int]:
    """Fold event times onto the flash period, keeping complete cycles only.

    Events before the first flash are dropped, and so is the incomplete cycle at the end of
    the record -- keeping it would add counts to the early phase bins without a cycle in the
    denominator, which reads as an excess exactly where the signal is expected.

    :param event_times: Event times in the record's axis (e.g. suppressed-period centres).
    :param phase_s: First flash onset, ``Comb.phase_s``.
    :param period_s: Flash period.
    :param record_s: Record length in the same axis (end time).
    :returns: ``(folded_times, n_cycles)`` -- times since the flash in ``[0, period_s)``, and
        the number of complete cycles they were drawn from.

    """
    t = np.asarray(event_times, dtype=np.float64).ravel()
    n_cycles = int(np.floor((record_s - phase_s) / period_s))
    if n_cycles < 1:
        return np.empty(0), 0
    keep = (t >= phase_s) & (t < phase_s + n_cycles * period_s)
    return (t[keep] - phase_s) % period_s, n_cycles


def folded_event_rate(
    folded_times: npt.ArrayLike,
    n_cycles: int,
    period_s: float,
    *,
    bins: int = 250,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Event rate against time since the flash, with Poisson errors.

    Pool several records by concatenating their folded times and summing their cycles.

    :param folded_times: From :func:`fold_events` (one or many records concatenated).
    :param n_cycles: Total complete cycles behind those times.
    :param period_s: Flash period.
    :param bins: Bins across one period -- 250 gives one gate period per bin at 5 kHz / 50 ms.
    :returns: ``(rate_hz, err_hz, edges_s)``.

    """
    counts, edges = np.histogram(
        np.asarray(folded_times, dtype=np.float64), bins=bins, range=(0.0, period_s)
    )
    if n_cycles < 1:
        raise ValueError("n_cycles must be at least 1")
    denom = n_cycles * np.diff(edges)
    return counts / denom, np.sqrt(counts) / denom, edges


def folded_fraction(
    folded_times: npt.ArrayLike,
    n_cycles: int,
    period_s: float,
    gate_period_s: float,
    *,
    bins: int = 250,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fraction of gate periods in a given state against time since the flash.

    For *folded_times* the centres of every suppressed gate period, this is the suppressed
    fraction per phase bin: counts divided by the gate periods each bin spans over all cycles.

    :param folded_times: From :func:`fold_events`.
    :param n_cycles: Total complete cycles.
    :param period_s: Flash period.
    :param gate_period_s: Gate period, ``1 / ramp_freq_hz``.
    :param bins: Bins across one flash period.
    :returns: ``(fraction, err, edges_s)`` with binomial errors.

    """
    counts, edges = np.histogram(
        np.asarray(folded_times, dtype=np.float64), bins=bins, range=(0.0, period_s)
    )
    periods_per_bin = n_cycles * np.diff(edges) / gate_period_s
    frac = counts / periods_per_bin
    err = np.sqrt(np.clip(frac * (1 - frac), 0, None) / periods_per_bin)
    return frac, err, edges
