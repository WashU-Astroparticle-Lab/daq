# -*- coding: utf-8 -*-
"""Time-domain reconstruction of a random-telegraph (parity) time stream.

The frequency-domain half of this analysis lives in :mod:`daq.analysis.noise` --
:func:`~daq.analysis.noise.fit_parity_psd` fits the switching rate out of the spectrum. This
module is its time-domain counterpart: it assigns each sample to one of the two levels, so the
switching events themselves are recoverable rather than only their aggregate rate.

The two are worth having together because they fail differently. The spectral fit is blind to
*when* things happened -- a record whose switching rate doubles halfway through fits a single
Lorentzian at some intermediate corner and gives no sign of it -- while the time-domain
reconstruction gives an independent rate estimate (flips per second) and localises the
episodes. :func:`detect_bursts` is what that localisation is for: a run of rapid switching
against an otherwise quiet record, which for a quasiparticle-sensitive device is an impact
event rather than a property of the operating point.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

__all__ = ["detect_bursts", "reconstruct_telegraph"]

#: Level separation, in units of the within-level noise, below which a record is not treated as
#: two-level at all. Under this the "flips" a threshold produces are noise crossings, not
#: switching events, and reporting them as a reconstruction would be worse than reporting
#: nothing.
#:
#: **This metric does not go to zero on structureless data**, which is why the threshold is
#: where it is rather than near 1. Splitting a unimodal Gaussian at its own mean leaves halves
#: whose means are ``sqrt(2/pi) = 0.80`` sigma either side and whose within-half spread is
#: ``sqrt(1 - 2/pi) = 0.60`` sigma, so 2-means on pure noise reports a "separation" of about
#: 2.6 noise widths. Measured over eight realisations of 2e5 Gaussian samples: 2.35-2.37,
#: tight enough to treat as a floor. A genuine telegraph at ``separation / noise = 2`` scores
#: 2.67 -- barely above that floor -- and its flip count over-reads the true rate by 27x, while
#: at 4.0 it scores 4.05 and the rate is 55 % high but the right order. 3.5 sits above the
#: floor and above the useless case, below the marginal-but-usable one.
MIN_SEPARATION_SNR = 3.5


def _two_levels(series: npt.NDArray[np.floating]) -> tuple:
    """Find the two levels of a telegraph series by 1-D 2-means.

    Initialised at the 10th and 90th percentiles, which is robust to the tails a threshold
    would otherwise be dragged by, and deterministic -- the same record always gives the same
    levels.

    :param series: The real-valued series.
    :returns: ``(low, high)`` level estimates.

    """
    low, high = np.percentile(series, [10.0, 90.0])
    for _ in range(50):
        mid = 0.5 * (low + high)
        below = series <= mid
        if not below.any() or below.all():
            break
        new_low = float(series[below].mean())
        new_high = float(series[~below].mean())
        if np.isclose(new_low, low) and np.isclose(new_high, high):
            low, high = new_low, new_high
            break
        low, high = new_low, new_high
    return float(low), float(high)


def _schmitt(
    series: npt.NDArray[np.floating], low_threshold: float, high_threshold: float
) -> npt.NDArray[np.bool_]:
    """Assign a two-state sequence with hysteresis, vectorised.

    A single threshold makes a sample sitting on it flip on every noise excursion, which shows
    up as a burst of spurious switching exactly where the signal is least informative. Two
    thresholds mean a state is held until the series crosses the *other* one.

    :param series: The real-valued series.
    :param low_threshold: Cross below this to enter the low state.
    :param high_threshold: Cross above this to enter the high state.
    :returns: Boolean array, ``True`` in the high state.

    """
    decided = np.zeros(series.shape[0], dtype=np.int8)
    decided[series > high_threshold] = 1
    decided[series < low_threshold] = -1

    nonzero = decided != 0
    if not nonzero.any():
        # Everything sits between the thresholds: no crossing was ever unambiguous.
        return np.zeros(series.shape[0], dtype=bool)

    # Forward-fill the last decided sample: between the thresholds the state is held.
    indices = np.maximum.accumulate(np.where(nonzero, np.arange(series.shape[0]), 0))
    state = decided[indices] > 0
    # Before the first decided sample there is nothing to hold, so adopt the first decision.
    first = int(np.argmax(nonzero))
    state[:first] = decided[first] > 0
    return state


def _drop_short_dwells(state: npt.NDArray[np.bool_], min_samples: int) -> npt.NDArray[np.bool_]:
    """Absorb dwells shorter than *min_samples* into the preceding one.

    :param state: The two-state sequence.
    :param min_samples: Shortest dwell to keep, in samples.
    :returns: The filtered sequence.

    """
    if min_samples <= 1:
        return state
    state = state.copy()
    for _ in range(10):
        edges = np.flatnonzero(np.diff(state)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [state.shape[0]]))
        short = np.flatnonzero((ends - starts) < min_samples)
        # The first segment has no predecessor to be absorbed into; leave it.
        short = short[short > 0]
        if short.size == 0:
            break
        for index in short:
            state[starts[index] : ends[index]] = state[starts[index] - 1]
    return state


def reconstruct_telegraph(
    series: npt.ArrayLike,
    fs: float,
    *,
    hysteresis: float = 0.25,
    min_dwell_s: Optional[float] = None,
    levels: Optional[tuple] = None,
    min_snr: float = MIN_SEPARATION_SNR,
) -> Dict[str, Any]:
    """Reconstruct the two-level switching sequence of a parity time stream.

    The levels are found by 2-means, the samples are assigned with a Schmitt trigger at
    ``mid +/- hysteresis * separation``, and the switching events are the transitions of that
    assignment. The flip count gives a rate estimate independent of the spectral fit: for a
    symmetric telegraph the mean dwell time is ``1 / Gamma_p``, so ``Gamma_p = n_flips /
    duration``. Comparing it against
    :func:`~daq.analysis.noise.fit_parity_psd`'s ``gamma_p`` is the cheapest check there is
    that either number means anything.

    **A record with no two-level structure is reported as such, not thresholded anyway.**
    ``separated`` is ``False`` when the levels are closer than *min_snr* noise widths, which is
    the case for a stream taken off the parity-sensitive operating point. The flips are still
    returned -- they are just noise crossings, and their count says so by being enormous. Note
    that ``snr`` bottoms out near 2.4 rather than 0 on structureless data, for the reason given
    at :data:`MIN_SEPARATION_SNR`; read it against that floor, not against zero.

    :param series: Real-valued projection of the readout, e.g.
        ``TimeStream._projection`` output. Complex input is rejected: which projection to take
        is a decision this function should not make silently.
    :param fs: Sample rate in hertz -- the *tuned* ``TimeStream.df``.
    :param hysteresis: Threshold offset from the midpoint, as a fraction of the level
        separation. ``0`` gives a single threshold at the midpoint (and the chatter that comes
        with it); the default holds a state until the series has crossed three quarters of the
        way to the other level.
    :param min_dwell_s: Absorb dwells shorter than this into the preceding one. ``None``
        (default) keeps every dwell the hysteresis allows -- set it only when you know the
        physical switching cannot be faster than some scale, since it biases the rate down.
    :param levels: Explicit ``(low, high)`` levels, skipping the 2-means step.
    :param min_snr: Level separation, in within-level noise widths, below which ``separated``
        is ``False``.
    :raises TypeError: If *series* is complex.
    :raises ValueError: If *series* is not 1-D with at least two samples, or *fs* is not
        positive.
    :returns: A dict carrying ``state`` (bool array, ``True`` in the high level),
        ``flip_indices`` / ``flip_times_s``, ``n_flips``, ``gamma_p_flips`` (Hz), ``levels``,
        ``separation``, ``noise`` (within-level std), ``snr``, ``separated``,
        ``low_threshold`` / ``high_threshold``, ``mean_dwell_s`` and ``duration_s``.

    """
    series = np.asarray(series)
    if np.iscomplexobj(series):
        raise TypeError(
            "reconstruct_telegraph needs a real projection of the readout, not complex I/Q; "
            "project it first (e.g. np.abs(signal[:, tone]))."
        )
    series = series.astype(np.float64, copy=False)
    if series.ndim != 1 or series.shape[0] < 2:
        raise ValueError(f"series must be 1-D with at least 2 samples, got shape {series.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    low, high = _two_levels(series) if levels is None else (float(levels[0]), float(levels[1]))
    if high < low:
        low, high = high, low
    separation = high - low
    midpoint = 0.5 * (low + high)

    if separation <= 0:
        state = np.zeros(series.shape[0], dtype=bool)
        low_threshold = high_threshold = midpoint
    else:
        low_threshold = midpoint - hysteresis * separation
        high_threshold = midpoint + hysteresis * separation
        state = _schmitt(series, low_threshold, high_threshold)
        if min_dwell_s is not None:
            state = _drop_short_dwells(state, int(round(min_dwell_s * fs)))

    # Noise about the assigned levels, which is what the separation has to beat. Measured
    # against the *assignment* rather than the whole record's std, since the latter is
    # dominated by the switching itself -- the quantity being tested against.
    residual = series.copy()
    if state.any():
        residual[state] -= series[state].mean()
    if (~state).any():
        residual[~state] -= series[~state].mean()
    noise = float(np.std(residual))
    snr = float(separation / noise) if noise > 0 else float("inf")

    flip_indices = np.flatnonzero(np.diff(state)) + 1
    duration_s = series.shape[0] / fs
    n_flips = int(flip_indices.size)

    return {
        "state": state,
        "flip_indices": flip_indices,
        "flip_times_s": flip_indices / fs,
        "n_flips": n_flips,
        # Symmetric telegraph: mean dwell = 1 / Gamma_p, so the flip rate *is* Gamma_p.
        "gamma_p_flips": n_flips / duration_s if duration_s > 0 else float("nan"),
        "mean_dwell_s": duration_s / n_flips if n_flips else float("nan"),
        "levels": (low, high),
        "separation": separation,
        "noise": noise,
        "snr": snr,
        "separated": bool(snr >= min_snr and separation > 0),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "duration_s": duration_s,
        "fs": float(fs),
    }


def _poisson_sf(k: int, mu: float) -> float:
    """Return ``P(N >= k)`` for a Poisson variable of mean *mu*.

    Summed directly for small means and normal-approximated above 50, where the direct sum
    starts at ``exp(-mu)`` and underflows.

    :param k: Count threshold.
    :param mu: Poisson mean.
    :returns: The upper-tail probability.

    """
    if k <= 0:
        return 1.0
    if mu <= 0:
        return 0.0
    if mu > 50:
        # Continuity-corrected normal tail; accurate to well within the decade that matters
        # for a detection threshold.
        z = (k - 0.5 - mu) / math.sqrt(mu)
        return 0.5 * math.erfc(z / math.sqrt(2.0))
    term = math.exp(-mu)
    cdf = term
    for i in range(1, k):
        term *= mu / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def detect_bursts(
    flip_times_s: npt.ArrayLike,
    duration_s: float,
    *,
    window_s: Optional[float] = None,
    expected_per_window: float = 5.0,
    p_value: float = 1e-3,
    min_flips: int = 20,
) -> List[Dict[str, Any]]:
    """Find runs of anomalously rapid switching in a reconstructed flip sequence.

    A quasiparticle burst -- an impact event depositing energy in the device -- raises the
    parity-switching rate for a short while and then decays. Against the record's own mean rate
    that is a local excess of flips, so this slides a window over the flip times and keeps the
    windows whose count is too high to be a fluctuation of a Poisson process at the mean rate.

    **The threshold is Bonferroni-corrected over the number of windows tested**, which is what
    keeps a long quiet record from producing "bursts" by sheer multiplicity: at ``p = 1e-3``
    per window, a thousand windows would otherwise be expected to yield one. The correction is
    why a quiet record returns an empty list rather than a plausible-looking span.

    Two caveats worth stating, since neither is visible in the output. The null model is a
    *homogeneous* Poisson process, so a rate that drifts slowly across the record (a
    temperature ramp, say) reads as a burst. And the window sets the timescale this is
    sensitive to: an episode much shorter than one window is diluted within it.

    :param flip_times_s: Switching times in seconds, as
        :func:`reconstruct_telegraph` returns them.
    :param duration_s: Length of the record in seconds.
    :param window_s: Window length. ``None`` (default) picks the length holding
        *expected_per_window* flips at the mean rate, so the test scales with the record
        instead of assuming a timescale.
    :param expected_per_window: Mean flips per window used to size the default window.
    :param p_value: Per-record false-positive rate, before the Bonferroni split across
        windows.
    :param min_flips: Records with fewer flips than this are not tested at all -- the mean rate
        is too poorly determined to call anything anomalous against it.
    :raises ValueError: If *duration_s* or *window_s* is not positive.
    :returns: One dict per burst, in time order, with ``start_s``, ``end_s``, ``n_flips`` and
        ``rate_hz``; empty when nothing is significant.

    """
    flip_times = np.sort(np.asarray(flip_times_s, dtype=np.float64).ravel())
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if window_s is not None and window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if flip_times.size < min_flips:
        return []

    rate = flip_times.size / duration_s
    if window_s is None:
        window_s = expected_per_window / rate
    window_s = float(min(window_s, duration_s / 4.0))
    if window_s <= 0:
        return []

    # Half-window steps: an episode straddling a bin boundary is still caught whole by one of
    # the two windows covering it.
    step = window_s / 2.0
    starts = np.arange(0.0, max(duration_s - window_s, 0.0) + step, step)
    if starts.size == 0:
        return []
    counts = np.searchsorted(flip_times, starts + window_s) - np.searchsorted(flip_times, starts)

    mu = rate * window_s
    alpha = p_value / starts.size
    threshold = max(int(math.ceil(mu)), 1)
    while threshold < flip_times.size and _poisson_sf(threshold, mu) >= alpha:
        threshold += 1

    flagged = counts >= threshold
    if not flagged.any():
        return []

    bursts: List[Dict[str, Any]] = []
    start_index = None
    for index, is_flagged in enumerate(np.append(flagged, False)):
        if is_flagged and start_index is None:
            start_index = index
        elif not is_flagged and start_index is not None:
            begin = float(starts[start_index])
            end = float(min(starts[index - 1] + window_s, duration_s))
            inside = int(np.searchsorted(flip_times, end) - np.searchsorted(flip_times, begin))
            bursts.append({
                "start_s": begin,
                "end_s": end,
                "n_flips": inside,
                "rate_hz": inside / (end - begin) if end > begin else float("nan"),
            })
            start_index = None
    return bursts
