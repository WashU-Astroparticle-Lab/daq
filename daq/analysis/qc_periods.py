# -*- coding: utf-8 -*-
"""Per-period digitisation of a gate-ramped readout.

A sawtooth-biased device traces one quantum-capacitance (QC) curve per gate period. When the
island is poisoned the curve is suppressed, so the spread of the samples *within* one period
is a per-period detector of that state: large spread, normal QC; small spread, suppressed. This
module owns that reduction -- the principal axis the samples are projected on, the
per-period standard deviation, the two-mode threshold between the populations, and the
resulting state sequence -- so the notebooks that consume it agree on every definition.

It imports only numpy, so it is usable on an analysis machine with no ``presto`` install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

__all__ = ["principal_axis"]


def principal_axis(
    z: npt.ArrayLike,
) -> Tuple[npt.NDArray[np.floating], complex]:
    """Return the direction in the I/Q plane along which *z* spreads the most, and its mean.

    The largest-eigenvalue eigenvector of the *centred* I/Q covariance. The divisor of the
    covariance (``N`` or ``N - ddof``) scales both eigenvalues alike and so does not move the
    eigenvector, which is why no ``ddof`` appears here. An eigenvector's sign is arbitrary; it
    is fixed so the larger component is positive, purely so a saved axis is deterministic.
    When the two eigenvalues are equal the direction is ambiguous, but the spread along it is
    not.

    This is the same eigenvector ``qpd.reconstruction.estimate_direction`` returns (which adds
    a noise estimate from the minor axis); it lives here so the measurement layer can use it
    without ``qpd``. It is *not* ``qpd``'s two-blob discrimination axis: a ramped QC trace is
    a continuous curve, not two clusters, so its covariance is the right object.

    :param z: Complex samples, any shape; flattened.
    :raises ValueError: If the covariance is not finite or there are fewer than two samples.
    :returns: ``(axis, origin)`` -- the ``[I, Q]`` unit vector and the complex mean the
        samples were centred on. Project with
        ``axis[0] * (z - origin).real + axis[1] * (z - origin).imag``.

    """
    z = np.asarray(z, dtype=np.complex128).ravel()
    if z.size < 2:
        raise ValueError(f"principal_axis needs at least two samples, got {z.size}")
    origin = complex(z.mean())
    centered = z - origin
    iq = np.vstack((centered.real, centered.imag))
    covariance = (iq @ iq.T) / z.size
    if not np.all(np.isfinite(covariance)):
        raise ValueError("The I/Q covariance is not finite")
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return axis, origin


def project(
    z: npt.ArrayLike, axis: npt.ArrayLike, origin: complex = 0.0
) -> npt.NDArray[np.floating]:
    """Project complex samples onto an ``[I, Q]`` axis about *origin*.

    :param z: Complex samples, any shape.
    :param axis: ``[I, Q]`` unit vector, e.g. from :func:`principal_axis`.
    :param origin: Complex point the projection is measured from.
    :returns: Real array with the shape of *z*.

    """
    z = np.asarray(z, dtype=np.complex128) - origin
    axis = np.asarray(axis, dtype=np.float64)
    return axis[0] * z.real + axis[1] * z.imag


__all__.append("project")


# ---------------------------------------------------------------------- per-period std


@dataclass
class PeriodStd:
    """One standard deviation per complete gate period, per tone.

    Built by :func:`period_std`. Row ``k`` covers input samples
    ``sample_edges[k]:sample_edges[k + 1]`` -- half-open, whole periods only, the leading and
    trailing fragments dropped -- and ``time_edges_s`` gives the same boundaries in seconds from
    the first input sample.
    """

    std: npt.NDArray[np.floating]
    """Shape ``(n_periods, n_tones)``; ``nan`` where a period held a non-finite sample."""
    sample_edges: npt.NDArray[np.int64]
    """Shape ``(n_periods + 1,)``, indices into the input."""
    time_edges_s: npt.NDArray[np.floating]
    """Shape ``(n_periods + 1,)``, seconds from the first input sample."""
    counts: npt.NDArray[np.int64]
    """Samples in each period; alternates by one when ``fs / period`` is not an integer."""
    axis: npt.NDArray[np.floating]
    """Shape ``(n_tones, 2)``, the ``[I, Q]`` axis each tone was projected on."""
    origin: npt.NDArray[np.complexfloating]
    """Shape ``(n_tones,)``, the origin of each projection."""
    period_s: float
    fs: float
    ddof: int
    phase_fraction: float

    @property
    def centers_s(self) -> npt.NDArray[np.floating]:
        """Mid-time of each period, seconds from the first input sample."""
        return 0.5 * (self.time_edges_s[:-1] + self.time_edges_s[1:])

    @property
    def n_periods(self) -> int:
        return int(self.std.shape[0])


def period_std(
    z: npt.ArrayLike,
    fs: float,
    period_s: float,
    *,
    axis: Optional[npt.ArrayLike] = None,
    origin: Optional[npt.ArrayLike] = None,
    phase_fraction: float = 0.0,
    ddof: int = 1,
    chunk_periods: int = 5000,
) -> PeriodStd:
    """Standard deviation of the projected samples inside each complete gate period.

    The per-period detector of a suppressed QC response. Period boundaries are computed from
    **timestamps**, ``k * period_s`` (plus ``phase_fraction * period_s``), and rounded to
    samples individually, so a non-integral ``fs * period_s`` alternates the sample count
    between neighbouring periods instead of accumulating drift down the record -- the
    difference between this and a fixed block length is what makes a 5 kHz ramp sampled at a
    slightly detuned rate still bin correctly over 25 000 periods.

    No folding: each period's spread is its own number. The projection axis is the principal
    axis of the whole input per tone unless one is passed in (a KID tone at low LED current
    wants the axis from a brighter file, for instance).

    :param z: Complex samples, ``(n_samples,)`` or ``(n_samples, n_tones)``.
    :param fs: Sample rate in hertz -- the **tuned** ``TimeStream.df``.
    :param period_s: Gate period in seconds, ``1 / ramp_freq_hz``.
    :param axis: ``[I, Q]`` per tone, shape ``(n_tones, 2)`` (or ``(2,)`` for one tone).
        ``None`` fits :func:`principal_axis` per tone.
    :param origin: Complex origin per tone, used only with *axis*; defaults to each tone's
        mean. Irrelevant to the std (a constant offset), kept for a consistent projection.
    :param phase_fraction: Shift of the period origin as a fraction of a period, in
        ``[0, 1)``. The default puts a boundary at sample zero; the ramp's own phase is
        fixed by the gate trigger but not measured, so this is the knob to align on it.
    :param ddof: Delta degrees of freedom of the std; ``1`` (sample std) is the reference
        analysis's choice. At 20 samples per period ``0`` reads 2.6 % lower.
    :param chunk_periods: Periods processed per pass, bounding memory.
    :raises ValueError: If the inputs are malformed or no complete period fits.
    :returns: A :class:`PeriodStd`.

    """
    z = np.asarray(z, dtype=np.complex128)
    if z.ndim == 1:
        z = z[:, np.newaxis]
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("z must be (n_samples,) or (n_samples, n_tones) with n_samples >= 2")
    n_samples, n_tones = z.shape
    if not (np.isfinite(fs) and fs > 0 and np.isfinite(period_s) and period_s > 0):
        raise ValueError("fs and period_s must be positive and finite")
    if not 0.0 <= phase_fraction < 1.0:
        raise ValueError(f"phase_fraction must be in [0, 1), got {phase_fraction}")
    if isinstance(ddof, bool) or int(ddof) != ddof or ddof < 0:
        raise ValueError(f"ddof must be a non-negative integer, got {ddof!r}")
    ddof = int(ddof)
    if fs * period_s < ddof + 2:
        raise ValueError(
            f"{fs * period_s:.2f} samples per period cannot give a std with ddof={ddof}"
        )

    if axis is None:
        fitted = [principal_axis(z[:, tone]) for tone in range(n_tones)]
        axes = np.array([a for a, _ in fitted], dtype=np.float64)
        origins = np.array([o for _, o in fitted], dtype=np.complex128)
    else:
        axes = np.atleast_2d(np.asarray(axis, dtype=np.float64))
        if axes.shape != (n_tones, 2):
            raise ValueError(f"axis must have shape ({n_tones}, 2), got {axes.shape}")
        origins = (
            z.mean(axis=0)
            if origin is None
            else np.atleast_1d(np.asarray(origin, dtype=np.complex128))
        )
        if origins.shape != (n_tones,):
            raise ValueError(f"origin must have shape ({n_tones},), got {origins.shape}")

    # Timestamp boundaries: period k starts at (k + phase_fraction) * period_s.
    offset_s = phase_fraction * period_s
    first = int(np.ceil(-offset_s / period_s - 1e-9))
    first = max(first, 0)
    stop = int(np.floor((n_samples / fs - offset_s) / period_s + 1e-9))
    if stop <= first:
        raise ValueError("No complete gate period fits in the record")
    edges_s = offset_s + np.arange(first, stop + 1, dtype=np.float64) * period_s
    sample_edges = np.ceil(edges_s * fs - 1e-7).astype(np.int64)
    sample_edges = np.clip(sample_edges, 0, n_samples)
    # Drop any period the rounding left incomplete at the ends.
    while sample_edges.size > 1 and sample_edges[-1] > n_samples:
        sample_edges = sample_edges[:-1]
        edges_s = edges_s[:-1]
    counts = np.diff(sample_edges)
    if counts.size == 0 or np.any(counts <= ddof):
        raise ValueError("Every period must hold more than ddof samples")

    std = np.full((counts.size, n_tones), np.nan)
    for left in range(0, counts.size, chunk_periods):
        right = min(left + chunk_periods, counts.size)
        block = z[sample_edges[left] : sample_edges[right]]
        projected = (block.real - origins.real) * axes[:, 0] + (block.imag - origins.imag) * axes[
            :, 1
        ]
        starts = sample_edges[left:right] - sample_edges[left]
        block_counts = counts[left:right]
        means = np.add.reduceat(projected, starts, axis=0) / block_counts[:, np.newaxis]
        residual = projected - np.repeat(means, block_counts, axis=0)
        variance = np.add.reduceat(residual**2, starts, axis=0) / (
            block_counts[:, np.newaxis] - ddof
        )
        std[left:right] = np.sqrt(np.maximum(variance, 0.0))
    std[~np.isfinite(std)] = np.nan

    return PeriodStd(
        std=std,
        sample_edges=sample_edges,
        time_edges_s=sample_edges / fs,
        counts=counts,
        axis=axes,
        origin=origins,
        period_s=float(period_s),
        fs=float(fs),
        ddof=ddof,
        phase_fraction=float(phase_fraction),
    )


# ---------------------------------------------------------------------- the cut


@dataclass
class TwoModeCut:
    """The threshold between the two populations of per-period stds, with its diagnostics."""

    cut: float
    """Threshold in the std's units; ``nan`` when the two modes were not resolved."""
    resolved: bool
    modes: Tuple[float, float]
    """Locations of the two selected modes (``nan`` when unresolved)."""
    valley: float
    """Location of the smoothed histogram's minimum between the modes (``nan`` if none)."""
    reason: str
    """Why the cut is what it is -- ``"ok"`` or what failed."""
    hist_counts: npt.NDArray[np.floating]
    hist_edges: npt.NDArray[np.floating]
    smoothed: npt.NDArray[np.floating]
    n_values: int


def two_mode_cut(
    values: npt.ArrayLike,
    *,
    bins: int = 100,
    smooth_sigma: float = 1.5,
    min_prominence: float = 0.05,
    min_valley_dip: float = 0.15,
    min_population: float = 0.005,
    min_values: int = 200,
    hist_range: Optional[Tuple[float, float]] = None,
) -> TwoModeCut:
    """Place a threshold at the midpoint of two resolved modes in a smoothed histogram.

    The reference analysis's heuristic, with its defaults: histogram the finite values,
    smooth with a Gaussian of *smooth_sigma* bins, find peaks whose prominence is at least
    *min_prominence* of the tallest, take the two most prominent, require the smoothed
    minimum between them to dip by at least *min_valley_dip* of the lower peak, and require
    at least *min_population* of the values on each side of the midpoint. Any failure leaves
    ``cut = nan`` and ``resolved = False`` with the *reason* -- **never** a forced threshold,
    since a device whose histogram is unimodal has no state to assign. It is a heuristic,
    not a physical state separator; inspect the histogram it returns.

    :param values: Per-period stds of one device, any shape; non-finite values are ignored.
    :param bins: Histogram bins.
    :param smooth_sigma: Gaussian smoothing width in histogram bins.
    :param min_prominence: Peak prominence floor, as a fraction of the tallest smoothed bin.
    :param min_valley_dip: Required drop of the valley below the lower of the two modes.
    :param min_population: Minimum fraction of values on either side of the cut.
    :param min_values: Minimum number of finite values before a cut is attempted.
    :param hist_range: Histogram range; defaults to the data's.
    :returns: A :class:`TwoModeCut`.

    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    counts, edges = (
        np.histogram(x, bins=bins, range=hist_range)
        if x.size
        else (
            np.zeros(bins),
            np.linspace(0.0, 1.0, bins + 1),
        )
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = (
        gaussian_filter1d(counts.astype(np.float64), smooth_sigma)
        if x.size
        else counts.astype(float)
    )
    nan2 = (np.nan, np.nan)

    def fail(reason: str) -> TwoModeCut:
        return TwoModeCut(np.nan, False, nan2, np.nan, reason, counts, edges, smoothed, int(x.size))

    if x.size < min_values:
        return fail(f"only {x.size} finite values (< {min_values})")
    peaks, props = find_peaks(smoothed, prominence=min_prominence * smoothed.max())
    if peaks.size < 2:
        return fail(f"{peaks.size} resolved mode(s), need 2")
    top = peaks[np.argsort(props["prominences"])[-2:]]
    lo, hi = int(top.min()), int(top.max())
    between = smoothed[lo : hi + 1]
    v = lo + int(np.argmin(between))
    lower_peak = min(smoothed[lo], smoothed[hi])
    if smoothed[v] > (1.0 - min_valley_dip) * lower_peak:
        return fail("no valley between the modes")
    cut = 0.5 * (centers[lo] + centers[hi])
    below = np.mean(x < cut)
    if below < min_population or 1.0 - below < min_population:
        return fail(
            f"population {below:.3g} below the cut is outside [{min_population}, "
            f"{1 - min_population}]"
        )
    return TwoModeCut(
        float(cut),
        True,
        (float(centers[lo]), float(centers[hi])),
        float(centers[v]),
        "ok",
        counts,
        edges,
        smoothed,
        int(x.size),
    )


# ---------------------------------------------------------------------- states


def classify_periods(std: npt.ArrayLike, cut: npt.ArrayLike) -> npt.NDArray[np.int8]:
    """Assign each period a state from its std and the device's cut.

    ``1`` where ``std < cut`` (suppressed QC -- the event), ``0`` where ``std >= cut``, ``-1``
    where the std is not finite or the cut is ``nan`` (unresolved device). Unknown is never
    silently ``0``.

    :param std: ``(n_periods,)`` or ``(n_periods, n_tones)``.
    :param cut: A scalar, or one cut per tone.
    :returns: ``int8`` array with the shape of *std*.

    """
    s = np.asarray(std, dtype=np.float64)
    c = np.asarray(cut, dtype=np.float64)
    if s.ndim == 2 and c.ndim == 1 and c.shape != (s.shape[1],):
        raise ValueError(f"cut must be scalar or have one entry per tone ({s.shape[1]})")
    c = np.broadcast_to(c, s.shape)
    states = np.full(s.shape, -1, dtype=np.int8)
    known = np.isfinite(s) & np.isfinite(c)
    states[known & (s < c)] = 1
    states[known & (s >= c)] = 0
    return states


def state_onsets(
    states: npt.ArrayLike, time_edges_s: Optional[npt.ArrayLike] = None
) -> List[npt.NDArray[np.floating]]:
    """Return the start of every run of suppressed periods, per tone.

    A ``0 -> 1`` transition; runs that begin at the record's first period, or right after an
    unknown period, are excluded since their start is not observed. With *time_edges_s* the
    start times of those periods are returned, otherwise their indices.

    :param states: ``(n_periods,)`` or ``(n_periods, n_tones)`` from :func:`classify_periods`.
    :param time_edges_s: ``(n_periods + 1,)`` period boundaries, e.g. ``PeriodStd.time_edges_s``.
    :returns: One array per tone.

    """
    s = np.asarray(states)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    out = []
    for tone in range(s.shape[1]):
        column = s[:, tone]
        onsets = np.flatnonzero((column[1:] == 1) & (column[:-1] == 0)) + 1
        if time_edges_s is not None:
            onsets = np.asarray(time_edges_s, dtype=np.float64)[onsets]
        out.append(onsets)
    return out


__all__ += [
    "PeriodStd",
    "period_std",
    "TwoModeCut",
    "two_mode_cut",
    "classify_periods",
    "state_onsets",
]
