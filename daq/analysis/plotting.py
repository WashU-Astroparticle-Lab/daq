# -*- coding: utf-8 -*-
"""Plotting helpers for noise / resonator analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import matplotlib.axes
    from daq.measurements.sweep import Sweep

Basis = Literal["electronic", "fractional", "resonator"]
Density = Literal["scatter", "kde", "contour", "hexbin", "hist2d"]


def _to_basis(
    z: npt.NDArray[np.complexfloating],
    env: npt.NDArray[np.complexfloating] | complex,
    phi0: float,
    basis: Basis,
) -> npt.NDArray[np.complexfloating]:
    """Project a complex response into the requested display basis.

    :param z: Complex response to transform (sweep trace, time-stream, or a
        single calibration point).
    :param env: Environmental term evaluated at the same frequencies as *z* --
        the full array for the sweep trace, or the scalar value at ``fr`` for a
        time-stream taken on resonance.
    :param phi0: Impedance-mismatch rotation angle from the resonator fit.
    :param basis: One of ``"electronic"`` (raw I/Q), ``"fractional"``
        (environment removed), or ``"resonator"`` (recentred on the resonance
        circle).
    :returns: The transformed complex response.
    """
    if basis == "electronic":
        return z
    if basis == "fractional":
        return z / env
    if basis == "resonator":
        return (z / env - 1) / np.exp(1j * phi0) + 1
    raise ValueError(f"basis must be one of 'electronic', 'fractional', 'resonator'; got {basis!r}")


def _add_kde_contours(
    ax: "matplotlib.axes.Axes",
    real: npt.NDArray[np.floating],
    imag: npt.NDArray[np.floating],
    color: str,
    max_points: int,
    grid_size: int,
) -> None:
    """Overlay 1-sigma / 2-sigma Gaussian-KDE density contours on *ax*."""
    from scipy.stats import gaussian_kde

    if real.size < 3:
        return

    rng = np.random.default_rng()

    # Subsample for the (expensive) KDE fit.
    if real.size > max_points:
        idx = rng.choice(real.size, max_points, replace=False)
        real_sub, imag_sub = real[idx], imag[idx]
    else:
        real_sub, imag_sub = real, imag

    kde = gaussian_kde(np.vstack([real_sub, imag_sub]))

    # Evaluate the density on a padded grid spanning the data.
    def _padded(lo: float, hi: float) -> Tuple[float, float]:
        pad = 0.1 * (hi - lo)
        return lo - pad, hi + pad

    x_min, x_max = _padded(real.min(), real.max())
    y_min, y_max = _padded(imag.min(), imag.max())
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Convert confidence fractions into density levels via the sampled density.
    sample = np.vstack([real_sub, imag_sub])
    if sample.shape[1] > 1000:
        sample = sample[:, rng.choice(sample.shape[1], 1000, replace=False)]
    sampled_density = kde(sample)
    level_1sigma = np.percentile(sampled_density, 100 - 68.3)
    level_2sigma = np.percentile(sampled_density, 100 - 95.4)

    ax.contour(
        xx,
        yy,
        density,
        levels=[level_2sigma],
        colors=[color],
        alpha=0.3,
        linewidths=1.0,
        linestyles=":",
    )
    ax.contour(
        xx,
        yy,
        density,
        levels=[level_1sigma],
        colors=[color],
        alpha=0.6,
        linewidths=1.5,
        linestyles="--",
    )


def _add_hist_contours(
    ax: "matplotlib.axes.Axes",
    real: npt.NDArray[np.floating],
    imag: npt.NDArray[np.floating],
    color: str,
    bins: int,
    smooth: float = 1.0,
) -> None:
    """Overlay 1-sigma / 2-sigma contours from a 2-D histogram on *ax*.

    Much faster than :func:`_add_kde_contours` for large clouds -- it bins with
    :func:`numpy.histogram2d` (no per-point kernel evaluation) and derives the
    contour levels from the enclosed cumulative mass, so the 1-sigma / 2-sigma
    lines bound the innermost 68.3% / 95.4% of the counts rather than fixed
    count thresholds.

    :param ax: Axis to draw on.
    :param real: Real (I) component of the cloud.
    :param imag: Imaginary (Q) component of the cloud.
    :param color: Contour colour.
    :param bins: Number of histogram bins per axis.
    :param smooth: Gaussian smoothing (in bins) applied to the histogram before
        contouring, to tame jagged lines. Set to ``0`` to disable. Defaults to
        ``1.0``.
    """
    if real.size < 3:
        return

    counts, x_edges, y_edges = np.histogram2d(real, imag, bins=bins)
    if smooth > 0:
        from scipy.ndimage import gaussian_filter

        counts = gaussian_filter(counts, smooth)

    total = counts.sum()
    if total <= 0:
        return

    # Density levels enclosing 68.3% / 95.4% of the mass (Z-values, decreasing).
    flat = np.sort(counts.ravel())[::-1]
    cum = np.cumsum(flat) / total
    level_1sigma = flat[min(int(np.searchsorted(cum, 0.683)), flat.size - 1)]
    level_2sigma = flat[min(int(np.searchsorted(cum, 0.954)), flat.size - 1)]

    x_ctr = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_ctr = 0.5 * (y_edges[:-1] + y_edges[1:])

    # counts is indexed [x, y]; contour wants Z[y, x], hence the transpose.
    ax.contour(
        x_ctr,
        y_ctr,
        counts.T,
        levels=[level_2sigma],
        colors=[color],
        alpha=0.3,
        linewidths=1.0,
        linestyles=":",
    )
    ax.contour(
        x_ctr,
        y_ctr,
        counts.T,
        levels=[level_1sigma],
        colors=[color],
        alpha=0.6,
        linewidths=1.5,
        linestyles="--",
    )


def plot_iq_comparison(
    ts: npt.NDArray[np.complexfloating],
    sw: "Sweep",
    qc: Optional[npt.NDArray[np.complexfloating]] = None,
    *,
    basis: Basis = "electronic",
    freq_shift: float = 400e3,
    density: Density = "scatter",
    fcrop: Optional[Tuple[float, float]] = None,
    max_points: int = 50_000,
    grid_size: int = 50,
    hexbin_gridsize: int = 30,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    device: Optional[str] = None,
    power_dbm: Optional[float] = None,
    title: Optional[str] = None,
    savefig: Optional[str] = None,
    show: bool = False,
) -> "matplotlib.axes.Axes":
    """Overlay time-stream I/Q data on the fitted resonator sweep circle.

    Renders three things in the same complex (I/Q) plane, all projected into the
    same *basis*:

    - the time-stream cloud *ts* (its density via *density*),
    - the smooth fitted sweep trace of *sw*, coloured by frequency detuning, and
    - marker points at the resonance ``fr`` and at ``fr ± freq_shift``.

    The sweep is re-fitted internally with :mod:`resonator_tools` so the smooth
    ``z_data_sim`` trace and the calibration parameters
    (``environmental_term``, ``phi0``, ``fr``) come from one self-consistent
    fit.

    :param ts: Complex time-stream data in the electronic (raw I/Q) basis. Any
        shape; it is flattened for the density plot.
    :param sw: A fitted :class:`~daq.measurements.sweep.Sweep` providing
        ``freq_arr`` and ``resp_arr``.
    :param qc: Optional complex "QC trace" calibration points (electronic
        basis) drawn as red circles. Skipped when ``None``.
    :param basis: Display basis passed to :func:`_to_basis`. One of
        ``"electronic"``, ``"fractional"``, ``"resonator"``. Defaults to
        ``"electronic"``.
    :param freq_shift: Detuning in Hz for the ``fr ± freq_shift`` marker
        diamonds. Defaults to ``400e3``.
    :param density: How to render the time-stream cloud: ``"scatter"`` (points),
        ``"kde"`` (scatter plus Gaussian-KDE contours; accurate but slow),
        ``"contour"`` (scatter plus fast histogram-based 1-sigma / 2-sigma
        contours), ``"hexbin"``, or ``"hist2d"``. Defaults to ``"scatter"``.
    :param fcrop: Optional ``(f_min, f_max)`` crop passed to the resonator
        autofit. When ``None``, the fit is cropped to the half-span centred on
        the amplitude minimum, matching :meth:`Sweep.fit`.
    :param max_points: Cap on the number of time-stream points used for KDE and
        scatter rendering; larger clouds are subsampled. Defaults to ``50_000``.
    :param grid_size: Grid resolution per axis for the ``"kde"`` contour grid
        and the ``"contour"`` histogram bins. Defaults to ``50``.
    :param hexbin_gridsize: Hexbin grid resolution. Defaults to ``30``.
    :param xlim: Optional x-axis limits ``(lo, hi)``.
    :param ylim: Optional y-axis limits ``(lo, hi)``.
    :param ax: Optional existing axis to draw on. A new figure is created when
        ``None``.
    :param device: Optional device label for the auto-generated title.
    :param power_dbm: Optional drive power (dBm at the device) for the title.
    :param title: Explicit title; overrides the auto-generated one when given.
    :param savefig: When given, save the figure to this path
        (``bbox_inches="tight"``).
    :param show: When ``True``, call :func:`matplotlib.pyplot.show`. Defaults to
        ``False``.
    :returns: The matplotlib axis the data was drawn on.
    :raises ValueError: If *basis* or *density* is not recognised, or *sw* is
        not fitted.
    """
    import matplotlib.pyplot as plt
    from resonator_tools import circuit

    if density not in ("scatter", "kde", "contour", "hexbin", "hist2d"):
        raise ValueError(
            "density must be one of 'scatter', 'kde', 'contour', 'hexbin', 'hist2d'; "
            f"got {density!r}"
        )
    if sw.freq_arr is None or sw.resp_arr is None:
        raise ValueError("sw must have freq_arr and resp_arr populated; run the sweep first")

    freq_arr = np.asarray(sw.freq_arr)
    resp_arr = np.asarray(sw.resp_arr)

    # --- Re-fit for a self-consistent smooth trace + calibration parameters ---
    if fcrop is None:
        f_ctr = freq_arr[np.argmin(np.abs(resp_arr))]
        span = float(freq_arr.max() - freq_arr.min())
        fcrop = (
            max(f_ctr - span / 4, freq_arr.min()),
            min(f_ctr + span / 4, freq_arr.max()),
        )
    port = circuit.notch_port(freq_arr, resp_arr)
    port.autofit(fcrop=fcrop)

    fit = port.fitresults
    env = np.asarray(fit["environmental_term"])
    phi0 = fit["phi0"]
    fr = fit["fr"]
    fr_idx = int(np.argmin(np.abs(freq_arr - fr)))
    env_fr = env[fr_idx]

    # --- Project everything into the requested basis ---
    swz = _to_basis(np.asarray(port.z_data_sim), env, phi0, basis)
    tsz = _to_basis(np.asarray(ts), env_fr, phi0, basis)
    qcz = None if qc is None else _to_basis(np.asarray(qc), env_fr, phi0, basis)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    ts_real = tsz.real.ravel()
    ts_imag = tsz.imag.ravel()

    # --- Time-stream cloud ---
    if density in ("scatter", "kde", "contour"):
        step = max(1, ts_real.size // max_points)
        ax.scatter(
            ts_real[::step],
            ts_imag[::step],
            color="tab:blue",
            s=0.05,
            alpha=0.005,
            label="time stream",
        )
        if density == "kde":
            _add_kde_contours(ax, ts_real, ts_imag, "tab:blue", max_points, grid_size)
        elif density == "contour":
            _add_hist_contours(ax, ts_real, ts_imag, "tab:blue", grid_size)
    elif density == "hexbin":
        if ts_real.size > max_points:
            step = ts_real.size // max_points
            ts_real, ts_imag = ts_real[::step], ts_imag[::step]
        ax.hexbin(ts_real, ts_imag, gridsize=hexbin_gridsize, alpha=0.6, cmap="Blues")
    else:  # hist2d
        if ts_real.size > max_points:
            step = ts_real.size // max_points
            ts_real, ts_imag = ts_real[::step], ts_imag[::step]
        ax.hist2d(ts_real, ts_imag, bins=50, alpha=0.6, cmap="Blues")

    # --- Marker points ---
    ax.scatter(
        swz.real[fr_idx],
        swz.imag[fr_idx],
        color="tab:red",
        s=50,
        marker="x",
        zorder=10,
        label=f"$f_r$ = {fr / 1e9:.6f} GHz",
    )
    if qcz is not None:
        ax.scatter(
            qcz.real,
            qcz.imag,
            color="red",
            s=50,
            marker="o",
            zorder=10,
            label="QC trace",
        )
    for sign in (+1, -1):
        s_idx = int(np.argmin(np.abs(freq_arr - (fr + sign * freq_shift))))
        ax.scatter(
            swz.real[s_idx],
            swz.imag[s_idx],
            color="k",
            s=50,
            marker="d",
            zorder=10,
            label=f"$f_r$ {'+' if sign > 0 else '-'} {freq_shift / 1e6:.2f} MHz",
        )

    # --- Sweep trace coloured by detuning ---
    trace = ax.scatter(
        swz.real,
        swz.imag,
        c=(freq_arr - fr) * 1e-3,
        cmap="magma",
    )
    ax.get_figure().colorbar(trace, ax=ax, label=r"Frequency shift from $f_r$ [kHz]")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("I [FS]")
    ax.set_ylabel("Q [FS]")
    if title is None:
        parts = []
        if device is not None:
            parts.append(str(device))
        parts.append(f"{basis} basis")
        if power_dbm is not None:
            parts.append(f"{power_dbm} dBm at device")
        title = "\n".join(parts)
    ax.set_title(title)

    if savefig is not None:
        ax.get_figure().savefig(savefig, bbox_inches="tight")
    if show:
        plt.show()

    return ax
