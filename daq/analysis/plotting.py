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
    freq_shift: Optional[float] = None,
    density: Density = "scatter",
    fcrop: Optional[Tuple[float, float]] = None,
    max_points: int = 50_000,
    grid_size: int = 50,
    hexbin_gridsize: int = 30,
    scatter_size: float = 0.05,
    scatter_alpha: float = 0.005,
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
    :param sw: A :class:`~daq.measurements.sweep.Sweep` that has been run,
        providing populated ``freq_arr`` and ``resp_arr``. The resonator fit is
        performed internally, so ``sw`` need not be fitted beforehand.
    :param qc: Optional complex "QC trace" calibration points (electronic
        basis) drawn as red circles. Skipped when ``None``.
    :param basis: Display basis passed to :func:`_to_basis`. One of
        ``"electronic"``, ``"fractional"``, ``"resonator"``. Defaults to
        ``"electronic"``.
    :param freq_shift: Detuning in Hz for the ``fr ± freq_shift`` marker
        diamonds. Defaults to ``400e3``. When ``None``, these markers are not
        drawn.
    :param density: How to render the time-stream cloud: ``"scatter"`` (points),
        ``"kde"`` (scatter plus Gaussian-KDE contours; accurate but slow),
        ``"contour"`` (scatter plus fast histogram-based 1-sigma / 2-sigma
        contours), ``"hexbin"``, or ``"hist2d"``. Defaults to ``"scatter"``.
    :param fcrop: Optional ``(f_min, f_max)`` crop passed to the resonator
        autofit. When ``None``, the fit is cropped to the half-span (a quarter
        of the data span on each side) centred on the amplitude minimum,
        mirroring :meth:`Sweep.fit`. Note the span is taken from the data
        (``freq_arr``) and may differ from ``sw.freq_span`` by up to one step.
    :param max_points: Cap on the number of time-stream points used for KDE and
        scatter rendering; larger clouds are subsampled. Defaults to ``50_000``.
    :param grid_size: Grid resolution per axis for the ``"kde"`` contour grid
        and the ``"contour"`` histogram bins. Defaults to ``50``.
    :param hexbin_gridsize: Hexbin grid resolution. Defaults to ``30``.
    :param scatter_size: Marker size for the ``"scatter"``/``"kde"``/``"contour"``
        cloud. The default ``0.05`` is tuned for million-point clouds; raise it
        (e.g. to ``1``-``5``) for smaller time streams so the points are
        visible.
    :param scatter_alpha: Marker alpha for the scatter cloud. The default
        ``0.005`` is tuned for million-point clouds; raise it (e.g. to
        ``0.1``-``0.5``) for smaller time streams.
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
    :raises ValueError: If *basis* or *density* is not recognised, or *sw* has
        no ``freq_arr``/``resp_arr`` (i.e. has not been run).
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
            s=scatter_size,
            alpha=scatter_alpha,
            label="time stream",
        )
        if density == "kde":
            _add_kde_contours(ax, ts_real, ts_imag, "tab:blue", max_points, grid_size)
        elif density == "contour":
            _add_hist_contours(ax, ts_real, ts_imag, "tab:blue", grid_size)
    elif ts_real.size == 0:
        pass  # nothing to bin
    elif density == "hexbin":
        if ts_real.size > max_points:
            step = ts_real.size // max_points
            ts_real, ts_imag = ts_real[::step], ts_imag[::step]
        ax.hexbin(ts_real, ts_imag, gridsize=hexbin_gridsize, alpha=0.6, cmap="Blues")
    else:  # hist2d
        if ts_real.size > max_points:
            step = ts_real.size // max_points
            ts_real, ts_imag = ts_real[::step], ts_imag[::step]
        ax.hist2d(ts_real, ts_imag, bins=grid_size, alpha=0.6, cmap="Blues")

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
            facecolors="none",
            edgecolors="tab:orange",
            s=5,
            marker="o",
            zorder=10,
            label="QC trace",
        )
   
    if freq_shift is not None:
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
            parts.append(f"{power_dbm:g} dBm at device")
        title = "\n".join(parts)
    ax.set_title(title)

    if savefig is not None:
        ax.get_figure().savefig(savefig, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def plot_qc_trace(
    time_ms: npt.ArrayLike,
    avg_iq: npt.ArrayLike,
    raw: Optional[npt.ArrayLike] = None,
    *,
    ax: Optional[Tuple["matplotlib.axes.Axes", "matplotlib.axes.Axes"]] = None,
    tone: int = 0,
    title: str = "QC trace (block-averaged)",
    savefig: Optional[str] = None,
    show: bool = False,
) -> Tuple["matplotlib.axes.Axes", "matplotlib.axes.Axes"]:
    """Plot a block-averaged QC trace: I and Q against time over one drive period.

    Takes the output of :func:`~daq.analysis.folding.fold_timestream`. Passing the unfolded
    time stream as *raw* overlays a single un-averaged period on the same panels, which shows
    directly how much the averaging bought.

    :param time_ms: Time axis of one drive period in milliseconds.
    :param avg_iq: Block-averaged trace of shape ``(2, n_samples)``; row 0 is I, row 1 is Q.
    :param raw: Optional unfolded time stream or complex array to overlay. It is aligned from
        the start and truncated to the length of one period.
    :param ax: Existing ``(ax_i, ax_q)`` pair to draw into. A new figure is created when
        ``None``.
    :param tone: Which tone to take from *raw*, for a multi-tone time stream.
    :param title: Figure title.
    :param savefig: Optional path to save the figure to.
    :param show: Whether to call ``plt.show()``.
    :raises ValueError: If *avg_iq* does not have two rows.
    :returns: The ``(ax_i, ax_q)`` pair that was drawn into.

    """
    import matplotlib.pyplot as plt

    from .folding import _as_complex

    time_ms = np.asarray(time_ms)
    avg_iq = np.asarray(avg_iq)
    if avg_iq.ndim != 2 or avg_iq.shape[0] != 2:
        raise ValueError(f"avg_iq must have shape (2, n_samples), got {avg_iq.shape}")

    if ax is None:
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), tight_layout=True)
        ax_i, ax_q = axes
    else:
        ax_i, ax_q = ax

    if raw is not None:
        # Draw the raw trace first so the averaged curve stays readable on top of it.
        z = _as_complex(raw, tone=tone)
        n = min(len(time_ms), avg_iq.shape[1], len(z))
        ax_i.plot(time_ms[:n], z.real[:n], color="tab:gray", alpha=0.6, lw=0.7, label="raw")
        ax_q.plot(time_ms[:n], z.imag[:n], color="tab:gray", alpha=0.6, lw=0.7, label="raw")

    ax_i.plot(time_ms, avg_iq[0], color="tab:blue", label="block-averaged")
    ax_i.set_ylabel("I [FS]")
    ax_i.grid(True, alpha=0.3)

    ax_q.plot(time_ms, avg_iq[1], color="tab:red", label="block-averaged")
    ax_q.set_ylabel("Q [FS]")
    ax_q.set_xlabel("Time [ms]")
    ax_q.grid(True, alpha=0.3)

    if raw is not None:
        ax_i.legend(loc="best", fontsize=8)
        ax_q.legend(loc="best", fontsize=8)

    ax_i.get_figure().suptitle(title)

    if savefig is not None:
        ax_i.get_figure().savefig(savefig, bbox_inches="tight")
    if show:
        plt.show()

    return ax_i, ax_q
