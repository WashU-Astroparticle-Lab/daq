# -*- coding: utf-8 -*-
"""Plotting helpers for noise / resonator analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import matplotlib.axes
    from daq.measurements.sweep import Sweep

Basis = Literal["electronic", "fractional", "resonator"]
Density = Literal["scatter", "kde", "contour", "hexbin", "hist2d"]

#: Bases a noise spectrum can be plotted in. Narrower than :data:`Basis` on purpose: the
#: repo's PSD producers (``averaged_psd_timestream``, ``averaged_psd_cleaned``) emit either
#: the resonator-basis dissipation/frequency pair or the raw I/Q pair, and nothing projects a
#: spectrum into the fractional basis.
PsdBasis = Literal["resonator", "electronic"]

#: Channel names and PSD units per basis. The resonator-basis response is dimensionless
#: (a fractional frequency / dissipation shift), so its spectrum is a plain ``1/Hz``.
_PSD_AXES: Dict[str, Tuple[Tuple[str, str], str]] = {
    "resonator": (("Dissipation", "Frequency"), "1/Hz"),
    "electronic": (("I", "Q"), "FS$^2$/Hz"),
}


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
        the full array for the sweep trace, or the scalar value at the single
        frequency a time stream or a set of QC points was acquired at. See
        :func:`_readout_env` for why that frequency, and not ``fr``.
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


def _readout_env(
    fit: Dict[str, Any],
    freq_arr: npt.NDArray[np.floating],
    readout_freq: Optional[float],
    basis: Basis,
) -> complex:
    """Return the environmental term at the frequency the cloud and QC points were taken at.

    Thin adapter over :func:`~daq.analysis.resonator.readout_environmental_term`, which owns
    the reasoning and the fallback warning. Kept as a separate function only to bind this
    plot's vocabulary (a display *basis*) to that general one.

    The sweep trace is normalized point by point, each frequency by its own environmental
    term; the cloud and its QC points sit at *one* frequency and need it evaluated there.
    Using ``fr`` instead leaves the cable-delay phase ``2 pi (f_ro - fr) tau`` uncorrected,
    rotating them rigidly about the origin -- **off** the fitted circle rather than along it,
    by roughly ``2 pi (f_ro - fr) tau`` against a circle of radius ``Ql / (2 |Qc|)``.

    :param fit: A :func:`~daq.analysis.resonator.fit_notch` ``fitresults`` mapping.
    :param freq_arr: The sweep frequencies, used only for the on-resonance fallback.
    :param readout_freq: Frequency in hertz the single-frequency data was acquired at.
        ``None`` falls back to ``fr``, i.e. assumes the data was taken on resonance.
    :param basis: The display basis. The electronic basis returns the data untouched, so the
        fallback cannot affect the plot and stays quiet there.
    :raises ValueError: If *readout_freq* is not positive and finite.
    :returns: The complex environmental term to divide the single-frequency data by.

    """
    from .resonator import readout_environmental_term

    return readout_environmental_term(
        fit,
        readout_freq,
        freq_arr=freq_arr,
        # The electronic basis returns z untouched, so the choice cannot matter there.
        warn=basis != "electronic",
        caller="plot_iq_comparison",
        remedy="Pass readout_freq=<acquisition frequency in Hz>",
        # plot_iq_comparison -> _readout_env -> readout_environmental_term -> warn, so 4
        # lands on the user's call. Update this if the chain gains or loses a hop.
        stacklevel=4,
    )


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
    readout_freq: Optional[float] = None,
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
        shape; it is flattened for the density plot. Taken at a single
        frequency -- pass it as *readout_freq* whenever it is not ``fr``.
    :param sw: A :class:`~daq.measurements.sweep.Sweep` that has been run,
        providing populated ``freq_arr`` and ``resp_arr``. The resonator fit is
        performed internally, so ``sw`` need not be fitted beforehand.
    :param qc: Optional complex "QC trace" calibration points (electronic
        basis) drawn as red circles. Skipped when ``None``. Assumed to come
        from the same acquisition as *ts*, hence the same *readout_freq*.
    :param basis: Display basis passed to :func:`_to_basis`. One of
        ``"electronic"``, ``"fractional"``, ``"resonator"``. Defaults to
        ``"electronic"``.
    :param readout_freq: Frequency in hertz that *ts* and *qc* were acquired
        at -- ``TimeStream.signal_freqs[i]`` for tone *i* (``lo_freq`` only
        when that tone's IF is zero), or ``QCTrace.readout_freq``. Used to
        evaluate the environmental term at
        that frequency rather than at ``fr``, which is what keeps the cloud on
        the fitted circle: the term carries the cable delay, so normalizing
        off-resonance data at ``fr`` leaves a phase ``2*pi*(f_ro - fr)*tau``
        that rotates the cloud about the origin. The displacement is about
        ``2*pi*(f_ro - fr)*tau`` against a circle of radius ``Ql/(2|Qc|)``, so
        on a shallow dip a few hundred kHz of detuning is enough to put the
        cloud entirely off the ring. ``None`` (the default) assumes the data
        was taken on resonance and warns in the normalizing bases.
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
    :raises ValueError: If *basis* or *density* is not recognised, *readout_freq*
        is not positive, or *sw* has no ``freq_arr``/``resp_arr`` (i.e. has not
        been run).
    """
    import matplotlib.pyplot as plt

    from .resonator import fit_notch

    if density not in ("scatter", "kde", "contour", "hexbin", "hist2d"):
        raise ValueError(
            "density must be one of 'scatter', 'kde', 'contour', 'hexbin', 'hist2d'; "
            f"got {density!r}"
        )
    if sw.freq_arr is None or sw.resp_arr is None:
        raise ValueError("sw must have freq_arr and resp_arr populated; run the sweep first")
    if readout_freq is not None:
        # Validated up front rather than where it is used, so a bad frequency does not cost a
        # full resonator fit first. Matches GateBiasMeasurement._init_readout.
        from .resonator import validate_readout_freq

        validate_readout_freq(readout_freq)

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
    port = fit_notch(freq_arr, resp_arr, fcrop=fcrop)

    fit = port.fitresults
    env = np.asarray(fit["environmental_term"])
    phi0 = fit["phi0"]
    fr = fit["fr"]
    fr_idx = int(np.argmin(np.abs(freq_arr - fr)))

    # The sweep trace spans frequencies and is normalized point by point; the time stream and
    # the QC points sit at one frequency and are normalized by the term evaluated there. See
    # _readout_env for why using fr for the latter throws the cloud off the circle.
    env_ro = _readout_env(fit, freq_arr, readout_freq, basis)

    # --- Project everything into the requested basis ---
    swz = _to_basis(np.asarray(port.z_data_sim), env, phi0, basis)
    tsz = _to_basis(np.asarray(ts), env_ro, phi0, basis)
    qcz = None if qc is None else _to_basis(np.asarray(qc), env_ro, phi0, basis)

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


def plot_psd(
    f: npt.ArrayLike,
    psd_a: Optional[npt.ArrayLike],
    psd_b: Optional[npt.ArrayLike] = None,
    *,
    basis: PsdBasis = "resonator",
    f_bw: Optional[float] = None,
    fit: bool = True,
    labels: Optional[Tuple[str, str]] = None,
    units: Optional[str] = None,
    tone_labels: Optional[Sequence[str]] = None,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    savefig: Optional[str] = None,
    show: bool = False,
    **fit_kwargs: Any,
) -> Tuple[Tuple["matplotlib.axes.Axes", ...], Dict[str, Any]]:
    r"""Plot one or two noise spectra on log-log axes, fitting each to the parity model.

    This is the drawing half of the repo's PSD workflow. Every producer here --
    :func:`~daq.analysis.noise.averaged_psd_timestream`,
    :func:`~daq.analysis.noise.averaged_psd_cleaned`, and a hand-rolled
    :func:`~daq.analysis.noise.compute_psd` -- returns ``(f, psd_a, psd_b)`` whose two
    channels mean different things in different bases, and *basis* is what names them:

    ==============  ==================  ====================  ============
    *basis*         *psd_a*             *psd_b*               units
    ==============  ==================  ====================  ============
    ``resonator``   dissipation (rad)   frequency (arc)       ``1/Hz``
    ``electronic``  I (real)            Q (imaginary)         ``FS^2/Hz``
    ==============  ==================  ====================  ============

    Pass *labels* / *units* to override, for a projection that is neither -- e.g.
    ``labels=("|S|", "")`` for :class:`~daq.measurements.bias_hunt.BiasHunt`'s magnitude
    spectrum.

    **Each channel is fit from the array passed in**, independently, with
    :func:`~daq.analysis.noise.fit_parity_psd`. Nothing is read off a measurement object: a
    stored ``fit_results`` describes only whichever single channel was last computed, which
    is not comparable with the other panel and need not correspond to the arrays being
    plotted at all.

    Each panel draws the raw periodogram faintly, the log-binned points the fit actually used
    as markers, and frames its y-axis on those. A bare periodogram scatters over ten decades,
    so framing on it flattens everything of interest into a line and makes a good fit look
    like a bad one.

    A fit that raises is reported on its panel and in the console rather than costing the
    spectrum -- a spectrum this model does not describe is itself a result.

    :param f: Frequency axis in hertz, as returned by
        :func:`~daq.analysis.noise.compute_psd`.
    :param psd_a: First channel, 1-D or 2-D with one PSD per row (one row per tone). ``None``
        skips its panel.
    :param psd_b: Second channel, same shape rules. ``None`` (the default) draws one panel.
    :param basis: Which basis the channels are in; sets the panel labels and units.
    :param f_bw: Sampling bandwidth in hertz held fixed by the fit -- the *tuned* sample rate
        the streams were taken at (``TimeStream.df``). ``None`` infers ``2 * f[-1]``, which is
        exact to within one frequency bin.
    :param fit: Fit and overlay the random-telegraph model. ``False`` draws the spectra alone.
    :param labels: ``(name_a, name_b)`` overriding the *basis* channel names.
    :param units: PSD unit string for the y-axis label, overriding the *basis* default.
    :param tone_labels: One legend label per row, for multi-tone input. Defaults to
        ``tone 0``, ``tone 1``, ...
    :param ax: An existing axis, or a sequence of them -- one per channel drawn. A new figure
        is created when ``None``.
    :param title: Figure title.
    :param savefig: Optional path to save the figure to.
    :param show: Whether to call ``plt.show()``.
    :param fit_kwargs: Passed to :func:`~daq.analysis.noise.fit_parity_psd` for **both**
        channels (e.g. ``fit_onef=True``, ``n_bins``, ``bin_weighting``). Fit one channel at a
        time to give them different settings.
    :raises ValueError: If both channels are ``None``, if *basis* is unknown, if a channel's
        last axis does not match *f*, or if *ax* holds the wrong number of axes.
    :returns: ``(axes, fits)`` -- the axes drawn into, one per channel present, and a dict
        mapping ``"a"``/``"b"`` to that channel's
        :func:`~daq.analysis.noise.fit_parity_psd` result. Following that function, the value
        is a dict for 1-D input and a list of dicts for 2-D; it is ``None`` where the channel
        was absent, not fitted, or failed.

    """
    import matplotlib.pyplot as plt

    from .noise import fit_parity_psd

    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 1 or f.size < 2:
        raise ValueError(f"f must be a 1-D frequency axis with at least 2 points, got {f.shape}")
    if basis not in _PSD_AXES:
        raise ValueError(f"basis must be one of {sorted(_PSD_AXES)}, got {basis!r}")

    default_labels, default_units = _PSD_AXES[basis]
    names = tuple(labels) if labels is not None else default_labels
    unit = default_units if units is None else units

    channels = []
    for key, psd, name in (("a", psd_a, names[0]), ("b", psd_b, names[1])):
        if psd is None:
            continue
        arr = np.asarray(psd, dtype=np.float64)
        if arr.ndim not in (1, 2):
            raise ValueError(f"psd_{key} must be 1-D or 2-D, got {arr.ndim}-D")
        if arr.shape[-1] != f.size:
            raise ValueError(
                f"psd_{key} has {arr.shape[-1]} points against {f.size} for f; they come from "
                "one compute_psd call and must match."
            )
        # Keep the caller's dimensionality: fit_parity_psd returns a dict for 1-D and a list
        # for 2-D, and the returned fits follow that so callers need no special case here.
        channels.append((key, np.atleast_2d(arr), name, arr.ndim))
    if not channels:
        raise ValueError("Both psd_a and psd_b are None; there is nothing to plot.")

    if f_bw is None:
        # f runs 0 .. fs/2, so twice the last bin recovers the rate to within one bin. That
        # only shifts the white floor by the same fraction, far below what it can resolve.
        f_bw = 2.0 * float(f[-1])

    if ax is None:
        _, axes_grid = plt.subplots(
            1,
            len(channels),
            figsize=(6.0 * len(channels), 4.5),
            squeeze=False,
            tight_layout=True,
        )
        axes = tuple(axes_grid[0])
    else:
        axes = (ax,) if hasattr(ax, "loglog") else tuple(ax)
        if len(axes) != len(channels):
            raise ValueError(f"ax holds {len(axes)} axes but {len(channels)} channels were given.")

    # The DC bin is unplottable on a log axis and carries no information anyway: it is the
    # operating point, which every projection here has already mean-subtracted.
    keep = f > 0
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    fits: Dict[str, Any] = {"a": None, "b": None}

    for axis, (key, arr, name, ndim) in zip(axes, channels):
        n_rows = arr.shape[0]

        result = None
        if fit:
            try:
                result = fit_parity_psd(f, arr if ndim == 2 else arr[0], f_bw=f_bw, **fit_kwargs)
                fits[key] = result
            except Exception as err:  # noqa: BLE001 - the spectrum is worth more than the fit
                print(f"WARN: the {name.lower()} fit failed, showing the spectrum alone: {err}")
                axis.set_title(f"{name}: parity fit failed", fontsize=8, color="tab:red")
        rows = [result] if isinstance(result, dict) else list(result or [])

        binned_lo, binned_hi = [], []
        for irow in range(n_rows):
            color = cycle[irow % len(cycle)]
            if tone_labels is not None and irow < len(tone_labels):
                row_label = str(tone_labels[irow])
            else:
                row_label = f"tone {irow}" if n_rows > 1 else "periodogram"
            axis.loglog(f[keep], arr[irow][keep], lw=0.5, color=color, alpha=0.25, label=row_label)

            res = rows[irow] if irow < len(rows) else None
            if res is None:
                continue
            # One colour per tone, but a lone spectrum reads better with the model in
            # contrast against it.
            model_color = "tab:red" if n_rows == 1 else color
            axis.loglog(
                res["f_binned"],
                res["psd_binned"],
                "o",
                ms=3.5,
                color=color,
                label=f"log-binned ({res['n_bins']} bins, fit to these)" if n_rows == 1 else None,
            )
            if n_rows == 1:
                model_label = (
                    rf"$\Gamma_p$ = {res['gamma_p']:.3g} $\pm$ {res['gamma_p_err']:.2g} Hz"
                    "\n"
                    rf"$f_c$ = {res['f_corner']:.3g} Hz,  $F$ = {res['fidelity']:.3f}"
                    "\n"
                    rf"resid = {res['resid_dex_rms']:.2f} dex"
                )
            else:
                model_label = (
                    rf"{row_label}: $\Gamma_p$ = {res['gamma_p']:.3g} Hz, "
                    rf"resid = {res['resid_dex_rms']:.2f} dex"
                )
            axis.loglog(
                f[keep],
                np.asarray(res["model"])[keep],
                color=model_color,
                lw=1.8,
                label=model_label,
            )
            axis.axvline(res["f_corner"], color=model_color, ls=":", lw=1.0)
            binned_lo.append(float(np.min(res["psd_binned"])))
            binned_hi.append(float(np.max(res["psd_binned"])))

        if binned_lo:
            # Frame on the binned points: the periodogram's low-power outliers otherwise drag
            # the y-range over ten decades and flatten everything of interest into a line.
            axis.set_ylim(min(binned_lo) / 30.0, max(binned_hi) * 30.0)
        axis.set_xlabel("Frequency [Hz]")
        axis.set_ylabel(f"{name} PSD [{unit}]" if name else f"PSD [{unit}]")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(loc="best", fontsize=8)

    if title is not None:
        axes[0].get_figure().suptitle(title)
    if savefig is not None:
        axes[0].get_figure().savefig(savefig, bbox_inches="tight")
    if show:
        plt.show()

    return tuple(axes), fits
