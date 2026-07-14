# -*- coding: utf-8 -*-
"""Noise analysis utilities."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from daq.measurements.sweep import Sweep
    from daq.measurements.timestream import TimeStream

FloatAny = Union[float, List[float], npt.NDArray[np.floating]]
BoolAny = Union[bool, List[bool], npt.NDArray[np.bool_]]


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
        ``welch`` is ``True``. Defaults to ``"constant"`` (mean removal); note
        this differs from the periodogram path, which applies no detrending.
        Pass ``False`` to match the periodogram's behavior.
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


def parity_psd_model(
    f: npt.ArrayLike,
    fidelity: float,
    gamma_p: float,
    f_bw: float,
    a_onef: float = 0.0,
    alpha: float = 1.0,
) -> npt.NDArray[np.floating]:
    r"""Random-telegraph parity PSD model (Eqn. 18 of arXiv:2601.16261).

    A parity time-stream that switches between two states at a characteristic rate
    :math:`\Gamma_p`, read out with fidelity :math:`F`, has a one-sided power
    spectral density

    .. math::

        \mathrm{PSD}(f) = F^2\,\frac{4\Gamma_p}{(2\Gamma_p)^2 + (2\pi f)^2}
                          + (1 - F^2)\,f_\mathrm{bw}^{-1}
                          + \frac{A}{f^{\alpha}}.

    The first term is the Lorentzian of the random-telegraph (parity-switching)
    process; the second is a white noise floor set by the finite readout fidelity
    and the sampling bandwidth. The optional third term is a ``1/f``-like
    low-frequency excess (e.g. drift or two-level-system noise); it is disabled by
    default (``a_onef = 0``), which recovers the two-term form of Eqn. 18. At
    :math:`f = 0` the Lorentzian reduces to the finite value :math:`F^2 / \Gamma_p`
    and the (divergent) ``1/f`` term is set to zero.

    :param f: Fourier frequency or frequencies in Hz (as returned by
        :func:`compute_psd`).
    :param fidelity: Readout fidelity :math:`F` (dimensionless, ``0`` to ``1``).
    :param gamma_p: Characteristic parity-switching rate :math:`\Gamma_p` in Hz.
    :param f_bw: Sampling bandwidth :math:`f_\mathrm{bw}` in Hz (typically the
        acquisition sample rate, e.g. ``TimeStream.df``).
    :param a_onef: Amplitude :math:`A` of the ``1/f`` term, in
        (units of the time series)²/Hz at ``f = 1 Hz``. ``0`` (default) disables it.
    :param alpha: Exponent :math:`\alpha` of the ``1/f`` term (``1.0`` for pure
        ``1/f``). Only relevant when *a_onef* is non-zero.
    :returns: The model PSD evaluated at *f*, same shape as *f*, in
        (units of the time series)²/Hz.
    """
    f = np.asarray(f, dtype=np.float64)
    lorentzian = fidelity**2 * (4.0 * gamma_p) / ((2.0 * gamma_p) ** 2 + (2.0 * np.pi * f) ** 2)
    floor = (1.0 - fidelity**2) / f_bw
    # 1/f term, guarded so that f == 0 (and the a_onef == 0 case) stays finite.
    with np.errstate(divide="ignore", invalid="ignore"):
        onef = a_onef * np.abs(f) ** (-alpha)
    onef = np.where(np.isfinite(onef), onef, 0.0)
    return lorentzian + floor + onef


def fit_parity_psd(
    f: npt.ArrayLike,
    psd: npt.ArrayLike,
    f_bw: float,
    p0: Optional[Sequence[float]] = None,
    sigma: Optional[npt.ArrayLike] = None,
    n_bins: int = 60,
    bin_reduce: str = "median",
    bin_weighting: str = "uniform",
    absolute_sigma: bool = False,
    drop_dc: bool = True,
    fit_onef: bool = False,
    fit_alpha: bool = False,
    alpha: float = 1.0,
) -> dict:
    r"""Fit a parity-timestream PSD to Eqn. 18 of arXiv:2601.16261.

    Fits the output of :func:`compute_psd` to :func:`parity_psd_model` to extract
    the readout fidelity :math:`F` and the characteristic parity-switching rate
    :math:`\Gamma_p`. The sampling bandwidth :math:`f_\mathrm{bw}` is held fixed at
    *f_bw* (pass the acquisition sample rate, e.g. ``TimeStream.df``); only *F* and
    :math:`\Gamma_p` are free parameters.

    The fit is performed in **log-log space** — the natural representation of a
    PSD. A periodogram is linearly spaced in frequency, so roughly all of its
    points sit in the top decade; a plain least-squares fit in linear units is
    therefore dominated by high-frequency structure. To avoid this, the spectrum
    is first reduced onto *n_bins* logarithmically-spaced frequency bins (each
    bin's frequency is the geometric mean of its members and its PSD the median of
    the periodogram power within it — robust to line-noise spikes; see *bin_reduce*),
    giving every decade equal representation. The fit then minimizes the residual
    of :math:`\log_{10}\mathrm{PSD}` against :math:`\log_{10}\text{model}`, so the
    multi-decade dynamic range is handled and no single frequency region drives
    the result. The DC (:math:`f = 0`) bin and any non-positive frequency are
    always excluded (a log axis requires :math:`f > 0`).

    Set *fit_onef* to add a ``1/f``-like low-frequency term :math:`A / f^{\alpha}`
    (e.g. drift or two-level-system noise that would otherwise bias :math:`\Gamma_p`
    and :math:`F`). The amplitude :math:`A` then becomes a free parameter; the
    exponent :math:`\alpha` is fixed at *alpha* unless *fit_alpha* is also ``True``,
    in which case it is fit too.

    For 2-D *psd* (shape ``(n_rows, n_freqs)``, as produced by
    :func:`compute_psd`, :func:`averaged_psd_timestream`, etc.), each row is fit
    independently and a list of result dicts is returned.

    :param f: Frequency axis in Hz (1-D), as returned by :func:`compute_psd`.
    :param psd: Power spectral density to fit. Either 1-D (``len(f)``) or 2-D
        (``(n_rows, len(f))``) with one PSD per row.
    :param f_bw: Sampling bandwidth in Hz, held fixed during the fit.
    :param p0: Optional initial guess for the free parameters, in order
        ``(fidelity, gamma_p[, a_onef[, alpha]])`` — length must match the enabled
        terms. When ``None`` the guess is estimated from the PSD.
    :param sigma: Optional per-point 1-sigma uncertainties on the *linear* PSD (same
        length as *f*). When given, they are propagated into the log-space,
        log-binned fit (the per-bin log-PSD error becomes the standard error of the
        bin's mean, delta-method converted to :math:`\log_{10}`). When ``None``
        (default), each bin's log-PSD error is estimated empirically from the scatter
        of the periodogram points that fall in it.
    :param n_bins: Number of logarithmically-spaced frequency bins to average the
        spectrum onto before fitting (default ``60``). Empty bins are dropped, so the
        effective count may be smaller; the value actually used is returned as
        ``n_bins``.
    :param bin_reduce: How to reduce the periodogram points within each bin.
        ``"median"`` (default) is robust to sparse line-noise spikes (50/60 Hz pickup
        and harmonics, glitches), which an arithmetic mean would otherwise let bleed
        into a spurious ``1/f`` term (a runaway low-frequency overshoot). ``"mean"``
        uses the arithmetic mean, which is unbiased for the underlying spectrum but
        spike-sensitive; prefer it only for clean, well-averaged, spike-free spectra.
    :param bin_weighting: How to weight the log-binned points when *sigma* is not
        given. ``"uniform"`` (default) weights every bin equally, so each decade
        contributes equally and the fit is not dictated by the densely-sampled high
        frequencies — the intended behaviour of a log-log fit. ``"count"`` instead
        weights by each bin's statistical precision (:math:`\sim 1/\sqrt{m}` for
        :math:`m` points in the bin), which is optimal when the model is trusted
        everywhere but hands the populous high-frequency bins far more influence.
        Ignored when *sigma* is supplied.
    :param absolute_sigma: When ``False`` (default), the reported parameter errors are
        rescaled by :math:`\sqrt{\chi^2/\mathrm{ndof}}` so they reflect the observed
        scatter (matching ``scipy``'s ``absolute_sigma=False``); appropriate when the
        per-bin errors are only relative weights. Set ``True`` when *sigma* is a true
        1-sigma uncertainty and the raw Hesse errors should be kept.
    :param drop_dc: Retained for compatibility. The DC bin is always excluded from a
        log-log fit (a log frequency axis requires ``f > 0``), so this has no effect.
    :param fit_onef: When ``True``, add a ``1/f`` term :math:`A / f^{\alpha}` with a
        free amplitude :math:`A`. Defaults to ``False`` (pure Eqn. 18). Enable it for a
        spectrum whose low-frequency rise is ``1/f``-like (drift / TLS noise) rather
        than a parity Lorentzian: without it the two-term model has nothing to describe
        the rise and collapses to the flat white floor.
    :param fit_alpha: When ``True`` (requires *fit_onef*), also fit the ``1/f``
        exponent :math:`\alpha`; otherwise it is held fixed at *alpha*.
    :param alpha: Fixed ``1/f`` exponent when *fit_alpha* is ``False`` (``1.0`` for
        pure ``1/f``), or the initial guess for :math:`\alpha` when *fit_alpha* is
        ``True``. Ignored when *fit_onef* is ``False``.
    :returns: For 1-D *psd*, a ``fit_results`` dict with a best-fit value and Hesse
        error for every term — ``fidelity`` / ``fidelity_err``, ``gamma_p`` (Hz) /
        ``gamma_p_err``, ``a_onef`` / ``a_onef_err``, ``alpha`` / ``alpha_err`` — plus
        the derived Lorentzian half-power frequency ``f_corner`` (Hz,
        :math:`\Gamma_p/\pi`) / ``f_corner_err``, the fixed ``f_bw``, the fit quality
        (``chi2``, ``ndof``, ``reduced_chi2`` — computed over the log-binned points),
        ``resid_dex_rms`` (the RMS of the ``log10`` data-vs-model residual over the
        binned points, in decades — a weighting-independent goodness-of-fit: ``~0.1`` is
        good, ``~1`` means the model is a decade off across the band, which
        ``reduced_chi2`` can hide under uniform weighting),
        the fitted ``model`` evaluated at every input *f* (including the dropped DC
        bin), the log-binned points that were actually fit (``f_binned``,
        ``psd_binned``) and the number of non-empty bins (``n_bins``), the underlying
        :class:`iminuit.Minuit` object under ``minuit``, and ``success``
        (``Minuit.valid``). A held-fixed ``1/f`` parameter is reported with a ``nan``
        error (``a_onef = 0`` when *fit_onef* is ``False``). For 2-D *psd*, a list of
        such dicts, one per row.
    :raises ValueError: If *f* is not 1-D, *psd* is not 1-D or 2-D, their frequency
        lengths do not match, *fit_alpha* is ``True`` without *fit_onef*,
        *bin_reduce* is not ``"median"`` or ``"mean"``, or *bin_weighting* is not
        ``"uniform"`` or ``"count"``.
    """
    from iminuit import Minuit
    from iminuit.cost import LeastSquares

    if fit_alpha and not fit_onef:
        raise ValueError("fit_alpha=True requires fit_onef=True")
    if bin_reduce not in ("median", "mean"):
        raise ValueError(f"bin_reduce must be 'median' or 'mean', got {bin_reduce!r}")
    if bin_weighting not in ("uniform", "count"):
        raise ValueError(f"bin_weighting must be 'uniform' or 'count', got {bin_weighting!r}")

    f = np.asarray(f, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError(f"f must be 1-D, got {f.ndim}-D")
    if psd.ndim not in (1, 2):
        raise ValueError(f"psd must be 1-D or 2-D, got {psd.ndim}-D")
    if psd.shape[-1] != f.shape[0]:
        raise ValueError(f"psd last axis ({psd.shape[-1]}) must match len(f) ({f.shape[0]})")

    if psd.ndim == 2:
        return [
            fit_parity_psd(
                f,
                row,
                f_bw,
                p0=p0,
                sigma=sigma,
                n_bins=n_bins,
                bin_reduce=bin_reduce,
                bin_weighting=bin_weighting,
                absolute_sigma=absolute_sigma,
                drop_dc=drop_dc,
                fit_onef=fit_onef,
                fit_alpha=fit_alpha,
                alpha=alpha,
            )
            for row in psd
        ]

    # --- Restrict to positive frequencies (a log-log fit requires f > 0) ---
    # `drop_dc` is retained for compatibility but a log axis always excludes f <= 0.
    mask = f > 0
    f_pos = f[mask]
    psd_pos = psd[mask]
    sigma_pos = np.asarray(sigma, dtype=np.float64)[mask] if sigma is not None else None
    if f_pos.size == 0:
        raise ValueError("no positive-frequency points to fit (all f <= 0)")
    free_names = ["fidelity", "gamma_p"]
    if fit_onef:
        free_names.append("a_onef")
    if fit_alpha:
        free_names.append("alpha")
    n_free = len(free_names)

    # --- Reduce onto log-spaced frequency bins, then fit in log-log space ---
    # A periodogram is linearly spaced in f, so most of its points crowd into the
    # top decade and would dominate a linear least-squares sum. Binning onto
    # log-spaced bins gives every decade equal representation, and fitting the
    # residual of log10(PSD) handles the multi-decade dynamic range. Each bin is
    # reduced by its median (default), which rejects sparse line-noise spikes
    # (50/60 Hz pickup and harmonics, glitches) that an arithmetic mean would let
    # bleed into a spurious 1/f term; pass bin_reduce="mean" for the (unbiased but
    # spike-sensitive) arithmetic mean.
    reduce_fn = np.mean if bin_reduce == "mean" else np.median
    ln10 = np.log(10.0)
    edges = np.logspace(np.log10(f_pos.min()), np.log10(f_pos.max()), int(n_bins) + 1)
    which = np.digitize(f_pos, edges[1:-1])  # -> bins 0 .. n_bins-1
    f_b, psd_b, logerr_b = [], [], []
    for b in range(int(n_bins)):
        sel = which == b
        m = int(np.count_nonzero(sel))
        if m == 0:
            continue
        pb = float(reduce_fn(psd_pos[sel]))  # median (robust) or mean power in the bin
        if pb <= 0.0 or not np.isfinite(pb):
            continue
        fb = float(np.exp(np.mean(np.log(f_pos[sel]))))  # geometric-mean frequency
        # Uncertainty on log10(pb), i.e. the LeastSquares yerror in log space.
        #  * With true per-point errors (`sigma`): propagate them — the error on the
        #    bin mean is sqrt(sum sigma_i^2)/m, converted to log10 by the delta method
        #    (d log10 y = dy / (y ln 10)).
        #  * bin_weighting="uniform" (default): every bin gets the same weight, so each
        #    log-frequency bin — hence each decade — contributes equally and the fit is
        #    not dictated by the densely-sampled high frequencies.
        #  * bin_weighting="count": weight by the statistical precision of the bin mean
        #    (~1/sqrt(m)); optimal if the model is trusted everywhere, but it hands the
        #    populous high-frequency bins much more weight than the sparse low-frequency
        #    ones (the very high-f dominance "uniform" avoids).
        if sigma_pos is not None:
            logerr = float(np.sqrt(np.sum(sigma_pos[sel] ** 2))) / m / (pb * ln10)
        elif bin_weighting == "count":
            logerr = 1.0 / (np.sqrt(m) * ln10)
        else:
            logerr = 1.0
        f_b.append(fb)
        psd_b.append(pb)
        logerr_b.append(logerr)

    f_b = np.asarray(f_b, dtype=np.float64)
    psd_b = np.asarray(psd_b, dtype=np.float64)
    logerr_b = np.asarray(logerr_b, dtype=np.float64)
    n_binned = int(f_b.size)
    if n_binned < n_free + 1:
        raise ValueError(
            f"need at least {n_free + 1} non-empty log-frequency bins to fit {n_free} "
            f"parameters, got {n_binned} (try lowering n_bins)"
        )
    # Guard non-positive / non-finite errors (LeastSquares divides by yerror).
    good = np.isfinite(logerr_b) & (logerr_b > 0)
    floor_val = float(np.min(logerr_b[good])) if np.any(good) else 1.0
    logerr_b[~good] = floor_val
    logpsd_b = np.log10(psd_b)

    # --- Initial guess: start from all four parameters, then fix the disabled ones ---
    start = {"fidelity": 0.9, "gamma_p": 1.0, "a_onef": 0.0, "alpha": alpha}
    # White floor from the top decade of the (binned) spectrum; fidelity from it.
    hi = f_b >= 0.1 * f_b.max()
    floor0 = float(np.median(psd_b[hi])) if np.any(hi) else float(np.median(psd_b))
    floor0 = max(floor0, 0.0)
    start["fidelity"] = float(np.sqrt(np.clip(1.0 - floor0 * f_bw, 1e-6, 1.0)))
    # Low-frequency plateau (Lorentzian DC value F^2 / gamma_p, sitting above floor).
    lo = f_b <= max(f_b.min() * 3.0, f_b[min(4, f_b.size - 1)])
    plateau0 = float(np.median(psd_b[lo])) if np.any(lo) else float(psd_b[0])
    lorentz_dc0 = max(plateau0 - floor0, np.finfo(float).tiny)
    # Guess gamma_p from the rolloff LOCATION, not the amplitude: the first frequency
    # where the spectrum falls to the geometric midpoint between the low-f plateau and
    # the high-f floor is the Lorentzian corner f_c, and gamma_p = pi * f_c. Deriving
    # gamma_p from the amplitude (F^2 / DC) instead misplaces the corner far outside the
    # band for a small-amplitude or 1/f-like spectrum, trapping MIGRAD in a flat solution.
    if plateau0 > floor0 > 0.0:
        mid_level = np.sqrt(plateau0 * floor0)  # geometric midpoint of the two levels
        below = np.nonzero(psd_b <= mid_level)[0]
        f_c0 = float(f_b[below[0]]) if below.size else float(f_b[-1])
    else:
        f_c0 = float(np.sqrt(f_b[0] * f_b[-1]))  # mid-band fallback
    f_c0 = float(np.clip(f_c0, f_b[0], f_b[-1]))
    start["gamma_p"] = max(np.pi * f_c0, 1e-3)
    if fit_onef:
        # Attribute the excess at the lowest fitted frequency (above the Lorentzian
        # plateau + floor) to the 1/f term: A ~ excess * f_min^alpha.
        f_min = float(f_b.min())
        excess = max(plateau0 - (lorentz_dc0 + floor0), 0.0)
        a_onef0 = excess * f_min**alpha
        start["a_onef"] = a_onef0 if a_onef0 > 0.0 else 0.1 * plateau0 * f_min**alpha

    if p0 is not None:
        p0 = tuple(p0)
        if len(p0) != n_free:
            raise ValueError(f"p0 must have {n_free} entries for the enabled terms, got {len(p0)}")
        for name, value in zip(free_names, p0):
            start[name] = float(value)

    tiny = float(np.finfo(np.float64).tiny)

    # Full four-parameter model in log10 space; disabled terms are frozen below.
    def _model_log(x, fidelity, gamma_p, a_onef, alpha):
        y = parity_psd_model(x, fidelity, gamma_p, f_bw, a_onef=a_onef, alpha=alpha)
        return np.log10(np.maximum(y, tiny))

    fit_results = {
        "fidelity": np.nan,
        "fidelity_err": np.nan,
        "gamma_p": np.nan,
        "gamma_p_err": np.nan,
        "f_corner": np.nan,
        "f_corner_err": np.nan,
        "a_onef": 0.0 if not fit_onef else np.nan,
        "a_onef_err": np.nan,
        "alpha": alpha,
        "alpha_err": np.nan,
        "f_bw": float(f_bw),
        "chi2": np.nan,
        "ndof": int(n_binned - n_free),
        "reduced_chi2": np.nan,
        "resid_dex_rms": np.nan,
        "model": None,
        "f_binned": f_b,
        "psd_binned": psd_b,
        "n_bins": n_binned,
        "minuit": None,
        "success": False,
    }

    try:
        cost = LeastSquares(f_b, logpsd_b, logerr_b, _model_log)
        minuit = Minuit(
            cost,
            fidelity=start["fidelity"],
            gamma_p=start["gamma_p"],
            a_onef=start["a_onef"],
            alpha=start["alpha"],
        )
        minuit.limits["fidelity"] = (0.0, 1.0)
        minuit.limits["gamma_p"] = (0.0, None)
        minuit.limits["a_onef"] = (0.0, None)
        minuit.limits["alpha"] = (0.0, None)
        if not fit_onef:
            minuit.values["a_onef"] = 0.0
            minuit.fixed["a_onef"] = True
        if not fit_alpha:
            minuit.values["alpha"] = alpha
            minuit.fixed["alpha"] = True
        minuit.migrad()
        minuit.hesse()
    except (RuntimeError, ValueError):
        return fit_results

    ndof = int(n_binned - n_free)
    chi2 = float(minuit.fval)
    reduced_chi2 = chi2 / ndof if ndof > 0 else np.nan
    # Rescale Hesse errors to the observed scatter unless the caller supplied true
    # uncertainties (absolute_sigma=True); mirrors scipy's absolute_sigma=False.
    if not absolute_sigma and ndof > 0 and np.isfinite(reduced_chi2) and reduced_chi2 > 0:
        err_scale = np.sqrt(reduced_chi2)
    else:
        err_scale = 1.0

    def _err(name: str, is_free: bool) -> float:
        return float(minuit.errors[name]) * err_scale if is_free else np.nan

    fidelity = float(minuit.values["fidelity"])
    gamma_p = float(minuit.values["gamma_p"])
    a_onef = float(minuit.values["a_onef"])
    alpha_fit = float(minuit.values["alpha"])
    gamma_p_err = _err("gamma_p", True)

    # Weighting-independent goodness-of-fit: RMS of the log10 data-vs-model residual
    # over the binned points, in decades. ~0.1 is a good fit; ~1 means the model is
    # roughly a decade off across the band (e.g. a flat fit to a sloped spectrum),
    # which reduced_chi2 can hide under uniform weighting. Use it to flag bad fits.
    model_b = parity_psd_model(f_b, fidelity, gamma_p, f_bw, a_onef=a_onef, alpha=alpha_fit)
    resid = logpsd_b - np.log10(np.maximum(model_b, tiny))
    resid_dex_rms = float(np.sqrt(np.mean(resid**2)))

    fit_results.update(
        fidelity=fidelity,
        fidelity_err=_err("fidelity", True),
        gamma_p=gamma_p,
        gamma_p_err=gamma_p_err,
        f_corner=gamma_p / np.pi,
        f_corner_err=gamma_p_err / np.pi,
        a_onef=a_onef,
        a_onef_err=_err("a_onef", fit_onef),
        alpha=alpha_fit,
        alpha_err=_err("alpha", fit_alpha),
        chi2=chi2,
        ndof=ndof,
        reduced_chi2=reduced_chi2,
        resid_dex_rms=resid_dex_rms,
        model=parity_psd_model(f, fidelity, gamma_p, f_bw, a_onef=a_onef, alpha=alpha_fit),
        minuit=minuit,
        success=bool(minuit.valid),
    )
    return fit_results


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


def averaged_psd_timestream(
    num_averages: int,
    lo_freq: float,
    if_freqs: FloatAny,
    df: float,
    pixel_counts: int,
    amp: FloatAny,
    output_port: int,
    input_port: int,
    is_usb: Optional[BoolAny] = None,
    sweeps: Optional[Sequence["Sweep"]] = None,
    welch: bool = False,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    discard_start_ms: float = 25.0,
    dither: bool = True,
    device: Optional[str] = None,
    filter: Optional[str] = None,
    notes: Optional[str] = None,
    external_trigger: bool = False,
    presto_address: Optional[str] = None,
    presto_port: Optional[int] = None,
    ext_ref_clk: bool = False,
    progress: bool = True,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    List["TimeStream"],
]:
    """Acquire many multi-tone time streams and return their averaged PSDs.

    This is a convenience wrapper around :class:`~daq.measurements.timestream.TimeStream`
    that repeats the common "take data, then show averaged PSD" workflow: it builds a
    multi-tone ``TimeStream`` with the given configuration, runs it ``num_averages``
    times, computes a per-tone PSD for every acquisition, and averages the PSDs across
    acquisitions (a running mean, so raw time streams are not all held in memory at
    once). Each :meth:`TimeStream.run` still writes its own HDF5 file and MongoDB
    document, exactly like running the measurement by hand.

    Two projections of each tone's complex signal are analysed:

    * When *sweeps* is provided (one fitted :class:`~daq.measurements.sweep.Sweep` per
      tone, aligned with *if_freqs*), each tone is projected into the resonator basis
      with :func:`from_elec_to_reson` and the two returned PSDs are the **dissipation**
      (radial, ``rad``) and **frequency** (arc-length, ``arc``) responses.
    * When *sweeps* is ``None``, no projection is available, so the two returned PSDs are
      simply the PSDs of the raw **I** (real) and **Q** (imaginary) quadratures.

    :param num_averages: Number of time-stream acquisitions to average over (e.g. ``100``).
    :param lo_freq: Local-oscillator frequency in Hz (see :class:`TimeStream`).
    :param if_freqs: IF frequency or frequencies in Hz; length sets the number of tones.
    :param df: Requested sample rate in Hz. The actual rate returned by the hardware
        ``tune`` is used for the PSD frequency axis.
    :param pixel_counts: Number of samples per acquisition.
    :param amp: DAC amplitude(s), one per tone or a scalar broadcast to all tones.
    :param output_port: DAC output port.
    :param input_port: ADC input port.
    :param is_usb: Per-tone USB/LSB selection passed through to :class:`TimeStream`.
    :param sweeps: Optional sequence of fitted :class:`Sweep` objects, one per tone in
        *if_freqs* order. When given, PSDs are computed in the resonator (dissipation /
        frequency) basis; otherwise the raw I / Q PSDs are returned.
    :param welch: Forwarded to :func:`compute_psd` (Welch's method when ``True``).
    :param nperseg: Forwarded to :func:`compute_psd` (Welch only).
    :param noverlap: Forwarded to :func:`compute_psd` (Welch only).
    :param window: Forwarded to :func:`compute_psd` (Welch only).
    :param detrend: Forwarded to :func:`compute_psd` (Welch only).
    :param discard_start_ms: Milliseconds of startup junk to discard from the start of
        every acquisition before analysis (default ``25.0``; set ``0`` to keep
        everything). This is forwarded to :class:`TimeStream`, which owns the trimming:
        each acquisition drops ``round(discard_start_ms * 1e-3 * fs)`` leading samples
        from its in-memory time-axis arrays (``signal``, ``usb``, ``lsb``, ``pixel_i``,
        ``pixel_q``) using the actual hardware sample rate, so both the PSD input and the
        returned :class:`TimeStream` objects reflect the same analysed window. The HDF5
        file saved by each ``run()`` retains the full, untrimmed acquisition.
    :param dither: Forwarded to :class:`TimeStream`.
    :param device: Forwarded to :class:`TimeStream` (used in the saved metadata).
    :param filter: Forwarded to :class:`TimeStream`.
    :param notes: Forwarded to :class:`TimeStream`.
    :param external_trigger: Forwarded to :class:`TimeStream`.
    :param presto_address: Forwarded to :meth:`TimeStream.run`.
    :param presto_port: Forwarded to :meth:`TimeStream.run`.
    :param ext_ref_clk: Forwarded to :meth:`TimeStream.run`.
    :param progress: When ``True`` and ``tqdm`` is installed, show a progress bar.
    :returns: ``(f, psd_a, psd_b, streams)`` where *f* is the PSD frequency axis in Hz,
        *psd_a* / *psd_b* are the averaged PSDs with shape ``(n_tones, n_freqs)``
        (dissipation / frequency when *sweeps* is given, else I / Q), and *streams* is
        the list of executed :class:`TimeStream` objects (each already saved).
    :raises ValueError: If *num_averages* < 1, *sweeps* length does not match the number
        of tones, or *discard_start_ms* is negative or would leave fewer than 2 samples
        (validated by :class:`TimeStream`).
    """
    from ..measurements.timestream import TimeStream

    if num_averages < 1:
        raise ValueError(f"num_averages must be >= 1, got {num_averages}")

    if_freqs = np.atleast_1d(np.asarray(if_freqs, dtype=np.float64))
    n_tones = if_freqs.shape[0]

    if sweeps is not None and len(sweeps) != n_tones:
        raise ValueError(f"sweeps must provide one Sweep per tone ({len(sweeps)} != {n_tones})")

    iterator = range(num_averages)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="averaged_psd_timestream")
        except ImportError:
            pass

    f: Optional[npt.NDArray[np.floating]] = None
    psd_a_sum: Optional[npt.NDArray[np.floating]] = None
    psd_b_sum: Optional[npt.NDArray[np.floating]] = None
    streams: List["TimeStream"] = []

    for _ in iterator:
        ts = TimeStream(
            lo_freq=lo_freq,
            if_freqs=if_freqs,
            df=df,
            pixel_counts=pixel_counts,
            amp=amp,
            output_port=output_port,
            input_port=input_port,
            is_usb=is_usb,
            dither=dither,
            device=device,
            filter=filter,
            notes=notes,
            external_trigger=external_trigger,
            discard_start_ms=discard_start_ms,
        )
        # run() saves the full acquisition and trims the leading junk from the
        # in-memory arrays, so ts.signal already reflects the analysed window.
        ts.run(
            presto_address=presto_address,
            presto_port=presto_port,
            ext_ref_clk=ext_ref_clk,
        )
        streams.append(ts)

        # Actual hardware sample rate (df is refined by tune during run()).
        fs = ts.df

        if sweeps is not None:
            rad_rows = []
            arc_rows = []
            for ch in range(n_tones):
                _, rad, arc = from_elec_to_reson(ts.signal[:, ch], sweeps[ch])
                rad_rows.append(rad)
                arc_rows.append(arc)
            a = np.asarray(rad_rows)
            b = np.asarray(arc_rows)
        else:
            # signal has shape (pixel_counts, n_tones); PSD wants tones on rows.
            a = np.real(ts.signal).T
            b = np.imag(ts.signal).T

        f, psd_a = compute_psd(
            a, fs, welch=welch, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend
        )
        _, psd_b = compute_psd(
            b, fs, welch=welch, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend
        )

        if psd_a_sum is None:
            psd_a_sum = psd_a
            psd_b_sum = psd_b
        else:
            psd_a_sum += psd_a
            psd_b_sum += psd_b

    psd_a_avg = psd_a_sum / num_averages
    psd_b_avg = psd_b_sum / num_averages

    return f, psd_a_avg, psd_b_avg, streams


def clean_correlated_streams(
    streams: Sequence["TimeStream"],
    signal_indices: Optional[Sequence[int]] = None,
    reference_indices: Optional[Sequence[int]] = None,
    min_t_s: Optional[float] = None,
    max_t_s: Optional[float] = None,
    return_coeffs: bool = False,
) -> (
    tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.floating]]
    | tuple[
        npt.NDArray[np.complexfloating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]
):
    """Batch-clean interleaved signal/reference time streams with :func:`remove_correlated_noise`.

    Designed for a list of :class:`~daq.measurements.timestream.TimeStream`
    acquisitions (e.g. the ``streams`` returned by :func:`averaged_psd_timestream`)
    whose tones are interleaved as ``[signal, reference, signal, reference, ...]``,
    where each *reference* tone is placed off resonance to track correlated
    electronics noise for its neighbouring on-resonance *signal* tone. For every
    stream and every (signal, reference) pair, the on-resonance tone is cleaned
    against its reference via :func:`remove_correlated_noise`, and only the cleaned
    signal tones are returned.

    By default the pairing is the interleaved layout above (signal = even tone
    indices ``0, 2, 4, ...``; reference = odd indices ``1, 3, 5, ...``). Pass
    *signal_indices* and *reference_indices* together to specify an arbitrary
    pairing (e.g. one shared reference reused for several signal tones).

    :param streams: Non-empty sequence of :class:`TimeStream` objects, all sharing
        the same tone layout and sample count. Each must have ``signal`` populated
        (shape ``(n_samples, n_tones)``).
    :param signal_indices: Tone indices to treat as on-resonance signals. Defaults to
        the even indices when *reference_indices* is also ``None``.
    :param reference_indices: Tone indices of the off-resonance reference paired with
        each entry of *signal_indices* (same length). Defaults to the odd indices.
    :param min_t_s: Forwarded to :func:`remove_correlated_noise`; start of the time
        window used to fit the cleaning coefficients (subtraction still applies to the
        full record).
    :param max_t_s: Forwarded to :func:`remove_correlated_noise`; end of that window.
    :param return_coeffs: When ``True``, also return the per-stream, per-pair cleaning
        coefficients ``x_r`` and ``x_rho``.
    :returns: ``(cleaned, freqs)`` where *cleaned* is the complex cleaned signal-tone
        data with shape ``(n_streams, n_samples, n_pairs)`` and *freqs* holds the
        physical frequencies (Hz) of the signal tones (shape ``(n_pairs,)``). When
        *return_coeffs* is ``True``, returns ``(cleaned, freqs, x_r, x_rho)`` with
        *x_r* / *x_rho* of shape ``(n_streams, n_pairs)``.
    :raises ValueError: If *streams* is empty; any stream's ``signal`` is unset, not
        2-D, or differs in shape from the first; the first stream's ``signal_freqs`` is
        unset; only one of *signal_indices* / *reference_indices* is given; the two index
        lists differ in length; an index is out of range; or the default pairing is
        requested with an odd number of tones.

    A :class:`UserWarning` is emitted if any signal tone is paired with itself as its own
    reference (which collapses that cleaned tone to ~zero).
    """
    streams = list(streams)
    if len(streams) == 0:
        raise ValueError("streams must be a non-empty sequence of TimeStream objects")

    # Validate every stream up front so failures name the offending index instead of
    # surfacing as a cryptic AttributeError/ValueError deep in the loop or np.stack.
    for i, ts in enumerate(streams):
        if getattr(ts, "signal", None) is None:
            raise ValueError(
                f"streams[{i}].signal is None; run the measurement (or load a file that "
                "stored signal data) before cleaning"
            )
        if ts.signal.ndim != 2:
            raise ValueError(
                f"streams[{i}].signal must be 2-D (n_samples, n_tones); got {ts.signal.ndim}-D"
            )
    n_samples, n_tones = streams[0].signal.shape
    for i, ts in enumerate(streams):
        if ts.signal.shape[0] != n_samples:
            raise ValueError(
                "all streams must have the same number of samples; "
                f"streams[0] has {n_samples} but streams[{i}] has {ts.signal.shape[0]}"
            )
        if ts.signal.shape[1] != n_tones:
            raise ValueError(
                "all streams must have the same number of tones; "
                f"streams[0] has {n_tones} but streams[{i}] has {ts.signal.shape[1]}"
            )
    if getattr(streams[0], "signal_freqs", None) is None:
        raise ValueError("streams[0].signal_freqs is None; cannot label the cleaned tones")

    if signal_indices is None and reference_indices is None:
        if n_tones % 2 != 0:
            raise ValueError(
                "Default interleaved pairing requires an even number of tones "
                f"([signal, reference, ...]); got {n_tones}. Pass signal_indices and "
                "reference_indices explicitly."
            )
        signal_indices = list(range(0, n_tones, 2))
        reference_indices = list(range(1, n_tones, 2))
    elif signal_indices is None or reference_indices is None:
        raise ValueError("provide both signal_indices and reference_indices, or neither")

    signal_indices = list(signal_indices)
    reference_indices = list(reference_indices)
    if len(signal_indices) != len(reference_indices):
        raise ValueError(
            "signal_indices and reference_indices must have the same length "
            f"({len(signal_indices)} != {len(reference_indices)})"
        )

    for name, idx in (("signal_indices", signal_indices), ("reference_indices", reference_indices)):
        out_of_range = [j for j in idx if j < 0 or j >= n_tones]
        if out_of_range:
            raise ValueError(
                f"{name} contains out-of-range tone indices {out_of_range}; "
                f"streams have {n_tones} tones (valid range 0..{n_tones - 1})"
            )

    self_paired = [si for si, ri in zip(signal_indices, reference_indices) if si == ri]
    if self_paired:
        warnings.warn(
            f"signal tone(s) {self_paired} are paired with themselves as their own "
            "reference; remove_correlated_noise then subtracts the signal from itself, "
            "collapsing the cleaned tone to ~zero. Check signal_indices/reference_indices.",
            stacklevel=2,
        )

    cleaned_all = []
    x_r_all = []
    x_rho_all = []
    for ts in streams:
        sig = ts.signal
        fs = ts.df
        cleaned_pairs = []
        x_r_pairs = []
        x_rho_pairs = []
        for si, ri in zip(signal_indices, reference_indices):
            cleaned, x_r, x_rho = remove_correlated_noise(
                sig[:, si], sig[:, ri], fs, min_t_s=min_t_s, max_t_s=max_t_s
            )
            cleaned_pairs.append(cleaned)
            x_r_pairs.append(x_r)
            x_rho_pairs.append(x_rho)
        cleaned_all.append(np.stack(cleaned_pairs, axis=-1))  # (n_samples, n_pairs)
        x_r_all.append(x_r_pairs)
        x_rho_all.append(x_rho_pairs)

    cleaned_arr = np.stack(cleaned_all, axis=0)  # (n_streams, n_samples, n_pairs)
    freqs = np.asarray(streams[0].signal_freqs)[signal_indices]

    if return_coeffs:
        return cleaned_arr, freqs, np.asarray(x_r_all), np.asarray(x_rho_all)
    return cleaned_arr, freqs


def averaged_psd_cleaned(
    cleaned: npt.NDArray[np.complexfloating],
    fs: float,
    sweeps: Optional[Sequence["Sweep"]] = None,
    welch: bool = False,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    detrend: str | bool = "constant",
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Average the PSDs of cleaned signal tones across acquisitions.

    This is the PSD stage that follows :func:`clean_correlated_streams`: given its
    ``cleaned`` output (the per-acquisition, per-signal-tone complex time series after
    correlated-noise removal), it computes a per-tone PSD for every acquisition and
    accumulates them into a running-mean average (only the PSD sum is held, not every
    acquisition's PSD). It mirrors :func:`averaged_psd_timestream`, so the two
    projections match:

    * When *sweeps* is provided (one fitted :class:`~daq.measurements.sweep.Sweep` per
      **signal** tone, aligned with the last axis of *cleaned*), each tone is projected
      into the resonator basis with :func:`from_elec_to_reson` and the two returned PSDs
      are the **dissipation** (``rad``) and **frequency** (``arc``) responses.
    * When *sweeps* is ``None``, the two returned PSDs are the PSDs of the raw **I**
      (real) and **Q** (imaginary) quadratures of the cleaned signal.

    :param cleaned: Complex cleaned signal data. Either the ``(n_streams, n_samples,
        n_signal_tones)`` array from :func:`clean_correlated_streams`, or a single
        ``(n_samples, n_signal_tones)`` acquisition (treated as one stream).
    :param fs: Sampling frequency in Hz (e.g. ``streams[0].df``). All acquisitions are
        assumed to share this rate.
    :param sweeps: Optional sequence of fitted :class:`Sweep` objects, one per signal
        tone. When given, PSDs are in the resonator (dissipation / frequency) basis;
        otherwise the raw I / Q PSDs are returned.
    :param welch: Forwarded to :func:`compute_psd` (Welch's method when ``True``).
    :param nperseg: Forwarded to :func:`compute_psd` (Welch only).
    :param noverlap: Forwarded to :func:`compute_psd` (Welch only).
    :param window: Forwarded to :func:`compute_psd` (Welch only).
    :param detrend: Forwarded to :func:`compute_psd` (Welch only).
    :returns: ``(f, psd_a, psd_b)`` where *f* is the PSD frequency axis in Hz and
        *psd_a* / *psd_b* are the averaged PSDs with shape
        ``(n_signal_tones, n_freqs)`` (dissipation / frequency when *sweeps* is given,
        else I / Q).
    :raises ValueError: If *cleaned* is not 2-D or 3-D, or *sweeps* length does not match
        the number of signal tones.
    """
    cleaned = np.asarray(cleaned)
    if cleaned.ndim == 2:
        cleaned = cleaned[np.newaxis, ...]
    if cleaned.ndim != 3:
        raise ValueError(
            "cleaned must be 2-D (n_samples, n_signal_tones) or 3-D "
            f"(n_streams, n_samples, n_signal_tones); got {cleaned.ndim}-D"
        )

    n_streams, _, n_signal = cleaned.shape
    if sweeps is not None and len(sweeps) != n_signal:
        raise ValueError(
            f"sweeps must provide one Sweep per signal tone ({len(sweeps)} != {n_signal})"
        )

    f: Optional[npt.NDArray[np.floating]] = None
    psd_a_sum: Optional[npt.NDArray[np.floating]] = None
    psd_b_sum: Optional[npt.NDArray[np.floating]] = None

    for t in range(n_streams):
        if sweeps is not None:
            rad_rows = []
            arc_rows = []
            for ch in range(n_signal):
                _, rad, arc = from_elec_to_reson(cleaned[t, :, ch], sweeps[ch])
                rad_rows.append(rad)
                arc_rows.append(arc)
            a = np.asarray(rad_rows)
            b = np.asarray(arc_rows)
        else:
            # cleaned[t] has shape (n_samples, n_signal); PSD wants tones on rows.
            a = cleaned[t].real.T
            b = cleaned[t].imag.T

        f, psd_a = compute_psd(
            a, fs, welch=welch, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend
        )
        _, psd_b = compute_psd(
            b, fs, welch=welch, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend
        )

        if psd_a_sum is None:
            psd_a_sum = psd_a
            psd_b_sum = psd_b
        else:
            psd_a_sum += psd_a
            psd_b_sum += psd_b

    return f, psd_a_sum / n_streams, psd_b_sum / n_streams
