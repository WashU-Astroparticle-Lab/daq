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
    discard_start_s: Optional[float] = None,
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
    :param discard_start_s: If given, discard this many seconds from the start of every
        acquisition before analysis (e.g. ``2e-4`` to drop the first 0.2 ms, which is
        often startup junk). The number of samples dropped is
        ``round(discard_start_s * fs)`` using the actual hardware sample rate. The cut is
        applied to both the PSD input and the time-axis arrays of the returned
        :class:`TimeStream` objects (``signal``, ``usb``, ``lsb``, ``pixel_i``,
        ``pixel_q``), so the in-memory objects match the analysed window. The HDF5 file
        saved by each ``run()`` retains the full, untrimmed acquisition.
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
        of tones, or *discard_start_s* is negative or would leave fewer than 2 samples.
    """
    from ..measurements.timestream import TimeStream

    if num_averages < 1:
        raise ValueError(f"num_averages must be >= 1, got {num_averages}")

    if_freqs = np.atleast_1d(np.asarray(if_freqs, dtype=np.float64))
    n_tones = if_freqs.shape[0]

    if sweeps is not None and len(sweeps) != n_tones:
        raise ValueError(f"sweeps must provide one Sweep per tone ({len(sweeps)} != {n_tones})")

    if discard_start_s is not None:
        if discard_start_s < 0:
            raise ValueError(f"discard_start_s must be non-negative, got {discard_start_s}")
        # Early sanity check against the requested rate (the tuned rate is nearly
        # identical) so we fail before running the hardware num_averages times.
        if int(round(discard_start_s * df)) >= pixel_counts - 1:
            raise ValueError(
                f"discard_start_s={discard_start_s} s discards ~all of the "
                f"{pixel_counts} samples at df={df} Hz; leaves fewer than 2 samples"
            )

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
        )
        ts.run(
            presto_address=presto_address,
            presto_port=presto_port,
            ext_ref_clk=ext_ref_clk,
        )
        streams.append(ts)

        # Actual hardware sample rate (df is refined by tune during run()).
        fs = ts.df

        # Drop the leading junk window from every time-axis array so both the PSDs
        # and the returned TimeStream objects reflect the same analysed window. The
        # saved HDF5 file already holds the full, untrimmed acquisition.
        if discard_start_s is not None:
            n_discard = int(round(discard_start_s * fs))
            if n_discard >= ts.signal.shape[0] - 1:
                raise ValueError(
                    f"discard_start_s={discard_start_s} s drops {n_discard} of "
                    f"{ts.signal.shape[0]} samples at fs={fs} Hz; leaves fewer than 2 samples"
                )
            if n_discard > 0:
                for attr in ("signal", "usb", "lsb", "pixel_i", "pixel_q"):
                    arr = getattr(ts, attr, None)
                    if arr is not None:
                        setattr(ts, attr, arr[n_discard:])

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
                f"streams[{i}].signal must be 2-D (n_samples, n_tones); got " f"{ts.signal.ndim}-D"
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
