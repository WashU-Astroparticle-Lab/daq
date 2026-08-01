# -*- coding: utf-8 -*-
"""Constant-gate-bias hunt for the largest parity contrast.

One ungated :class:`~daq.measurements.timestream.TimeStream` per candidate gate voltage, ranked
by the spread of the readout magnitude.
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt

from ..instruments import Agilent33220A
from ._gate_bias import GateBiasMeasurement
from .timestream import TimeStream

FloatAny = Union[float, List[float], npt.NDArray[np.floating]]


class BiasHunt(GateBiasMeasurement):
    """Find the gate bias with the largest charge-parity contrast.

    The gate is parked at each entry of :attr:`bias_voltages` in turn and a single-tone time
    stream of :attr:`ts_duration_s` is recorded at :attr:`readout_freq`. Charge-parity
    switching moves the resonator between two frequencies, so a two-level-telegraph response
    shows up as a spread in the readout magnitude: ``std(|signal|)`` is the scalar used to rank
    the candidates, and the winner is the operating point to take parity spectra at.

    Nothing here is gated. The bias is a DC level written over SCPI before each acquisition,
    so the Presto trigger stays deasserted throughout -- the opposite of
    :class:`~daq.measurements.qc_trace.QCTrace`, whose ramp only runs while the trigger is
    asserted.

    Like ``QCTrace``, this measurement does **not** locate the resonance; read out where a
    fitted :class:`~daq.measurements.sweep.Sweep` says to.

    The measurement is fully specified at construction: the bias voltages are drawn in
    :meth:`__init__`, so the object (and its saved record) pin down exactly what was measured
    regardless of the random seed. Pass *bias_voltages* explicitly -- e.g. a
    ``numpy.linspace`` -- to scan the range systematically instead of sampling it.

    Requires the Presto **and** the 33220A over VISA; it cannot run without hardware.

    :param readout_freq: Readout frequency in hertz, normally a fitted ``fr``.
    :param amp: Drive amplitude in DAC full scale. Convert from dBm with
        :func:`~daq.calibrations.power_dbm_to_amp`.
    :param output_port: Presto output port.
    :param input_port: Presto input port.
    :param v_min: Lower bound of the random bias draw in volts. Required when *bias_voltages*
        is not given, and **ignored** when it is -- the attribute then reports the span of the
        list you passed.
    :param v_max: Upper bound of the random bias draw in volts. Required when *bias_voltages*
        is not given, and **ignored** when it is.
    :param n_bias_try: Number of constant-bias tries to draw. Ignored when *bias_voltages* is
        given.
    :param bias_voltages: Explicit bias voltages to try, instead of a random draw.
    :param ts_duration_s: Length in seconds of each try, excluding the discarded start-up
        window.
    :param sampling_frequency: Time-stream sample rate in hertz.
    :param discard_start_ms: Leading milliseconds of start-up junk each time stream drops from
        its in-memory arrays. Passed through to
        :class:`~daq.measurements.timestream.TimeStream`.
    :param dither: Whether to dither the Presto output.
    :param seed: Seed for the random bias draw. The drawn voltages are saved either way, so
        this only matters for reproducing the draw itself.
    :param device: Device name, required for database logging.
    :param filter: Filter / amplifier chain description, for database logging.
    :param notes: Free-text note. Also prefixed onto each time stream's own note.
    :raises ValueError: If any parameter is out of range, or the draw bounds are missing.

    """

    def __init__(
        self,
        readout_freq: float,
        amp: float,
        output_port: int,
        input_port: int,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        n_bias_try: int = 20,
        bias_voltages: Optional[FloatAny] = None,
        ts_duration_s: float = 5.0,
        sampling_frequency: float = 5e4,
        discard_start_ms: float = 25.0,
        dither: bool = True,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._init_readout(
            readout_freq=readout_freq,
            amp=amp,
            output_port=output_port,
            input_port=input_port,
            sampling_frequency=sampling_frequency,
            discard_start_ms=discard_start_ms,
            dither=dither,
            device=device,
            filter=filter,
            notes=notes,
        )

        if ts_duration_s <= 0:
            raise ValueError(f"ts_duration_s must be positive, got {ts_duration_s}")
        self.ts_duration_s = ts_duration_s

        # Draw here rather than in run(), so the measurement is fully specified -- and its
        # saved record exactly reproducible -- before any hardware is touched.
        self._seed = seed
        if bias_voltages is None:
            # No default range: the usable gate span is a property of the device and of what
            # the generator is wired through, and guessing it would put an arbitrary voltage
            # on somebody's sample. Name the bounds, or name the voltages.
            if v_min is None or v_max is None:
                raise ValueError(
                    "Give the bounds of the random bias draw (v_min and v_max, in volts), or "
                    "pass bias_voltages explicitly (e.g. numpy.linspace(0, 2, 20)). There is "
                    "no default gate range -- it depends on the device and the wiring."
                )
            if v_max < v_min:
                raise ValueError(f"v_max={v_max} is below v_min={v_min}")
            if n_bias_try < 1:
                raise ValueError(f"n_bias_try must be at least 1, got {n_bias_try}")
            rng = np.random.default_rng(seed)
            self.bias_voltages = rng.uniform(v_min, v_max, n_bias_try)
            self.v_min = float(v_min)
            self.v_max = float(v_max)
        else:
            self.bias_voltages = np.atleast_1d(np.asarray(bias_voltages, dtype=np.float64))
            if self.bias_voltages.size < 1:
                raise ValueError("bias_voltages must hold at least one voltage")
            # Report the span actually tried, so the record describes the measurement whether
            # the voltages were drawn or handed in.
            self.v_min = float(np.min(self.bias_voltages))
            self.v_max = float(np.max(self.bias_voltages))
        self.n_bias_try = int(self.bias_voltages.size)

        # Results - replaced by run()
        self.parity_contrast = None
        """``std(|signal|)`` for each entry of :attr:`bias_voltages`."""
        self.best_bias = None
        """Bias voltage with the largest parity contrast."""
        self.best_contrast = None
        """The largest parity contrast found."""
        self.bias_files = None
        """HDF5 path of each try's time stream, in :attr:`bias_voltages` order."""
        self.best_bias_file = None
        """Path of the winning try's HDF5 file."""

        self.fit_results = None
        """Parity-model fit of :attr:`psd_avg`, from :meth:`fit_psd`.

        Not written to HDF5 or MongoDB -- ``Base`` skips ``fit_results`` by name, and this one
        holds a live ``iminuit`` object besides. Like
        :class:`~daq.measurements.sweep_power.SweepPower`'s per-amplitude fits it lives on the
        object only.
        """

        # Constituent acquisitions and the spectrum derived from them, kept off the saved
        # record (Base skips underscore-prefixed attributes) and exposed through read-only
        # properties. The spectrum is computed by analyze(), i.e. after run() has already
        # saved, so a public attribute here would only ever reach the file as None.
        self._bias_streams: List[TimeStream] = []
        self._psd_freqs = None
        self._psd_avg = None
        self._psd_fs = None
        self._psd_quantity = None
        self._psd_n_averaged = None

    def _reset_results(self) -> None:
        """Drop every result of a previous run, so a failed re-run leaves nothing stale.

        :attr:`parity_contrast` and :attr:`bias_streams` are read together -- by
        :attr:`best_bias_stream` and by :meth:`analyze` -- so they must never be left
        describing different runs.

        """
        self.parity_contrast = None
        self.best_bias = None
        self.best_contrast = None
        self.bias_files = None
        self.best_bias_file = None
        self._bias_streams = []
        # The spectrum and its fit are derived from the streams, so they go stale with them.
        self._psd_freqs = None
        self._psd_avg = None
        self._psd_fs = None
        self._psd_quantity = None
        self._psd_n_averaged = None
        self.fit_results = None

    # ------------------------------------------------------------------ constituent objects

    @property
    def bias_streams(self) -> List[TimeStream]:
        """The constant-bias time streams, in :attr:`bias_voltages` order.

        :returns: One time stream per try; empty before :meth:`run`.

        """
        return self._bias_streams

    @property
    def best_bias_stream(self) -> Optional[TimeStream]:
        """The time stream with the largest parity contrast.

        This is the stream to feed to the parity analysis --
        :func:`~daq.analysis.noise.compute_psd` and
        :func:`~daq.analysis.noise.fit_parity_psd`.

        :returns: The winning time stream, or ``None`` before :meth:`run`.

        """
        if not self._bias_streams or self.parity_contrast is None:
            return None
        return self._bias_streams[int(np.nanargmax(self.parity_contrast))]

    @property
    def psd_freqs(self) -> Optional[npt.NDArray[np.floating]]:
        """Frequency axis of :attr:`psd_avg` in hertz, or ``None`` before
        :meth:`average_psd`.

        :returns: The frequency axis.

        """
        return self._psd_freqs

    @property
    def psd_avg(self) -> Optional[npt.NDArray[np.floating]]:
        """Noise PSD averaged over the tries, or ``None`` before :meth:`average_psd`.

        :returns: The averaged PSD.

        """
        return self._psd_avg

    @property
    def psd_fs(self) -> Optional[float]:
        """Tuned sample rate :attr:`psd_avg` was computed at.

        This is the ``f_bw`` :meth:`fit_psd` holds fixed -- the *tuned* rate the hardware
        settled on, not the requested :attr:`sampling_frequency`.

        :returns: The sample rate in hertz, or ``None`` before :meth:`average_psd`.

        """
        return self._psd_fs

    @property
    def psd_quantity(self) -> Optional[str]:
        """Which projection of the readout :attr:`psd_avg` is the spectrum of.

        :returns: ``"abs"``, ``"real"`` or ``"imag"``; ``None`` before :meth:`average_psd`.

        """
        return self._psd_quantity

    @property
    def psd_n_averaged(self) -> Optional[int]:
        """How many tries went into :attr:`psd_avg`.

        :returns: The number of spectra averaged, or ``None`` before :meth:`average_psd`.

        """
        return self._psd_n_averaged

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def parity_contrast_of(stream: TimeStream) -> float:
        """Return the parity contrast of a time stream.

        :param stream: A run single-tone time stream.
        :returns: ``std(|signal|)`` over the trimmed record.

        """
        return float(np.std(np.abs(np.asarray(stream.signal)[:, 0])))

    @staticmethod
    def _parity_series(stream: TimeStream, quantity: str = "abs") -> npt.NDArray[np.floating]:
        """Return the real-valued series whose spectrum is the parity spectrum.

        :param stream: A run single-tone time stream.
        :param quantity: ``"abs"``, ``"real"`` or ``"imag"`` -- which projection of the
            complex readout to take. ``"abs"`` matches :meth:`parity_contrast_of`.
        :raises ValueError: If *quantity* is not one of the three.
        :returns: The mean-subtracted series, i.e. the fluctuation about the operating point.

        """
        signal = np.asarray(stream.signal)[:, 0]
        if quantity == "abs":
            series = np.abs(signal)
        elif quantity == "real":
            series = np.real(signal)
        elif quantity == "imag":
            series = np.imag(signal)
        else:
            raise ValueError(f"quantity must be 'abs', 'real' or 'imag', got {quantity!r}")
        # The parity signal is the *fluctuation*, not the operating point it sits on. For the
        # bare periodogram a constant offset lands entirely in the f=0 bin (which the fit drops
        # anyway), so this is cosmetic there -- but it is load-bearing on the Welch path when
        # the caller passes detrend=False.
        return series - series.mean()

    # ------------------------------------------------------------------ noise spectrum

    def average_psd(
        self,
        streams: Optional[Sequence[TimeStream]] = None,
        *,
        quantity: str = "abs",
        welch: bool = False,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Average the noise PSD over every constant-bias try.

        One PSD per try, averaged. Each try is short, so its own periodogram is noisy; the
        mean over :attr:`n_bias_try` of them beats that scatter down by ``sqrt(n_bias_try)``
        and leaves a spectrum worth fitting.

        **This averages across different operating points.** Each try sits at a different gate
        voltage, and the parity-switching rate is a property of the operating point, so a rate
        fitted to this average is an ensemble figure for the range scanned -- not the rate at
        any one bias. For the rate at the best point, run this on that stream alone::

            hunt.average_psd([hunt.best_bias_stream])

        The result is stored on :attr:`psd_freqs` / :attr:`psd_avg` and reused by
        :meth:`fit_psd` and :meth:`analyze`.

        :param streams: Streams to average over. Defaults to every try (:attr:`bias_streams`).
        :param quantity: Which projection of the complex readout to take the spectrum of --
            ``"abs"`` (default, matching the contrast metric), ``"real"`` or ``"imag"``.
        :param welch: Use Welch's method instead of the bare periodogram. Trades frequency
            resolution for variance, on top of the averaging across tries.
        :param nperseg: Welch segment length. Ignored unless *welch*.
        :param noverlap: Welch segment overlap. Ignored unless *welch*.
        :raises RuntimeError: If no streams are available -- :meth:`load` restores the derived
            record but not the raw acquisitions, so reload them from :attr:`bias_files` and
            pass them in.
        :raises ValueError: If the streams disagree on record length, so their frequency axes
            cannot be averaged together.
        :returns: ``(f, psd)`` -- frequencies in Hz and the averaged PSD.

        """
        from ..analysis.noise import compute_psd

        if streams is None:
            streams = self._bias_streams
        streams = [s for s in streams if s is not None]
        if not streams:
            raise RuntimeError(
                "No time streams to take a spectrum of. Run the measurement first, or -- if "
                "this measurement was loaded -- reload the acquisitions and pass them in: "
                "hunt.average_psd([TimeStream.load(p) for p in hunt.bias_files])."
            )

        lengths = {np.asarray(s.signal).shape[0] for s in streams}
        if len(lengths) > 1:
            raise ValueError(
                f"The tries hold different numbers of samples ({sorted(lengths)}), so their "
                "PSDs sit on different frequency axes and cannot be averaged. Fit them "
                "individually instead."
            )

        # The tuned rate, not the requested sampling_frequency: run() lets the hardware refine
        # df, and the frequency axis (and the f_bw the fit holds fixed) follow the tuned value.
        rates = {float(s.df) for s in streams}
        if len(rates) > 1:
            warnings.warn(
                f"The tries were taken at different tuned sample rates ({sorted(rates)} Hz), "
                "so their frequency axes do not line up exactly. Averaging them anyway on the "
                "first stream's axis; the spread is normally a tuning artefact of order 1 Hz.",
                stacklevel=2,
            )
        self._psd_fs = float(streams[0].df)

        psd_sum = None
        for stream in streams:
            f, psd = compute_psd(
                self._parity_series(stream, quantity),
                self._psd_fs,
                welch=welch,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            psd_sum = psd if psd_sum is None else psd_sum + psd

        self._psd_freqs = f
        self._psd_avg = psd_sum / len(streams)
        self._psd_quantity = quantity
        self._psd_n_averaged = len(streams)
        # A spectrum this fit did not see is not a spectrum this fit describes.
        self.fit_results = None
        return self._psd_freqs, self._psd_avg

    def fit_psd(self, **kwargs) -> dict:
        """Fit the averaged PSD to the random-telegraph parity model.

        Runs :meth:`average_psd` first when it has not been run. The sampling bandwidth
        ``f_bw`` is held fixed at the tuned sample rate the streams were taken at, as
        :func:`~daq.analysis.noise.fit_parity_psd` expects.

        The fitted ``gamma_p`` is an ensemble rate over the gate range scanned, for the reason
        given in :meth:`average_psd`. Check ``resid_dex_rms`` before believing any of it --
        ``~0.1`` is a good fit, ``~1`` means the model is a decade off across the band.

        The result lives on the object only (as :attr:`fit_results`); like
        :class:`~daq.measurements.sweep_power.SweepPower`'s per-amplitude fits it is **not**
        written to HDF5 or MongoDB, since it is derived from streams the saved record does not
        contain and is cheap to recompute.

        :param kwargs: Passed through to :func:`~daq.analysis.noise.fit_parity_psd` --
            ``fit_onef=True`` for a ``1/f``-like low-frequency rise, ``n_bins``,
            ``bin_weighting``, and so on.
        :raises RuntimeError: If no streams are available; see :meth:`average_psd`.
        :returns: The ``fit_results`` dict from
            :func:`~daq.analysis.noise.fit_parity_psd`.

        """
        from ..analysis.noise import fit_parity_psd

        if self._psd_avg is None:
            self.average_psd()

        self.fit_results = fit_parity_psd(
            self._psd_freqs, self._psd_avg, f_bw=self._psd_fs, **kwargs
        )
        return self.fit_results

    # ------------------------------------------------------------------ acquisition

    def run(
        self,
        bias: Optional[Agilent33220A] = None,
        *,
        presto_address: Optional[str] = None,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
        save_filename: Optional[str] = None,
    ) -> str:
        """Acquire one time stream per candidate bias and save the derived record.

        Every try saves its own HDF5 + MongoDB record through the normal path with
        ``attach(bias=...)`` applied, so each raw acquisition stays individually loadable. This
        measurement saves one further record holding the contrast-versus-bias curve, the
        winning bias and the constituent file paths.

        The gate-bias generator's output is forced off when the hunt finishes -- including on
        exception -- so no bias is left on the device. When *bias* is omitted the generator is
        opened and closed here; when it is passed in the caller keeps ownership of the VISA
        session and only the output is de-energised.

        As with :class:`~daq.measurements.qc_trace.QCTrace`, this ``run`` takes the *bias
        generator* as its first (and only) positional argument, not ``presto_address``; the
        Presto connection parameters are keyword-only.

        :param bias: An open :class:`~daq.instruments.function_generator.Agilent33220A`. When
            ``None``, one is discovered and opened for the duration of the run.
        :param presto_address: Presto address. Defaults to ``DAQ_PRESTO_ADDRESS``.
        :param presto_port: Presto port. Defaults to the presto default.
        :param ext_ref_clk: Whether to use an external reference clock.
        :param save_filename: Explicit path for this measurement's own HDF5 file.
        :returns: Path of this measurement's HDF5 file.

        """
        run_kwargs: Dict[str, Any] = dict(
            presto_address=presto_address,
            presto_port=presto_port,
            ext_ref_clk=ext_ref_clk,
        )

        # Clear every result up front, not just the streams. A try that raises part-way through
        # a *re-run* would otherwise leave the previous run's parity_contrast beside this run's
        # shorter _bias_streams, and best_bias_stream indexes one by the argmax of the other --
        # an IndexError at best, and at worst a new stream silently reported with an old
        # contrast and an old bias voltage.
        self._reset_results()

        with ExitStack() as stack:
            if bias is None:
                bias = stack.enter_context(Agilent33220A())
            else:
                # The caller owns the session, but never leave a bias on the gate.
                stack.callback(setattr, bias, "output", False)

            print(f"Bias hunt: reading out at {self.readout_freq / 1e9:.6f} GHz")

            ts_samples = self._stream_samples(self.ts_duration_s)
            contrasts = np.full(self.n_bias_try, np.nan)
            paths: List[str] = []
            for ii, voltage in enumerate(self.bias_voltages):
                bias.constant(float(voltage))
                stream = self._make_timestream(
                    ts_samples,
                    # Ungated: the bias is a DC level already written over SCPI, so there is
                    # nothing for a trigger to start.
                    external_trigger=False,
                    notes=f"Constant bias {voltage:.4f} V (try {ii + 1}/{self.n_bias_try})",
                )
                stream.attach(bias=bias)
                paths.append(stream.run(**run_kwargs))

                contrasts[ii] = self.parity_contrast_of(stream)
                self._bias_streams.append(stream)
                print(
                    f"Bias try {ii + 1}/{self.n_bias_try}: {voltage:.4f} V, "
                    f"std(|signal|) = {contrasts[ii]:.4e}"
                )

            self.parity_contrast = contrasts
            self.bias_files = paths
            best = int(np.nanargmax(contrasts))
            self.best_bias = float(self.bias_voltages[best])
            self.best_contrast = float(contrasts[best])
            self.best_bias_file = paths[best]
            print(
                f"Max parity contrast at bias {self.best_bias:.4f} V, "
                f"std(|signal|) = {self.best_contrast:.4e}"
            )

        # Saved after the bias is de-energised, so a failure here cannot leave it applied.
        return self.save(save_filename=save_filename)

    def save(self, save_filename: Optional[str] = None) -> str:
        """Write this measurement's HDF5 file and MongoDB record.

        :param save_filename: Explicit path. Generated under ``DAQ_DATA_FOLDER`` when ``None``.
        :returns: Path of the written file.

        """
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "BiasHunt":
        """Rebuild a measurement from its saved HDF5 file.

        The contrast-versus-bias curve, the winner and the constituent file paths are restored.
        The time streams themselves are not: load them from :attr:`bias_files` with
        :meth:`TimeStream.load <daq.measurements.timestream.TimeStream.load>` when you need the
        raw records.

        :param load_filename: Path of the HDF5 file to load.
        :returns: The reconstructed measurement.

        """
        with h5py.File(load_filename, "r") as h5f:
            attrs = h5f.attrs

            self = cls(
                readout_freq=float(attrs["readout_freq"]),  # type: ignore
                amp=float(attrs["amp"]),  # type: ignore
                output_port=int(attrs["output_port"]),  # type: ignore
                input_port=int(attrs["input_port"]),  # type: ignore
                bias_voltages=h5f["bias_voltages"][()],  # type: ignore
                ts_duration_s=float(attrs["ts_duration_s"]),  # type: ignore
                sampling_frequency=float(attrs["sampling_frequency"]),  # type: ignore
                discard_start_ms=float(attrs["discard_start_ms"]),  # type: ignore
                dither=bool(attrs["dither"]),  # type: ignore
                device=attrs.get("device", None),
                filter=attrs.get("filter", None),
                notes=attrs.get("notes", None),
            )
            # Restored rather than recomputed from bias_voltages: when the voltages were drawn,
            # these are the bounds they were drawn from, which the extremes only approximate.
            self.v_min = float(attrs["v_min"])  # type: ignore
            self.v_max = float(attrs["v_max"])  # type: ignore

            self.best_bias = float(attrs["best_bias"]) if "best_bias" in attrs else None
            self.best_contrast = float(attrs["best_contrast"]) if "best_contrast" in attrs else None
            self.best_bias_file = attrs.get("best_bias_file", None)
            if "parity_contrast" in h5f:
                self.parity_contrast = h5f["parity_contrast"][()]  # type: ignore
            if "bias_files" in h5f:
                # h5py hands back bytes for a variable-length string dataset.
                self.bias_files = [
                    path.decode() if isinstance(path, bytes) else str(path)
                    for path in h5f["bias_files"][()]  # type: ignore
                ]

        return self

    # ------------------------------------------------------------------ analysis

    def analyze(
        self,
        psd: bool = True,
        fit: bool = True,
        title: Optional[str] = None,
        **fit_kwargs,
    ):
        """Plot the bias hunt, and the averaged noise spectrum with its parity fit.

        The upper panel is the search result: parity contrast against gate bias, sorted by
        voltage so a random draw reads as a curve, with the winner marked. The lower panel is
        the PSD averaged over every try (:meth:`average_psd`) on log-log axes, overlaid with
        the random-telegraph fit (:meth:`fit_psd`) and annotated with the fitted rate, corner
        and fidelity.

        Both derived products are computed lazily here if they have not been already, and
        reused if they have -- so ``analyze()`` after an explicit ``fit_psd(fit_onef=True)``
        draws that fit rather than refitting with the defaults.

        The spectrum panel needs the raw acquisitions, which :meth:`load` does not restore. On
        a loaded measurement it is skipped with a note saying how to get it back, rather than
        raising -- the contrast curve is still worth seeing.

        :param psd: Draw the averaged-spectrum panel. ``False`` gives the contrast curve alone.
        :param fit: Overlay the parity-model fit on the spectrum. Ignored when *psd* is
            ``False``. A fit that raises is reported on the panel instead of aborting the plot.
        :param title: Figure title. Defaults to naming the device and the readout frequency.
        :param fit_kwargs: Passed to :meth:`fit_psd` when the fit is computed here (e.g.
            ``fit_onef=True``). Ignored if a fit already exists.
        :raises RuntimeError: If the measurement has not been run or loaded.
        :returns: The created figure.

        """
        if self.parity_contrast is None:
            raise RuntimeError("No bias hunt available. Run or load the measurement first.")

        import matplotlib.pyplot as plt

        show_psd = psd and bool(self._bias_streams)
        if psd and not show_psd:
            print(
                "INFO: no raw acquisitions on this object, so the spectrum panel is skipped. "
                "Reload them and pass them in to get it: "
                "hunt.average_psd([TimeStream.load(p) for p in hunt.bias_files])."
            )

        if show_psd:
            fig, (ax, ax_psd) = plt.subplots(2, 1, figsize=(8, 8), tight_layout=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
            ax_psd = None

        # Sort by voltage so a random draw still reads as a curve rather than a zigzag.
        order = np.argsort(self.bias_voltages)
        ax.plot(
            np.asarray(self.bias_voltages)[order],
            np.asarray(self.parity_contrast)[order],
            ".-",
            color="tab:green",
        )
        if self.best_bias is not None:
            ax.axvline(
                self.best_bias,
                color="tab:red",
                ls="--",
                lw=1.0,
                label=f"best bias {self.best_bias:.4f} V",
            )
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("Gate bias [V]")
        ax.set_ylabel(r"Parity contrast std($|S|$) [FS]")
        ax.grid(True, alpha=0.3)

        if title is None:
            parts = ["Bias hunt"]
            if self.device is not None:
                parts.append(str(self.device))
            parts.append(f"{self.readout_freq / 1e9:.6f} GHz")
            title = " -- ".join(parts)
        if show_psd:
            fig.suptitle(title)
        else:
            ax.set_title(title)

        if show_psd:
            self._draw_psd(ax_psd, fit=fit, fit_kwargs=fit_kwargs)

        plt.show()
        return fig

    def _draw_psd(self, ax, *, fit: bool, fit_kwargs: dict) -> None:
        """Draw the averaged spectrum and, optionally, its parity fit.

        :param ax: Axis to draw into.
        :param fit: Whether to overlay the parity-model fit.
        :param fit_kwargs: Extra arguments for :meth:`fit_psd`, used only if no fit exists.

        """
        if self._psd_avg is None:
            self.average_psd()

        # The DC bin is not plottable on a log frequency axis, and carries no parity
        # information -- it is the operating point, which _parity_series already removed.
        keep = np.asarray(self._psd_freqs) > 0
        n = self._psd_n_averaged
        ax.loglog(
            np.asarray(self._psd_freqs)[keep],
            np.asarray(self._psd_avg)[keep],
            lw=0.5,
            color="tab:blue",
            alpha=0.25,
            label=f"periodogram, mean of {n} {'try' if n == 1 else 'tries'}",
        )

        if fit:
            try:
                if self.fit_results is None:
                    self.fit_psd(**fit_kwargs)
                res = self.fit_results
                # The log-binned points are what the fit actually saw. Without them the panel
                # shows a smooth model over a periodogram scattering across ten decades, which
                # reads as a bad fit even when it is a good one.
                ax.loglog(
                    res["f_binned"],
                    res["psd_binned"],
                    "o",
                    ms=3.5,
                    color="tab:blue",
                    label=f"log-binned ({res['n_bins']} bins, fit to these)",
                )
                ax.loglog(
                    np.asarray(self._psd_freqs)[keep],
                    np.asarray(res["model"])[keep],
                    color="tab:red",
                    lw=1.8,
                    label=(
                        rf"$\Gamma_p$ = {res['gamma_p']:.3g} $\pm$ {res['gamma_p_err']:.2g} Hz"
                        "\n"
                        rf"$f_c$ = {res['f_corner']:.3g} Hz,  $F$ = {res['fidelity']:.3f}"
                        "\n"
                        rf"resid = {res['resid_dex_rms']:.2f} dex"
                    ),
                )
                ax.axvline(res["f_corner"], color="tab:red", ls=":", lw=1.0)
                # Frame on the binned points: the periodogram's low-power outliers drag the
                # y-range over ten decades and flatten everything of interest into a line.
                lo = float(np.min(res["psd_binned"]))
                hi = float(np.max(res["psd_binned"]))
                ax.set_ylim(lo / 30.0, hi * 30.0)
            except Exception as err:
                # A failed fit must not cost the spectrum: the data is the point, the model is
                # an overlay, and a spectrum this model does not describe is itself a result.
                print(f"WARN: the parity fit failed, showing the spectrum alone: {err}")
                ax.set_title("parity fit failed", fontsize=8, color="tab:red")

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(rf"PSD of ${self.psd_quantity}(S)$ [FS$^2$/Hz]")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)
