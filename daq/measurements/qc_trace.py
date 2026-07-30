# -*- coding: utf-8 -*-
"""Quantum-capacitance (QC) trace measurement.

The bench routine for charge-parity / quasiparticle-tunnelling work on a single device,
composing a :class:`~daq.measurements.sweep.Sweep`, several
:class:`~daq.measurements.timestream.TimeStream` acquisitions and an
:class:`~daq.instruments.function_generator.Agilent33220A` gate-bias source.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from .._base import Base
from ..analysis.folding import fold_timestream
from ..instruments import Agilent33220A
from .sweep import Sweep
from .timestream import TimeStream

FloatAny = Union[float, List[float], npt.NDArray[np.floating]]


class QCTrace(Base):
    """Quantum-capacitance trace: the full parity / quasiparticle-tunnelling routine.

    Runs, in order, the four steps that make up a QC-trace measurement on one device:

    1. A frequency :class:`~daq.measurements.sweep.Sweep` with auto-fit, to locate ``fr``.
       Everything downstream reads out at that frequency.
    2. The **QC trace** itself: a gated voltage ramp on the gate, an externally-triggered
       :class:`~daq.measurements.timestream.TimeStream` spanning ``num_periods`` whole ramp
       periods, block-averaged into a single period by
       :func:`~daq.analysis.folding.fold_timestream`. Uncorrelated noise falls by
       ``sqrt(num_periods)``, leaving the device's response to one sweep of the gate voltage.
    3. A **bias hunt**: one constant-bias time stream per entry of :attr:`bias_voltages`
       (drawn at random over the ramp's voltage range by default), keeping the one with the
       largest parity contrast, ``std(|signal|)``.
    4. One time stream under a **free-running ramp**, the same length as the constant-bias
       tries, for comparison against the best static bias.

    Every step saves its own HDF5 file and MongoDB record through the usual path, so the raw
    sweep and each individual time stream stay individually loadable. This class saves one
    further record holding the derived products -- the fitted ``fr``, the folded QC trace, the
    contrast-versus-bias curve and the winning bias -- plus the file paths of the constituent
    acquisitions.

    The measurement is fully specified at construction: the bias voltages are drawn in
    :meth:`__init__`, so the object (and its saved record) pin down exactly what was measured
    regardless of the random seed.

    Requires the Presto **and** the 33220A over VISA; it cannot run without hardware.

    :param freq_center: Centre frequency of the locating sweep in hertz.
    :param amp: Drive amplitude in DAC full scale, shared by the sweep and every time stream.
        Convert from dBm with :func:`~daq.calibrations.power_dbm_to_amp`.
    :param output_port: Presto output port.
    :param input_port: Presto input port.
    :param ramp_vpp: Ramp peak-to-peak amplitude in volts.
    :param ramp_freq_hz: Ramp repetition frequency in hertz.
    :param ramp_offset_v: Ramp DC offset in volts. Defaults to ``ramp_vpp / 2``, making the
        ramp unipolar-positive (the lab convention).
    :param ramp_symmetry_pct: Ramp symmetry in percent; ``100`` gives a ramp-up sawtooth.
    :param sampling_frequency: Time-stream sample rate in hertz.
    :param num_periods: Number of whole ramp periods the QC trace spans and averages over.
    :param n_bias_try: Number of constant-bias tries in the bias hunt. Ignored when
        *bias_voltages* is given.
    :param bias_voltages: Explicit bias voltages to try, instead of a random draw -- e.g. a
        ``numpy.linspace`` to scan the range systematically.
    :param v_min: Lower bound of the random bias draw in volts. Defaults to the ramp's minimum
        voltage, ``ramp_offset_v - ramp_vpp / 2``.
    :param v_max: Upper bound of the random bias draw in volts. Defaults to the ramp's maximum
        voltage, ``ramp_offset_v + ramp_vpp / 2``.
    :param ts_duration_s: Length in seconds of each constant-bias try and of the free-running
        ramp stream, excluding the discarded start-up window.
    :param freq_span: Span of the locating sweep in hertz.
    :param sweep_df: Frequency step of the locating sweep in hertz.
    :param sweep_num_averages: Averages per point in the locating sweep.
    :param discard_start_ms: Leading milliseconds of start-up junk each time stream drops from
        its in-memory arrays. Passed through to
        :class:`~daq.measurements.timestream.TimeStream` and accounted for in the QC trace's
        sample count.
    :param dither: Whether to dither the Presto output.
    :param seed: Seed for the random bias draw. The drawn voltages are saved either way, so
        this only matters for reproducing the draw itself.
    :param device: Device name, required for database logging.
    :param filter: Filter / amplifier chain description, for database logging.
    :param notes: Free-text note. Also prefixed onto each sub-measurement's own note.
    :raises ValueError: If any parameter is out of range.

    """

    def __init__(
        self,
        freq_center: float,
        amp: float,
        output_port: int,
        input_port: int,
        ramp_vpp: float = 2.0,
        ramp_freq_hz: float = 500.0,
        ramp_offset_v: Optional[float] = None,
        ramp_symmetry_pct: float = 100.0,
        sampling_frequency: float = 5e4,
        num_periods: int = 200,
        n_bias_try: int = 20,
        bias_voltages: Optional[FloatAny] = None,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        ts_duration_s: float = 5.0,
        freq_span: float = 0.7e6,
        sweep_df: float = 5e3,
        sweep_num_averages: int = 50,
        discard_start_ms: float = 25.0,
        dither: bool = True,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        if not 0.0 < amp < 1.0:
            raise ValueError(f"amp must be between 0 and 1 (DAC full scale), got {amp}")
        if freq_span <= 0:
            raise ValueError(f"freq_span must be positive, got {freq_span}")
        if sweep_df <= 0:
            raise ValueError(f"sweep_df must be positive, got {sweep_df}")
        if sweep_num_averages < 1:
            raise ValueError(f"sweep_num_averages must be at least 1, got {sweep_num_averages}")
        if ramp_vpp <= 0:
            raise ValueError(f"ramp_vpp must be positive, got {ramp_vpp}")
        if ramp_freq_hz <= 0:
            raise ValueError(f"ramp_freq_hz must be positive, got {ramp_freq_hz}")
        if not 0.0 <= ramp_symmetry_pct <= 100.0:
            raise ValueError(
                f"ramp_symmetry_pct must be between 0 and 100, got {ramp_symmetry_pct}"
            )
        if sampling_frequency <= 0:
            raise ValueError(f"sampling_frequency must be positive, got {sampling_frequency}")
        if num_periods < 1:
            raise ValueError(f"num_periods must be at least 1, got {num_periods}")
        if ts_duration_s <= 0:
            raise ValueError(f"ts_duration_s must be positive, got {ts_duration_s}")
        if sampling_frequency < ramp_freq_hz:
            raise ValueError(
                f"sampling_frequency={sampling_frequency} Hz gives fewer than one sample per "
                f"{ramp_freq_hz} Hz ramp period; raise the sample rate or slow the ramp."
            )

        self.freq_center = freq_center
        self.freq_span = freq_span
        self.sweep_df = sweep_df
        self.sweep_num_averages = sweep_num_averages
        self.amp = amp
        self.output_port = output_port
        self.input_port = input_port
        self.dither = dither

        self.ramp_vpp = ramp_vpp
        self.ramp_freq_hz = ramp_freq_hz
        self.ramp_offset_v = ramp_vpp / 2.0 if ramp_offset_v is None else ramp_offset_v
        self.ramp_symmetry_pct = ramp_symmetry_pct

        self.sampling_frequency = sampling_frequency
        self.num_periods = num_periods
        self.ts_duration_s = ts_duration_s
        self.discard_start_ms = float(discard_start_ms)

        # The random draw defaults to the span the ramp itself covers, so the hunt looks for
        # the operating point inside the range the QC trace swept through.
        self.v_min = self.ramp_offset_v - ramp_vpp / 2.0 if v_min is None else v_min
        self.v_max = self.ramp_offset_v + ramp_vpp / 2.0 if v_max is None else v_max
        if self.v_max < self.v_min:
            raise ValueError(f"v_max={self.v_max} is below v_min={self.v_min}")

        # Draw here rather than in run(), so the measurement is fully specified -- and its
        # saved record exactly reproducible -- before any hardware is touched.
        self._seed = seed
        if bias_voltages is None:
            if n_bias_try < 1:
                raise ValueError(f"n_bias_try must be at least 1, got {n_bias_try}")
            rng = np.random.default_rng(seed)
            self.bias_voltages = rng.uniform(self.v_min, self.v_max, n_bias_try)
        else:
            self.bias_voltages = np.atleast_1d(np.asarray(bias_voltages, dtype=np.float64))
            if self.bias_voltages.size < 1:
                raise ValueError("bias_voltages must hold at least one voltage")
        self.n_bias_try = int(self.bias_voltages.size)

        self.device = device
        self.filter = filter
        self.notes = notes

        # Results - replaced by run()
        self.fr = None
        """Fitted resonant frequency in hertz, read out by every time stream."""
        self.fr_err = None
        """Fit uncertainty on :attr:`fr` in hertz."""
        self.fit_results = None
        """``resonator_tools`` fit dict from the locating sweep (not written to HDF5)."""
        self.time_ms = None
        """Time axis of one ramp period in milliseconds."""
        self.avg_iq = None
        """Block-averaged QC trace, shape ``(2, n_samples)``; row 0 is I, row 1 is Q."""
        self.parity_contrast = None
        """``std(|signal|)`` for each entry of :attr:`bias_voltages`."""
        self.best_bias = None
        """Bias voltage with the largest parity contrast."""
        self.best_contrast = None
        """The largest parity contrast found."""
        self.sweep_file = None
        """Path of the locating sweep's HDF5 file."""
        self.qc_file = None
        """Path of the gated-ramp QC-trace time stream's HDF5 file."""
        self.best_bias_file = None
        """Path of the winning constant-bias time stream's HDF5 file."""
        self.ramp_file = None
        """Path of the free-running-ramp time stream's HDF5 file."""

        # Constituent measurement objects, kept off the saved record (Base skips
        # underscore-prefixed attributes) and exposed through read-only properties.
        self._sweep: Optional[Sweep] = None
        self._qc_stream: Optional[TimeStream] = None
        self._bias_streams: List[TimeStream] = []
        self._ramp_stream: Optional[TimeStream] = None

    # ------------------------------------------------------------------ constituent objects

    @property
    def sweep(self) -> Optional[Sweep]:
        """The locating :class:`~daq.measurements.sweep.Sweep`, or ``None`` before
        :meth:`run`.

        :returns: The sweep measurement.

        """
        return self._sweep

    @property
    def qc_stream(self) -> Optional[TimeStream]:
        """The gated-ramp time stream the QC trace was folded from.

        :returns: The time stream, or ``None`` before :meth:`run`.

        """
        return self._qc_stream

    @property
    def bias_streams(self) -> List[TimeStream]:
        """The constant-bias time streams, in :attr:`bias_voltages` order.

        :returns: One time stream per bias try; empty before :meth:`run`.

        """
        return self._bias_streams

    @property
    def best_bias_stream(self) -> Optional[TimeStream]:
        """The constant-bias time stream with the largest parity contrast.

        This is the stream to feed to the parity analysis --
        :func:`~daq.analysis.noise.compute_psd` and
        :func:`~daq.analysis.noise.fit_parity_psd`.

        :returns: The winning time stream, or ``None`` before :meth:`run`.

        """
        if not self._bias_streams or self.parity_contrast is None:
            return None
        return self._bias_streams[int(np.nanargmax(self.parity_contrast))]

    @property
    def ramp_stream(self) -> Optional[TimeStream]:
        """The time stream taken under a free-running ramp.

        :returns: The time stream, or ``None`` before :meth:`run`.

        """
        return self._ramp_stream

    # ------------------------------------------------------------------ helpers

    def _notes(self, step: str) -> str:
        """Compose a sub-measurement note, keeping this measurement's own note as a prefix.

        :param step: Description of the step the sub-measurement belongs to.
        :returns: The note to hand to the sub-measurement.

        """
        return step if self.notes is None else f"{self.notes} -- {step}"

    def _make_timestream(
        self,
        pixel_counts: int,
        *,
        external_trigger: bool,
        notes: str,
    ) -> TimeStream:
        """Build a single-tone time stream on resonance, shared by every acquisition step.

        :param pixel_counts: Number of samples to acquire, including the discarded start.
        :param external_trigger: Whether the Presto asserts its trigger output, which gates
            the ramp when the generator is in gated-burst mode.
        :param notes: Step description for the sub-measurement's note.
        :returns: The configured time stream.

        """
        return TimeStream(
            lo_freq=self.fr,
            if_freqs=[0.0],
            df=self.sampling_frequency,
            pixel_counts=pixel_counts,
            amp=self.amp,
            output_port=self.output_port,
            input_port=self.input_port,
            dither=self.dither,
            device=self.device,
            filter=self.filter,
            notes=self._notes(notes),
            external_trigger=external_trigger,
            discard_start_ms=self.discard_start_ms,
        )

    def _stream_samples(self) -> int:
        """Return the sample count of a fixed-duration stream, including the discarded start.

        :returns: Number of samples for the constant-bias tries and the ramp stream.

        """
        n_discard = int(round(self.discard_start_ms * 1e-3 * self.sampling_frequency))
        return n_discard + int(round(self.ts_duration_s * self.sampling_frequency))

    @staticmethod
    def _parity_contrast(stream: TimeStream) -> float:
        """Return the parity contrast of a time stream.

        Charge-parity switching moves the resonator between two frequencies, so a
        two-level-telegraph response shows up as a spread in the readout magnitude. The
        standard deviation of ``|signal|`` is the cheap scalar proxy used to rank candidate
        gate biases.

        :param stream: A run single-tone time stream.
        :returns: ``std(|signal|)`` over the trimmed record.

        """
        return float(np.std(np.abs(np.asarray(stream.signal)[:, 0])))

    # ------------------------------------------------------------------ acquisition

    def run(
        self,
        bias: Optional[Agilent33220A] = None,
        presto_address: Optional[str] = None,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
        save_filename: Optional[str] = None,
    ) -> str:
        """Run the four-step QC-trace sequence and save the derived record.

        The gate-bias generator's output is forced off before the locating sweep and again
        when the sequence finishes -- including on exception -- so no bias is left on the
        device. When *bias* is omitted the generator is opened and closed here; when it is
        passed in the caller keeps ownership of the VISA session and only the output is
        de-energised.

        :param bias: An open :class:`~daq.instruments.function_generator.Agilent33220A`. When
            ``None``, one is discovered and opened for the duration of the run.
        :param presto_address: Presto address. Defaults to ``DAQ_PRESTO_ADDRESS``.
        :param presto_port: Presto port. Defaults to the presto default.
        :param ext_ref_clk: Whether to use an external reference clock.
        :param save_filename: Explicit path for this measurement's own HDF5 file.
        :raises RuntimeError: If the locating sweep does not yield a usable ``fr``; the sweep's
            own file is still saved, and its path is given in the message.
        :returns: Path of this measurement's HDF5 file.

        """
        run_kwargs: Dict[str, Any] = dict(
            presto_address=presto_address,
            presto_port=presto_port,
            ext_ref_clk=ext_ref_clk,
        )

        with ExitStack() as stack:
            if bias is None:
                bias = stack.enter_context(Agilent33220A())
            else:
                # The caller owns the session, but never leave a bias on the gate.
                stack.callback(setattr, bias, "output", False)

            # --- 1. Locate fr, with the gate unbiased ---
            bias.output = False
            self._sweep = Sweep(
                freq_center=self.freq_center,
                freq_span=self.freq_span,
                df=self.sweep_df,
                num_averages=self.sweep_num_averages,
                amp=self.amp,
                output_port=self.output_port,
                input_port=self.input_port,
                dither=self.dither,
                device=self.device,
                filter=self.filter,
                notes=self._notes("Locate fr for the QC trace"),
                auto_fit=True,
            )
            self.sweep_file = self._sweep.run(**run_kwargs)

            fit = self._sweep.fit_results
            if not fit or not np.isfinite(fit.get("fr", np.nan)):
                raise RuntimeError(
                    "The locating sweep did not yield a usable fr, so the QC trace cannot be "
                    f"read out. The sweep itself is saved at {self.sweep_file}; inspect it "
                    "with Sweep.load(...).analyze() and retry with a corrected freq_center "
                    "or freq_span."
                )
            self.fit_results = fit
            self.fr = float(fit["fr"])
            self.fr_err = float(fit.get("fr_err", np.nan))
            print(f"QC trace: reading out at fr = {self.fr / 1e9:.6f} GHz")

            # --- 2. QC trace: gated ramp, externally triggered ---
            bias.sawtooth(
                vpp=self.ramp_vpp,
                freq_hz=self.ramp_freq_hz,
                offset_v=self.ramp_offset_v,
                symmetry_pct=self.ramp_symmetry_pct,
                gated=True,
            )
            self._qc_stream = self._make_timestream(
                bias.samples_for_periods(
                    self.num_periods,
                    self.sampling_frequency,
                    freq_hz=self.ramp_freq_hz,
                    discard_ms=self.discard_start_ms,
                ),
                external_trigger=True,
                notes="Gated sawtooth QC trace",
            )
            self._qc_stream.attach(bias=bias)
            self.qc_file = self._qc_stream.run(**run_kwargs)
            # Fold on the ramp's own period at the tuned sample rate rather than dividing the
            # record by num_periods: tuning can shift df slightly, and a period window off by
            # a sample would smear the average across blocks instead of dropping a leftover.
            self.time_ms, self.avg_iq = fold_timestream(
                self._qc_stream,
                self._qc_stream.df,
                period_s=1.0 / self.ramp_freq_hz,
            )

            # --- 3. Bias hunt: keep the constant bias with the largest parity contrast ---
            ts_samples = self._stream_samples()
            contrasts = np.full(self.n_bias_try, np.nan)
            best_so_far = -np.inf
            self._bias_streams = []
            for ii, voltage in enumerate(self.bias_voltages):
                bias.constant(float(voltage))
                stream = self._make_timestream(
                    ts_samples,
                    external_trigger=False,
                    notes=f"Constant bias {voltage:.4f} V (try {ii + 1}/{self.n_bias_try})",
                )
                stream.attach(bias=bias)
                path = stream.run(**run_kwargs)

                contrasts[ii] = self._parity_contrast(stream)
                self._bias_streams.append(stream)
                print(
                    f"Bias try {ii + 1}/{self.n_bias_try}: {voltage:.4f} V, "
                    f"std(|signal|) = {contrasts[ii]:.4e}"
                )
                if contrasts[ii] > best_so_far:
                    best_so_far = contrasts[ii]
                    self.best_bias_file = path

            self.parity_contrast = contrasts
            best = int(np.nanargmax(contrasts))
            self.best_bias = float(self.bias_voltages[best])
            self.best_contrast = float(contrasts[best])
            print(
                f"Max parity contrast at bias {self.best_bias:.4f} V, "
                f"std(|signal|) = {self.best_contrast:.4e}"
            )

            # --- 4. Free-running ramp, same length as the constant-bias tries ---
            # Free-running, not gated: the Presto trigger is deasserted for this acquisition
            # (external_trigger=False), so a gated ramp would sit at its burst start level and
            # the stream would record a static bias instead of a swept one.
            bias.sawtooth(
                vpp=self.ramp_vpp,
                freq_hz=self.ramp_freq_hz,
                offset_v=self.ramp_offset_v,
                symmetry_pct=self.ramp_symmetry_pct,
                gated=False,
            )
            self._ramp_stream = self._make_timestream(
                ts_samples,
                external_trigger=False,
                notes="Free-running sawtooth bias, matching the constant-bias tries",
            )
            self._ramp_stream.attach(bias=bias)
            self.ramp_file = self._ramp_stream.run(**run_kwargs)

        # Saved after the bias is de-energised, so a failure here cannot leave it applied.
        return self.save(save_filename=save_filename)

    def save(self, save_filename: Optional[str] = None) -> str:
        """Write this measurement's HDF5 file and MongoDB record.

        :param save_filename: Explicit path. Generated under ``DAQ_DATA_FOLDER`` when ``None``.
        :returns: Path of the written file.

        """
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "QCTrace":
        """Rebuild a measurement from its saved HDF5 file.

        The derived products -- the fitted ``fr``, the folded QC trace and the
        contrast-versus-bias curve -- are restored, along with the paths of the constituent
        acquisitions. The constituent objects themselves are not: load them from those paths
        with :meth:`Sweep.load <daq.measurements.sweep.Sweep.load>` and
        :meth:`TimeStream.load <daq.measurements.timestream.TimeStream.load>` when you need
        the raw records.

        :param load_filename: Path of the HDF5 file to load.
        :returns: The reconstructed measurement.

        """
        with h5py.File(load_filename, "r") as h5f:
            attrs = h5f.attrs

            self = cls(
                freq_center=float(attrs["freq_center"]),  # type: ignore
                amp=float(attrs["amp"]),  # type: ignore
                output_port=int(attrs["output_port"]),  # type: ignore
                input_port=int(attrs["input_port"]),  # type: ignore
                ramp_vpp=float(attrs["ramp_vpp"]),  # type: ignore
                ramp_freq_hz=float(attrs["ramp_freq_hz"]),  # type: ignore
                ramp_offset_v=float(attrs["ramp_offset_v"]),  # type: ignore
                ramp_symmetry_pct=float(attrs["ramp_symmetry_pct"]),  # type: ignore
                sampling_frequency=float(attrs["sampling_frequency"]),  # type: ignore
                num_periods=int(attrs["num_periods"]),  # type: ignore
                bias_voltages=h5f["bias_voltages"][()],  # type: ignore
                v_min=float(attrs["v_min"]),  # type: ignore
                v_max=float(attrs["v_max"]),  # type: ignore
                ts_duration_s=float(attrs["ts_duration_s"]),  # type: ignore
                freq_span=float(attrs["freq_span"]),  # type: ignore
                sweep_df=float(attrs["sweep_df"]),  # type: ignore
                sweep_num_averages=int(attrs["sweep_num_averages"]),  # type: ignore
                discard_start_ms=float(attrs["discard_start_ms"]),  # type: ignore
                dither=bool(attrs["dither"]),  # type: ignore
                device=attrs.get("device", None),
                filter=attrs.get("filter", None),
                notes=attrs.get("notes", None),
            )

            self.fr = float(attrs["fr"]) if "fr" in attrs else None
            self.fr_err = float(attrs["fr_err"]) if "fr_err" in attrs else None
            self.best_bias = float(attrs["best_bias"]) if "best_bias" in attrs else None
            self.best_contrast = float(attrs["best_contrast"]) if "best_contrast" in attrs else None
            for name in ("sweep_file", "qc_file", "best_bias_file", "ramp_file"):
                setattr(self, name, attrs.get(name, None))
            for name in ("time_ms", "avg_iq", "parity_contrast"):
                if name in h5f:
                    setattr(self, name, h5f[name][()])  # type: ignore

        return self

    # ------------------------------------------------------------------ analysis

    def analyze(self, raw: bool = True, title: Optional[str] = None):
        """Plot the bias hunt and the block-averaged QC trace.

        The top panel shows the parity contrast against gate bias, with the winning bias
        marked. The lower two show the folded QC trace's I and Q over one ramp period, drawn
        by :func:`~daq.analysis.plotting.plot_qc_trace`.

        :param raw: Overlay one un-averaged period of the gated-ramp stream on the folded
            trace, to show what the averaging bought. Only possible right after :meth:`run`,
            since :meth:`load` does not restore the raw stream.
        :param title: Figure title. Defaults to naming the device and the readout frequency.
        :raises RuntimeError: If the measurement has not been run or loaded.
        :returns: The created figure.

        """
        if self.avg_iq is None or self.time_ms is None:
            raise RuntimeError("No QC trace available. Run or load the measurement first.")
        if self.parity_contrast is None:
            raise RuntimeError("No bias hunt available. Run or load the measurement first.")

        import matplotlib.pyplot as plt

        from ..analysis.plotting import plot_qc_trace

        fig = plt.figure(figsize=(8, 9), tight_layout=True)
        ax_bias = fig.add_subplot(3, 1, 1)
        ax_i = fig.add_subplot(3, 1, 2)
        ax_q = fig.add_subplot(3, 1, 3, sharex=ax_i)

        # Sort by voltage so a random draw still reads as a curve rather than a zigzag.
        order = np.argsort(self.bias_voltages)
        ax_bias.plot(
            np.asarray(self.bias_voltages)[order],
            np.asarray(self.parity_contrast)[order],
            ".-",
            color="tab:green",
        )
        if self.best_bias is not None:
            ax_bias.axvline(
                self.best_bias,
                color="tab:red",
                ls="--",
                lw=1.0,
                label=f"best bias {self.best_bias:.4f} V",
            )
            ax_bias.legend(loc="best", fontsize=8)
        ax_bias.set_xlabel("Gate bias [V]")
        ax_bias.set_ylabel(r"Parity contrast std($|S|$) [FS]")
        ax_bias.grid(True, alpha=0.3)

        if title is None:
            parts = ["QC trace (block-averaged)"]
            if self.device is not None:
                parts.append(str(self.device))
            if self.fr is not None:
                parts.append(f"{self.fr / 1e9:.6f} GHz")
            title = " -- ".join(parts)

        plot_qc_trace(
            self.time_ms,
            self.avg_iq,
            raw=self._qc_stream if raw else None,
            ax=(ax_i, ax_q),
            title=title,
        )

        plt.show()
        return fig
