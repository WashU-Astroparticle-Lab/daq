# -*- coding: utf-8 -*-
"""Quantum-capacitance (QC) trace measurement.

One gated sawtooth on the gate, one externally-triggered
:class:`~daq.measurements.timestream.TimeStream` spanning a whole number of ramp periods, and
the fold of that record into a single period.
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import numpy.typing as npt

from ..analysis.folding import fold_timestream
from ..instruments import Agilent33220A
from ..triggers import (
    TriggerAny,
    describe_trigger_states,
    resolve_trigger_states,
    trigger_for,
)
from ._gate_bias import GateBiasMeasurement
from .timestream import TimeStream


class QCTrace(GateBiasMeasurement):
    """Quantum-capacitance trace: a gated gate-voltage ramp, folded into one period.

    The gate is swept by a sawtooth that repeats at ``ramp_freq_hz`` while a single-tone time
    stream records continuously at :attr:`readout_freq`. The record spans ``num_periods`` whole
    ramp periods, and :meth:`fold` block-averages it into one period: uncorrelated noise falls
    by ``sqrt(num_periods)`` and what remains is the device's response to one sweep of the gate
    voltage.

    The ramp is *gated* on a Presto digital output, so it starts with the acquisition rather
    than free-running against it -- which is what makes the blocks line up well enough to
    average at all.

    This measurement does **not** locate the resonance. Read out where a fitted
    :class:`~daq.measurements.sweep.Sweep` says to::

        sw = Sweep(freq_center=2.8e9, ..., auto_fit=True)
        sw.run()
        qct = QCTrace(readout_freq=sw.fit_results["fr"], ...)

    and see :class:`~daq.measurements.bias_hunt.BiasHunt` for the companion measurement that
    parks the gate at constant voltages and ranks them by parity contrast.

    Requires the Presto **and** the 33220A over VISA; it cannot run without hardware.

    **Trigger routing.** The acquisition is gated, by default on whichever Presto digital output
    port the bias generator says it is wired to --
    :attr:`Agilent33220A.trigger_port <daq.instruments._visa.VisaInstrument.trigger_port>`,
    port 1 in the lab's default setup, overridable per instrument or through
    ``DAQ_FGEN_TRIGGER_PORT``. The generator is consulted on *every* run, so rewiring it and
    running again gates the new port; :meth:`load` restores the stored routing for inspection
    but does not pin it. Pass *trigger_states* to override the routing for one measurement.
    Getting this wrong is a silent failure -- an ungated ramp sits at its burst start level and
    the acquisition records a static bias rather than a swept one -- so the resolved states are
    validated up front, refused if they gate nothing, warned about if they leave the
    generator's own port unasserted, and saved with the record.

    :param readout_freq: Readout frequency in hertz, normally a fitted ``fr``.
    :param amp: Drive amplitude in DAC full scale. Convert from dBm with
        :func:`~daq.calibrations.power_dbm_to_amp`.
    :param output_port: Presto output port.
    :param input_port: Presto input port.
    :param ramp_vpp: Ramp peak-to-peak amplitude in volts.
    :param ramp_freq_hz: Ramp repetition frequency in hertz.
    :param ramp_offset_v: Ramp DC offset in volts. Defaults to ``ramp_vpp / 2``, making the
        ramp unipolar-positive (the lab convention).
    :param ramp_symmetry_pct: Ramp symmetry in percent; ``100`` gives a ramp-up sawtooth.
    :param sampling_frequency: Time-stream sample rate in hertz.
    :param num_periods: Number of whole ramp periods the acquisition spans and averages over.
    :param discard_start_ms: Leading milliseconds of start-up junk the time stream drops from
        its in-memory arrays. Passed through to
        :class:`~daq.measurements.timestream.TimeStream` and accounted for in the sample count.
    :param trigger_states: Which Presto digital output ports gate the acquisition, as presto's
        per-port states (``[1]`` for port 1, ``[0, 1]`` for port 2, ``True`` as shorthand for
        ``[1]``). ``None`` (the default) reads the port off the bias generator handed to
        :meth:`run` -- on *every* run, so rewiring the generator and running again gates the new
        port. Validated in ``__init__`` when given explicitly, so a bad routing raises before
        any hardware is touched; an explicit routing that leaves the generator's own port
        unasserted warns, since the ramp would then never be gated.
    :param dither: Whether to dither the Presto output.
    :param device: Device name, required for database logging.
    :param filter: Filter / amplifier chain description, for database logging.
    :param notes: Free-text note. Also prefixed onto the time stream's own note.
    :raises ValueError: If any parameter is out of range.

    """

    def __init__(
        self,
        readout_freq: float,
        amp: float,
        output_port: int,
        input_port: int,
        ramp_vpp: float = 2.0,
        ramp_freq_hz: float = 500.0,
        ramp_offset_v: Optional[float] = None,
        ramp_symmetry_pct: float = 100.0,
        sampling_frequency: float = 5e4,
        num_periods: int = 200,
        discard_start_ms: float = 25.0,
        trigger_states: Optional[TriggerAny] = None,
        dither: bool = True,
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

        if ramp_vpp <= 0:
            raise ValueError(f"ramp_vpp must be positive, got {ramp_vpp}")
        if ramp_freq_hz <= 0:
            raise ValueError(f"ramp_freq_hz must be positive, got {ramp_freq_hz}")
        if not 0.0 <= ramp_symmetry_pct <= 100.0:
            raise ValueError(
                f"ramp_symmetry_pct must be between 0 and 100, got {ramp_symmetry_pct}"
            )
        if num_periods < 1:
            raise ValueError(f"num_periods must be at least 1, got {num_periods}")
        if self.sampling_frequency < ramp_freq_hz:
            raise ValueError(
                f"sampling_frequency={sampling_frequency} Hz gives fewer than one sample per "
                f"{ramp_freq_hz} Hz ramp period; raise the sample rate or slow the ramp."
            )

        self.ramp_vpp = ramp_vpp
        self.ramp_freq_hz = ramp_freq_hz
        self.ramp_offset_v = ramp_vpp / 2.0 if ramp_offset_v is None else ramp_offset_v
        self.ramp_symmetry_pct = ramp_symmetry_pct
        self.num_periods = num_periods
        self._warn_if_period_not_integral()

        # Which digital output ports gate the acquisition. An explicit routing is resolved
        # here, so a bad one raises before the hardware is touched; None defers to run(), where
        # the generator that knows its own wiring is in hand.
        #
        # The caller's choice is kept privately as well as on the record, because run() writes
        # the *resolved* states onto `trigger_states` for saving. Without the private copy, a
        # second run() of the same object (or of a load()ed one) would read back its own
        # earlier answer and ignore a rewired generator -- reintroducing the silent failure
        # this class is meant to have eliminated.
        self._trigger_states_arg = (
            None if trigger_states is None else self._check_trigger_states(trigger_states)
        )
        self.trigger_states = self._trigger_states_arg

        # Results - replaced by run()
        self.time_ms = None
        """Time axis of one ramp period in milliseconds."""
        self.avg_iq = None
        """Block-averaged QC trace, shape ``(2, n_samples)``; row 0 is I, row 1 is Q."""
        self.num_periods_folded = None
        """Blocks :meth:`fold` actually averaged -- the ``N`` in the ``sqrt(N)`` noise gain.

        Normally :attr:`num_periods`, but lower whenever the record does not divide evenly
        into whole periods. Recorded separately because the requested count over-states the
        averaging in that case, and the difference is not recoverable from the saved file.
        """
        self.qc_file = None
        """Path of the gated-ramp time stream's HDF5 file."""

        # The acquisition itself, kept off the saved record (Base skips underscore-prefixed
        # attributes) and exposed through a read-only property.
        self._qc_stream: Optional[TimeStream] = None

    # ------------------------------------------------------------------ constituent objects

    @property
    def qc_stream(self) -> Optional[TimeStream]:
        """The gated-ramp time stream the QC trace was folded from.

        :returns: The time stream, or ``None`` before :meth:`run`.

        """
        return self._qc_stream

    # ------------------------------------------------------------------ helpers

    def _warn_if_period_not_integral(self) -> None:
        """Warn when one ramp period is not a whole number of samples.

        Folding cuts the record into blocks of ``round(period_s * fs)`` samples -- an integer.
        When the true period is fractional, every block starts a fraction of a sample later
        than the last and the error accumulates over :attr:`num_periods`, so a feature sharp
        on the scale of the drift is averaged away rather than reinforced. At 50 kHz with a
        300 Hz ramp (166.67 samples per period) a sharp feature loses about 90 % of its
        contrast over 200 periods, and nothing about the resulting trace says so.

        Warned rather than refused: a slow, smooth QC trace tolerates the drift, and the user
        may know that. The cure is to pick a ``sampling_frequency`` that is a whole multiple of
        ``ramp_freq_hz``.

        This uses the *requested* sample rate, which is the one the caller can act on;
        ``TimeStream.run`` tunes it slightly, so the realised drift differs a little. That
        tuning is small and cannot rescue a ratio that is far from integral.

        """
        samples_per_period = self.sampling_frequency / self.ramp_freq_hz
        drift = abs(samples_per_period - round(samples_per_period))
        if drift <= 1e-6 * samples_per_period:
            return
        warnings.warn(
            f"sampling_frequency={self.sampling_frequency:g} Hz is not a whole multiple of "
            f"ramp_freq_hz={self.ramp_freq_hz:g} Hz: one ramp period is "
            f"{samples_per_period:.4f} samples, so each folded block starts {drift:.4f} "
            f"samples later than the last and drifts {drift * self.num_periods:.1f} samples "
            f"({100 * drift * self.num_periods / samples_per_period:.0f} % of a period) over "
            f"{self.num_periods} periods. Features sharper than that are averaged away, and "
            "the folded trace gives no sign of it. Pick a sampling_frequency that divides "
            f"evenly by the ramp rate (e.g. {round(samples_per_period) * self.ramp_freq_hz:g} "
            "Hz).",
            stacklevel=3,
        )

    @staticmethod
    def _check_trigger_states(trigger_states: TriggerAny) -> npt.NDArray[np.int64]:
        """Resolve *trigger_states* and refuse a routing that gates nothing.

        A gated ramp with no port asserted is exactly the silent failure this measurement
        cannot afford: the generator holds its burst start level, the acquisition succeeds,
        and the trace is flat because the gate never moved. ``False`` and all-zero states are
        therefore rejected rather than run.

        :param trigger_states: Anything :func:`~daq.triggers.resolve_trigger_states` accepts.
        :raises ValueError: If the states are invalid, or gate no port at all.
        :returns: The resolved per-port states.

        """
        states = resolve_trigger_states(trigger_states)
        if not states.any():
            raise ValueError(
                f"trigger_states={trigger_states!r} gates no digital output port, so the "
                "QC-trace ramp would never run: the generator would hold its burst start "
                "level and the acquisition would record a static bias instead of a swept "
                "one. Pass the port that gates the bias generator (True or [1] for port 1), "
                "or leave trigger_states unset to take it from the generator's own "
                "trigger_port."
            )
        return states

    def _resolve_run_trigger_states(self, bias: Agilent33220A) -> npt.NDArray[np.int64]:
        """Decide which ports gate this run.

        Reads the caller's own argument, never the states a previous run resolved, so the
        default ("ask the generator") holds on every run of an object -- including one
        restored by :meth:`load`, whose stored routing describes the run that produced the
        file rather than the bench in front of you now.

        :param bias: The gate-bias generator this run is using.
        :raises ValueError: If the routing gates no port, or the generator does not declare
            a ``trigger_port``.
        :returns: The resolved per-port states.

        """
        if self._trigger_states_arg is not None:
            states = self._check_trigger_states(self._trigger_states_arg)
        else:
            states = self._check_trigger_states(trigger_for(bias))
        self._warn_if_generator_ungated(states, bias)
        return states

    @staticmethod
    def _warn_if_generator_ungated(states: npt.NDArray[np.int64], bias: Agilent33220A) -> None:
        """Warn when the routing does not assert the port the generator says it is on.

        Only an explicit *trigger_states* can produce this: it is the one remaining way to
        gate a port while the ramp waits on another, which the acquisition records as a
        static bias. A warning rather than an error, since the override may be deliberate --
        an instrument whose declared ``trigger_port`` is itself wrong.

        :param states: The resolved per-port states for this run.
        :param bias: The gate-bias generator this run is using.

        """
        port = getattr(bias, "trigger_port", None)
        if port is None or (port <= states.size and states[port - 1]):
            return
        warnings.warn(
            f"The QC trace gates {describe_trigger_states(states)}, but the bias generator "
            f"reports trigger_port={port}, which is not among them. Its gated ramp will wait "
            "on a port nothing asserts and the acquisition will record a static bias. Correct "
            "the generator's wiring (bias.trigger_port, or DAQ_FGEN_TRIGGER_PORT) or include "
            f"port {port} in trigger_states.",
            stacklevel=3,
        )

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
        """Acquire the gated-ramp time stream, fold it, and save the derived record.

        The gate-bias generator's output is forced off when the acquisition finishes --
        including on exception -- so no bias is left on the device. When *bias* is omitted the
        generator is opened and closed here; when it is passed in the caller keeps ownership
        of the VISA session and only the output is de-energised.

        Unlike the other measurement classes, this ``run`` takes the *bias generator* as its
        first (and only) positional argument, not ``presto_address`` -- the Presto connection
        parameters are keyword-only, so ``run("172.23.20.29")`` cannot silently bind an
        address string to *bias*. Pass the address as ``run(presto_address=...)``.

        :param bias: An open :class:`~daq.instruments.function_generator.Agilent33220A`. When
            ``None``, one is discovered and opened for the duration of the run.
        :param presto_address: Presto address. Defaults to ``DAQ_PRESTO_ADDRESS``.
        :param presto_port: Presto port. Defaults to the presto default.
        :param ext_ref_clk: Whether to use an external reference clock.
        :param save_filename: Explicit path for this measurement's own HDF5 file.
        :raises ValueError: If the trigger routing gates no port -- including when
            *trigger_states* was left unset and *bias* declares no ``trigger_port``. Raised
            before the acquisition, since an ungated ramp records a static bias.
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

            # Settle the routing before the acquisition: the generator is open, so its wiring
            # is knowable, and a bad routing should abort the run rather than surface as a flat
            # QC trace afterwards.
            self.trigger_states = self._resolve_run_trigger_states(bias)
            print(
                "QC trace: gating the ramp on Presto digital output "
                f"{describe_trigger_states(self.trigger_states)}"
            )
            print(f"QC trace: reading out at {self.readout_freq / 1e9:.6f} GHz")

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
                external_trigger=self.trigger_states,
                notes="Gated sawtooth QC trace",
            )
            self._qc_stream.attach(bias=bias)
            self.qc_file = self._qc_stream.run(**run_kwargs)
            self.fold()

        # Saved after the bias is de-energised, so a failure here cannot leave it applied.
        return self.save(save_filename=save_filename)

    def fold(
        self,
        stream: Optional[TimeStream] = None,
        *,
        period_s: Optional[float] = None,
        n_periods: Optional[int] = None,
        tone: int = 0,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Block-average the acquisition into a single ramp period.

        Thin wrapper over :func:`~daq.analysis.folding.fold_timestream` that knows this
        measurement's ramp period and sample rate, so the usual call is a bare ``qct.fold()``.
        :meth:`run` calls it once; call it again to re-fold on a different period (say, to
        check the ramp really ran at the frequency it was told to) or on a stream reloaded
        from :attr:`qc_file`. Either way the result is stored on :attr:`time_ms` and
        :attr:`avg_iq`, replacing what was there.

        By default the fold uses the ramp's own period, ``1 / ramp_freq_hz``, at the *tuned*
        sample rate -- not the record divided by :attr:`num_periods`. Tuning can shift ``df``
        slightly, and a period window off by a sample smears the average across blocks instead
        of dropping a leftover.

        The number of blocks actually averaged lands on :attr:`num_periods_folded`, which is
        what the ``sqrt(N)`` noise reduction is over. It is normally :attr:`num_periods` but
        falls short whenever the record does not divide evenly, so the saved record says what
        the average was really taken over rather than what was asked for.

        :param stream: Time stream to fold. Must expose ``df`` (the tuned sample rate) and
            ``signal`` -- i.e. a :class:`~daq.measurements.timestream.TimeStream`. Defaults to
            :attr:`qc_stream`, this measurement's own acquisition. To fold a bare array, call
            :func:`~daq.analysis.folding.fold_timestream` directly with an explicit ``fs``.
        :param period_s: Fold on this period in seconds instead of the ramp's.
        :param n_periods: Fold on the record divided into this many periods instead. Mutually
            exclusive with *period_s*.
        :param tone: Which tone to fold; the acquisition is single-tone, so ``0``.
        :raises RuntimeError: If no stream is available -- :meth:`load` restores the folded
            trace but not the raw record, so pass *stream* after loading.
        :raises TypeError: If *stream* carries no ``df``.
        :raises ValueError: If both *period_s* and *n_periods* are given, or the record is
            too short to hold one period.
        :returns: ``(time_ms, avg_iq)``, as :func:`~daq.analysis.folding.fold_timestream`.

        """
        if stream is None:
            stream = self._qc_stream
        if stream is None:
            raise RuntimeError(
                "No time stream to fold. Run the measurement first, or pass a stream "
                "reloaded from qc_file: qct.fold(TimeStream.load(qct.qc_file))."
            )
        # No fallback to the *requested* sampling_frequency. The whole point of folding on
        # `stream.df` is that the hardware tunes the rate away from what was asked for, so
        # quietly substituting the untuned value for an input that does not carry one would
        # reintroduce exactly the smearing this default exists to avoid.
        fs = getattr(stream, "df", None)
        if fs is None:
            raise TypeError(
                f"fold() needs a time stream carrying its tuned sample rate as .df, got "
                f"{type(stream).__name__}. For a bare array of samples call "
                "daq.analysis.fold_timestream(array, fs, period_s=...) directly, with the "
                "rate the data was actually taken at."
            )
        if period_s is None and n_periods is None:
            period_s = 1.0 / self.ramp_freq_hz

        self.time_ms, self.avg_iq = fold_timestream(
            stream,
            fs,
            period_s=period_s,
            n_periods=n_periods,
            tone=tone,
        )
        # fold_timestream computes the block count and discards it. Recover it from the window
        # it produced, so the record carries the averaging actually achieved: the requested
        # num_periods over-states it whenever the record does not divide evenly.
        n_samples = np.asarray(stream.signal).shape[0]
        self.num_periods_folded = int(n_samples // self.avg_iq.shape[1])
        return self.time_ms, self.avg_iq

    def save(self, save_filename: Optional[str] = None) -> str:
        """Write this measurement's HDF5 file and MongoDB record.

        :param save_filename: Explicit path. Generated under ``DAQ_DATA_FOLDER`` when ``None``.
        :returns: Path of the written file.

        """
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "QCTrace":
        """Rebuild a measurement from its saved HDF5 file.

        The folded trace is restored, along with the path of the raw acquisition. The time
        stream itself is not: load it from :attr:`qc_file` with
        :meth:`TimeStream.load <daq.measurements.timestream.TimeStream.load>` when you need the
        raw record (or want to :meth:`fold` it again).

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
                ramp_vpp=float(attrs["ramp_vpp"]),  # type: ignore
                ramp_freq_hz=float(attrs["ramp_freq_hz"]),  # type: ignore
                ramp_offset_v=float(attrs["ramp_offset_v"]),  # type: ignore
                ramp_symmetry_pct=float(attrs["ramp_symmetry_pct"]),  # type: ignore
                sampling_frequency=float(attrs["sampling_frequency"]),  # type: ignore
                num_periods=int(attrs["num_periods"]),  # type: ignore
                discard_start_ms=float(attrs["discard_start_ms"]),  # type: ignore
                dither=bool(attrs["dither"]),  # type: ignore
                device=attrs.get("device", None),
                filter=attrs.get("filter", None),
                notes=attrs.get("notes", None),
            )

            # The stored routing describes the run that produced this file, so it is restored
            # onto the record but *not* as a caller-supplied override: re-running a loaded
            # measurement reads the generator in front of you now.
            if "trigger_states" in h5f:
                self.trigger_states = resolve_trigger_states(h5f["trigger_states"][()])  # type: ignore

            self.qc_file = attrs.get("qc_file", None)
            if "num_periods_folded" in attrs:
                self.num_periods_folded = int(attrs["num_periods_folded"])  # type: ignore
            for name in ("time_ms", "avg_iq"):
                if name in h5f:
                    setattr(self, name, h5f[name][()])  # type: ignore

        return self

    # ------------------------------------------------------------------ analysis

    def analyze(self, raw: bool = True, title: Optional[str] = None):
        """Plot the block-averaged QC trace over one ramp period.

        :param raw: Overlay one un-averaged period of the acquisition on the folded trace, to
            show what the averaging bought. Only possible right after :meth:`run`, since
            :meth:`load` does not restore the raw stream.
        :param title: Figure title. Defaults to naming the device and the readout frequency.
        :raises RuntimeError: If the measurement has not been run or loaded.
        :returns: The created figure.

        """
        if self.avg_iq is None or self.time_ms is None:
            raise RuntimeError("No QC trace available. Run or load the measurement first.")

        import matplotlib.pyplot as plt

        from ..analysis.plotting import plot_qc_trace

        fig, (ax_i, ax_q) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, tight_layout=True)

        if title is None:
            parts = ["QC trace (block-averaged)"]
            if self.device is not None:
                parts.append(str(self.device))
            parts.append(f"{self.readout_freq / 1e9:.6f} GHz")
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
