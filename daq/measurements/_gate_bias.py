# -*- coding: utf-8 -*-
"""Shared machinery for the gate-biased readout measurements.

:class:`~daq.measurements.qc_trace.QCTrace`, :class:`~daq.measurements.bias_hunt.BiasHunt`
and :class:`~daq.measurements.sweep_std_dev.StdDevSweep` are separate measurements -- one
sweeps the gate with a ramp and folds the response, one parks the gate at a series of constant
voltages and ranks them, one repeats the ramp at a series of readout frequencies and ranks
those -- but they read the device out the same way: through the same Presto ports, at the
same sample rate, driven by an :class:`~daq.instruments.function_generator.Agilent33220A` on
the gate. Normally that is one tone at zero IF on a caller-supplied frequency; a multitone
``StdDevSweep`` reads several devices at once through the same builder with a fixed LO.

This module holds only that shared readout, the shared ramp, the trigger routing and the
gated-ramp acquisition sequence, so the measurements differ in their own files by exactly what
makes them different measurements.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from .._base import Base
from ..instruments import Agilent33220A
from ..triggers import (
    TriggerAny,
    describe_trigger_states,
    resolve_trigger_states,
    trigger_for,
)
from .timestream import TimeStream


class GateBiasMeasurement(Base):
    """Base for a readout of a gate-biased device.

    Not a measurement in its own right: it validates and stores the readout parameters the
    concrete measurements share, and builds the :class:`~daq.measurements.timestream.TimeStream`
    they all acquire through. Subclasses call :meth:`_init_readout` from ``__init__`` and then
    add whatever their own step needs; the ones that sweep the gate with a ramp also call
    :meth:`_init_ramp` and acquire through :meth:`_run_gated_ramp`.
    """

    def _init_readout(
        self,
        readout_freq: float,
        amp: float,
        output_port: int,
        input_port: int,
        sampling_frequency: float,
        discard_start_ms: float,
        dither: bool,
        device: Optional[str],
        filter: Optional[str],
        notes: Optional[str],
    ) -> None:
        """Validate and store the readout parameters shared by the gate-bias measurements.

        :param readout_freq: Readout frequency in hertz -- normally a resonance located by a
            preceding :class:`~daq.measurements.sweep.Sweep`.
        :param amp: Drive amplitude in DAC full scale. Convert from dBm with
            :func:`~daq.calibrations.power_dbm_to_amp`.
        :param output_port: Presto output port.
        :param input_port: Presto input port.
        :param sampling_frequency: Time-stream sample rate in hertz.
        :param discard_start_ms: Leading milliseconds of start-up junk each time stream drops
            from its in-memory arrays.
        :param dither: Whether to dither the Presto output.
        :param device: Device name, required for database logging.
        :param filter: Filter / amplifier chain description, for database logging.
        :param notes: Free-text note. Also prefixed onto each sub-measurement's own note.
        :raises ValueError: If any parameter is out of range.

        """
        # A nan passes every one of the comparisons below, so finiteness is checked first.
        for name, value in (
            ("readout_freq", readout_freq),
            ("amp", amp),
            ("sampling_frequency", sampling_frequency),
            ("discard_start_ms", discard_start_ms),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if readout_freq <= 0:
            raise ValueError(f"readout_freq must be positive, got {readout_freq}")
        if not 0.0 < amp < 1.0:
            raise ValueError(f"amp must be between 0 and 1 (DAC full scale), got {amp}")
        if sampling_frequency <= 0:
            raise ValueError(f"sampling_frequency must be positive, got {sampling_frequency}")
        if discard_start_ms < 0:
            raise ValueError(f"discard_start_ms must be non-negative, got {discard_start_ms}")

        self.readout_freq = float(readout_freq)
        self.amp = float(amp)
        self.output_port = output_port
        self.input_port = input_port
        self.sampling_frequency = float(sampling_frequency)
        self.discard_start_ms = float(discard_start_ms)
        self.dither = dither

        self.device = device
        self.filter = filter
        self.notes = notes

    def _init_ramp(
        self,
        ramp_vpp: float,
        ramp_freq_hz: float,
        ramp_offset_v: Optional[float],
        ramp_symmetry_pct: float,
        num_periods: int,
    ) -> None:
        """Validate and store the gate-ramp parameters shared by the folding measurements.

        Call after :meth:`_init_readout`, since the ramp rate is checked against the sample
        rate. Warns through :meth:`_warn_if_period_not_integral` when one ramp period is not a
        whole number of samples.

        :param ramp_vpp: Ramp peak-to-peak amplitude in volts.
        :param ramp_freq_hz: Ramp repetition frequency in hertz.
        :param ramp_offset_v: Ramp DC offset in volts. ``None`` gives ``ramp_vpp / 2``, making
            the ramp unipolar-positive (the lab convention).
        :param ramp_symmetry_pct: Ramp symmetry in percent; ``100`` gives a ramp-up sawtooth.
        :param num_periods: Number of whole ramp periods each acquisition spans.
        :raises ValueError: If any parameter is out of range, or the sample rate gives fewer
            than one sample per ramp period.

        """
        for name, value in (
            ("ramp_vpp", ramp_vpp),
            ("ramp_freq_hz", ramp_freq_hz),
            ("ramp_offset_v", ramp_offset_v),
            ("ramp_symmetry_pct", ramp_symmetry_pct),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
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
                f"sampling_frequency={self.sampling_frequency} Hz gives fewer than one sample "
                f"per {ramp_freq_hz} Hz ramp period; raise the sample rate or slow the ramp."
            )

        self.ramp_vpp = float(ramp_vpp)
        self.ramp_freq_hz = float(ramp_freq_hz)
        self.ramp_offset_v = self.ramp_vpp / 2.0 if ramp_offset_v is None else float(ramp_offset_v)
        self.ramp_symmetry_pct = float(ramp_symmetry_pct)
        self.num_periods = int(num_periods)
        self._warn_if_period_not_integral()

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
            # __init__ -> _init_ramp -> here: point the warning at the caller's constructor.
            stacklevel=4,
        )

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
        external_trigger: TriggerAny,
        notes: str,
        lo_freq: Optional[float] = None,
        if_freqs: Optional[npt.ArrayLike] = None,
        is_usb: Optional[npt.ArrayLike] = None,
        amp: Optional[npt.ArrayLike] = None,
    ) -> TimeStream:
        """Build the time stream a gate-bias measurement acquires through.

        By default a single tone at zero IF on the Presto's own LO at :attr:`readout_freq`, so
        the readout frequency is the mixer frequency and no sideband bookkeeping is needed. The
        keyword overrides build a multitone stream instead -- a fixed *lo_freq* with one
        *if_freqs*/*is_usb*/*amp* entry per tone -- through the same port, rate, trigger and
        discard settings, so the two configurations cannot drift apart.

        :param pixel_counts: Number of samples to acquire, including the discarded start.
        :param external_trigger: Which Presto digital output ports assert a trigger. ``False``
            for an ungated acquisition; for a gated ramp, the ports the bias generator is
            wired to.
        :param notes: Step description for the sub-measurement's note.
        :param lo_freq: Mixer frequency in hertz. Defaults to :attr:`readout_freq`.
        :param if_freqs: Per-tone IF in hertz. Defaults to a single zero-IF tone.
        :param is_usb: Per-tone sideband selection. Defaults to ``TimeStream``'s all-USB.
        :param amp: Per-tone drive in DAC full scale. Defaults to :attr:`amp`.
        :returns: The configured time stream.

        """
        return TimeStream(
            lo_freq=self.readout_freq if lo_freq is None else lo_freq,
            if_freqs=[0.0] if if_freqs is None else if_freqs,
            is_usb=is_usb,
            df=self.sampling_frequency,
            pixel_counts=pixel_counts,
            amp=self.amp if amp is None else amp,
            output_port=self.output_port,
            input_port=self.input_port,
            dither=self.dither,
            device=self.device,
            filter=self.filter,
            notes=self._notes(notes),
            external_trigger=external_trigger,
            discard_start_ms=self.discard_start_ms,
        )

    def _run_gated_ramp(
        self,
        bias: Agilent33220A,
        stream: TimeStream,
        run_kwargs: Dict[str, Any],
    ) -> str:
        """Run *stream* under this measurement's gated sawtooth and de-energise the gate.

        The one place the ramp-under-acquisition sequence lives: put the generator into the
        gated ramp, record its settings on the stream, acquire, and force the output off --
        on the exception path too, so no bias is left on the device whatever happened. The
        stream must already carry the trigger routing that gates the ramp.

        :param bias: The open gate-bias generator.
        :param stream: The configured time stream to acquire.
        :param run_kwargs: Presto connection keywords for ``TimeStream.run``.
        :returns: Path of the time stream's HDF5 file.

        """
        try:
            bias.sawtooth(
                vpp=self.ramp_vpp,
                freq_hz=self.ramp_freq_hz,
                offset_v=self.ramp_offset_v,
                symmetry_pct=self.ramp_symmetry_pct,
                gated=True,
            )
            stream.attach(bias=bias)
            return stream.run(**run_kwargs)
        finally:
            bias.output = False

    # ------------------------------------------------------------------ trigger routing

    @staticmethod
    def _check_trigger_states(trigger_states: TriggerAny) -> npt.NDArray[np.int64]:
        """Resolve *trigger_states* and refuse a routing that gates nothing.

        A gated ramp with no port asserted is exactly the silent failure the folding
        measurements cannot afford: the generator holds its burst start level, the acquisition
        succeeds, and the trace is flat because the gate never moved. ``False`` and all-zero
        states are therefore rejected rather than run.

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

        Reads the caller's own argument (``_trigger_states_arg``), never the states a previous
        run resolved, so the default ("ask the generator") holds on every run of an object --
        including one restored by ``load``, whose stored routing describes the run that
        produced the file rather than the bench in front of you now.

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

    def _stream_samples(self, duration_s: float) -> int:
        """Return the sample count of a fixed-duration stream, including the discarded start.

        :param duration_s: Length of the usable record in seconds, after the discarded start.
        :returns: Number of samples to request.

        """
        n_discard = int(round(self.discard_start_ms * 1e-3 * self.sampling_frequency))
        return n_discard + int(round(duration_s * self.sampling_frequency))
