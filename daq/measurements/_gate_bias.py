# -*- coding: utf-8 -*-
"""Shared machinery for the gate-biased single-tone readout measurements.

:class:`~daq.measurements.qc_trace.QCTrace` and
:class:`~daq.measurements.bias_hunt.BiasHunt` are separate measurements -- one sweeps the gate
with a ramp and folds the response, the other parks the gate at a series of constant voltages
and ranks them -- but they read the device out the same way: one tone at a caller-supplied
frequency, through the same Presto ports, at the same sample rate, driven by an
:class:`~daq.instruments.function_generator.Agilent33220A` on the gate.

This module holds only that shared readout, so the two measurements differ in their file by
exactly what makes them different measurements.
"""

from __future__ import annotations

from typing import Optional

from .._base import Base
from ..triggers import TriggerAny
from .timestream import TimeStream


class GateBiasMeasurement(Base):
    """Base for a single-tone readout of a gate-biased device.

    Not a measurement in its own right: it validates and stores the readout parameters the
    concrete measurements share, and builds the :class:`~daq.measurements.timestream.TimeStream`
    they both acquire through. Subclasses call :meth:`_init_readout` from ``__init__`` and then
    add whatever their own step needs.
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
        """Validate and store the readout parameters shared by both measurements.

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
    ) -> TimeStream:
        """Build a single-tone time stream at :attr:`readout_freq`.

        The tone is placed at zero IF on the Presto's own LO, so the readout frequency is the
        mixer frequency and no sideband bookkeeping is needed.

        :param pixel_counts: Number of samples to acquire, including the discarded start.
        :param external_trigger: Which Presto digital output ports assert a trigger. ``False``
            for an ungated acquisition; for a gated ramp, the ports the bias generator is
            wired to.
        :param notes: Step description for the sub-measurement's note.
        :returns: The configured time stream.

        """
        return TimeStream(
            lo_freq=self.readout_freq,
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

    def _stream_samples(self, duration_s: float) -> int:
        """Return the sample count of a fixed-duration stream, including the discarded start.

        :param duration_s: Length of the usable record in seconds, after the discarded start.
        :returns: Number of samples to request.

        """
        n_discard = int(round(self.discard_start_ms * 1e-3 * self.sampling_frequency))
        return n_discard + int(round(duration_s * self.sampling_frequency))
