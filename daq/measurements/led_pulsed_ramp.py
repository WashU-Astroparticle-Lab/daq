# -*- coding: utf-8 -*-
"""A gated gate-voltage ramp recorded under a train of LED flashes."""

from __future__ import annotations

import math
import time
import warnings
from contextlib import ExitStack
from typing import Any, Dict, Optional, Sequence, Union

import h5py
import numpy as np
import numpy.typing as npt

from ..instruments import DC2200, Agilent33220A
from ..triggers import TriggerAny, describe_trigger_states, resolve_trigger_states
from ._gate_bias import GateBiasMeasurement
from .sweep_std_dev import MAX_IF_HZ, MAX_TONES
from .timestream import TimeStream

__all__ = ["LEDPulsedRamp"]


class LEDPulsedRamp(GateBiasMeasurement):
    """One gated-ramp time stream recorded while a DC2200 fires a train of LED flashes.

    The gate is swept by a sawtooth at ``ramp_freq_hz`` -- gated on a Presto digital output
    exactly as :class:`~daq.measurements.qc_trace.QCTrace` does -- while every device in
    *readout_freqs* is read out simultaneously through one multitone
    :class:`~daq.measurements.timestream.TimeStream` for ``duration_s``. The DC2200 runs its
    **internal** pulse engine (``led_on_s`` on, ``led_period_s`` between flash starts, at
    ``led_current_a``), started by a software write from inside ``TimeStream.run``'s
    ``on_acquire`` hook, immediately before the acquisition starts. That is the closest the
    train can be put to sample zero, and it is not a hardware synchronisation: the offset is
    a per-file network latency at the millisecond scale, so the flash phase must be read off
    the data -- normally off a KID tone in the same record -- before anything is folded on
    it. The Presto trigger cannot do better: presto's ``Lockin`` trigger is window-locked and
    holds the line high for a whole record, and its sum-window variant decimates the stream.

    A record with ``led_current_a=0`` is a **dark** file: the same acquisition with the LED
    output never enabled. Dark and illuminated records therefore come from one code path and
    differ only in that field, which is what makes them comparable.

    **The LED's trigger port is never asserted.** In pulse mode the DC2200's modulation SMA
    is an *output* (a TTL copy of the train), so a Presto trigger on that port would drive
    an output into an output. The routing is resolved from the gate generator's own port, and
    a routing that also asserts the LED's port is refused before any hardware moves. The
    inverse of ``QCTrace``'s refusal to run an ungated ramp, and for the same reason: the
    failure would be silent.

    Requires the Presto, the 33220A and the DC2200 over VISA.

    :param readout_freqs: One readout frequency per device in hertz; a scalar for a single
        tone. Every tone must lie within ``MAX_IF_HZ`` of *lo_freq*.
    :param amp: Drive per tone in DAC full scale: a scalar broadcast to every tone or one
        value per tone, summing below 1. Convert from dBm with
        :func:`~daq.calibrations.power_dbm_to_amp` at each tone's frequency.
    :param output_port: Presto output port.
    :param input_port: Presto input port.
    :param duration_s: Usable record length in seconds, after the discarded start.
    :param led_on_s: LED flash length in seconds (the DC2200's pulse ON time).
    :param led_period_s: Time between flash starts in seconds (ON + OFF). Warned about when
        it is not a whole number of gate periods, since the analysis folds on both.
    :param led_current_a: Flash current in amperes; ``0`` records a dark file. The DC2200
        takes pulse amplitude as a percentage of its front-panel current limit, so the
        resolution near the bottom of the range depends on that limit.
    :param expected_led_limit_a: The current limit the DC2200 is expected to read back. When
        given, a different readback aborts the run before the ramp is armed -- the limit sets
        what ``led_current_a`` means, and it is a front-panel setting nobody records.
    :param led_terminal: DC2200 output terminal to select (``1`` or ``2``). ``None`` leaves
        the front-panel selection alone; either way the selected terminal is recorded.
    :param ramp_vpp: Ramp peak-to-peak amplitude in volts.
    :param ramp_freq_hz: Ramp repetition frequency in hertz.
    :param ramp_offset_v: Ramp DC offset in volts; ``None`` gives ``ramp_vpp / 2``.
    :param ramp_symmetry_pct: Ramp symmetry in percent; ``100`` is a ramp-up sawtooth.
    :param sampling_frequency: Time-stream sample rate in hertz. Should be a whole multiple
        of *ramp_freq_hz*; see :meth:`~GateBiasMeasurement._warn_if_period_not_integral`.
    :param discard_start_ms: Leading milliseconds the stream drops from its in-memory
        arrays. Defaults to ``0`` here, unlike the other measurements: the flashes start at
        sample zero and the analysis wants them.
    :param trigger_states: Which Presto digital output ports gate the ramp, as presto's
        per-port states. ``None`` (the default) reads the gate generator's own port on every
        run. Refused if it gates nothing, or if it asserts the LED's port.
    :param dither: Whether to dither the Presto output.
    :param lo_freq: Fixed mixer frequency in hertz. Defaults to the midpoint of
        *readout_freqs*; a single tone then sits at zero IF.
    :param tone_labels: One label per tone, e.g. ``["dev0", ..., "dev9 (KID)"]``.
    :param save_arrays: Forwarded to the stream; see
        :data:`~daq.measurements.timestream.SAVE_ARRAYS`. ``None`` takes its default.
    :param save_dtype: Forwarded to the stream. ``None`` takes its default.
    :param device: Device name, required for database logging.
    :param filter: Filter / amplifier chain description, for database logging.
    :param notes: Free-text note. Also prefixed onto the stream's own note.
    :raises ValueError: If any parameter is out of range.

    """

    def __init__(
        self,
        readout_freqs: Union[float, Sequence[float]],
        amp: Union[float, Sequence[float]],
        output_port: int,
        input_port: int,
        *,
        duration_s: float,
        led_on_s: float,
        led_period_s: float,
        led_current_a: float,
        expected_led_limit_a: Optional[float] = None,
        led_terminal: Optional[int] = None,
        ramp_vpp: float,
        ramp_freq_hz: float,
        ramp_offset_v: Optional[float] = None,
        ramp_symmetry_pct: float = 100.0,
        sampling_frequency: float,
        discard_start_ms: float = 0.0,
        trigger_states: Optional[TriggerAny] = None,
        dither: bool = True,
        lo_freq: Optional[float] = None,
        tone_labels: Optional[Sequence[str]] = None,
        save_arrays: Optional[str] = None,
        save_dtype: Optional[str] = None,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        freqs = np.atleast_1d(np.array(readout_freqs, dtype=np.float64, copy=True))
        if (
            freqs.ndim != 1
            or freqs.size == 0
            or not np.all(np.isfinite(freqs))
            or np.any(freqs <= 0)
        ):
            raise ValueError("readout_freqs must be a non-empty 1-D array of positive hertz")
        if np.unique(freqs).size != freqs.size:
            raise ValueError("readout_freqs must be distinct")
        self.readout_freqs = freqs
        """Readout frequency of each tone in hertz, shape ``(n_tones,)``."""
        self.n_tones = int(freqs.size)
        if self.n_tones > MAX_TONES:
            raise ValueError(
                f"A multitone acquisition can demodulate at most {MAX_TONES} tones on one "
                f"input port, got {self.n_tones}"
            )

        amplitudes = np.asarray(amp, dtype=np.float64)
        if amplitudes.ndim == 0:
            amplitudes = np.full(self.n_tones, amplitudes.item())
        if amplitudes.shape != (self.n_tones,):
            raise ValueError("amp must be a scalar or one amplitude per tone")
        if (
            not np.all(np.isfinite(amplitudes))
            or np.any(amplitudes <= 0)
            or amplitudes.sum() >= 1.0
        ):
            raise ValueError("Tone amplitudes must be finite, positive and sum to less than 1")

        self.lo_freq = float((freqs.min() + freqs.max()) / 2 if lo_freq is None else lo_freq)
        """Mixer frequency in hertz, shared by every tone."""
        if not math.isfinite(self.lo_freq) or self.lo_freq <= 0:
            raise ValueError("lo_freq must be finite and positive")
        if np.any(np.abs(freqs - self.lo_freq) >= MAX_IF_HZ):
            raise ValueError(f"Every tone must be within {MAX_IF_HZ / 1e6:g} MHz of lo_freq")

        if tone_labels is None:
            tone_labels = [f"Tone {tone}" for tone in range(self.n_tones)]
        if (
            isinstance(tone_labels, str)
            or len(tone_labels) != self.n_tones
            or any(not isinstance(label, str) or not label for label in tone_labels)
        ):
            raise ValueError("tone_labels must be one non-empty string per tone")
        self.tone_labels = list(tone_labels)

        self._init_readout(
            readout_freq=float(freqs[0]),
            amp=float(amplitudes[0]),
            output_port=output_port,
            input_port=input_port,
            sampling_frequency=sampling_frequency,
            discard_start_ms=discard_start_ms,
            dither=dither,
            device=device,
            filter=filter,
            notes=notes,
        )
        # As in StdDevSweep: there is no single readout frequency, and a stray scalar would
        # be saved -- and calibrated into a power -- as if there were.
        del self.readout_freq
        self.amp = amplitudes
        """Drive per tone in DAC full scale, shape ``(n_tones,)``."""

        for name, value in (
            ("duration_s", duration_s),
            ("led_on_s", led_on_s),
            ("led_period_s", led_period_s),
            ("led_current_a", led_current_a),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        if led_on_s <= 0:
            raise ValueError(f"led_on_s must be positive, got {led_on_s}")
        if led_period_s <= led_on_s:
            raise ValueError(f"led_period_s={led_period_s} s must exceed led_on_s={led_on_s} s")
        if led_current_a < 0:
            raise ValueError(f"led_current_a must be non-negative (0 = dark), got {led_current_a}")
        if expected_led_limit_a is not None and not (
            math.isfinite(expected_led_limit_a) and expected_led_limit_a > 0
        ):
            raise ValueError(f"expected_led_limit_a must be positive, got {expected_led_limit_a}")
        if (
            led_current_a > 0
            and expected_led_limit_a is not None
            and led_current_a >= expected_led_limit_a
        ):
            raise ValueError(
                f"led_current_a={led_current_a} A is not below the expected limit "
                f"{expected_led_limit_a} A; the DC2200 refuses it"
            )
        if led_terminal is not None and led_terminal not in (1, 2):
            raise ValueError(f"led_terminal must be 1 or 2, got {led_terminal}")

        self.duration_s = float(duration_s)
        self.led_on_s = float(led_on_s)
        self.led_period_s = float(led_period_s)
        self.led_current_a = float(led_current_a)
        self.dark = self.led_current_a == 0.0
        """Whether this record is a dark file (LED output never enabled)."""
        self.expected_led_limit_a = expected_led_limit_a
        self.led_terminal = led_terminal

        self._init_ramp(
            ramp_vpp=ramp_vpp,
            ramp_freq_hz=ramp_freq_hz,
            ramp_offset_v=ramp_offset_v,
            ramp_symmetry_pct=ramp_symmetry_pct,
            num_periods=max(1, int(round(self.duration_s * ramp_freq_hz))),
        )
        self._warn_if_led_period_not_integral()

        self._trigger_states_arg = (
            None if trigger_states is None else self._check_trigger_states(trigger_states)
        )
        self.trigger_states = self._trigger_states_arg
        self.save_arrays = save_arrays
        self.save_dtype = save_dtype

        # Results - replaced by run()
        self.raw_file = None
        """Path of the time stream's HDF5 file."""
        self.df_tuned = None
        """The *tuned* sample rate the stream was acquired at, in hertz."""
        self.n_samples = None
        """Samples in the stream's in-memory record, after the discarded start."""
        self.led_flashes_expected = None
        """``duration_s / led_period_s`` -- flashes the record should contain (0 when dark)."""
        self._stream: Optional[TimeStream] = None

    # ------------------------------------------------------------------ validation

    def _warn_if_led_period_not_integral(self) -> None:
        """Warn when the flash period is not a whole number of gate periods.

        The analysis bins the record by gate period and folds those bins on the flash period;
        when the two are commensurate every flash lands at the same gate phase and the folded
        histogram's bins are whole gate periods. Otherwise the flash walks through the gate
        period and the fold smears by up to one gate period. Warned rather than refused:
        the smear is one bin wide and may be acceptable.

        """
        ratio = self.led_period_s * self.ramp_freq_hz
        drift = abs(ratio - round(ratio))
        if drift <= 1e-6 * max(ratio, 1.0):
            return
        warnings.warn(
            f"led_period_s={self.led_period_s:g} s is {ratio:.3f} gate periods at "
            f"ramp_freq_hz={self.ramp_freq_hz:g} Hz, not a whole number: successive flashes "
            "land at different gate phases, so a fold on the flash period smears by up to one "
            f"gate period ({1 / self.ramp_freq_hz * 1e6:.0f} us). Pick led_period_s a multiple "
            f"of {1 / self.ramp_freq_hz:g} s.",
            stacklevel=3,
        )

    @staticmethod
    def _refuse_led_port(states: npt.NDArray[np.int64], led: Any) -> None:
        """Raise if *states* assert the port the LED says it is wired to.

        :param states: The resolved per-port states for this run.
        :param led: The LED driver.
        :raises ValueError: If the LED's port is asserted.

        """
        port = getattr(led, "trigger_port", None)
        if port is None or port > states.size or not states[port - 1]:
            return
        raise ValueError(
            f"trigger routing {describe_trigger_states(states)} asserts port {port}, which the "
            "LED driver reports as its modulation input. In pulse mode that connector is an "
            "OUTPUT -- a TTL copy of the pulse train -- so a Presto trigger on it would drive "
            "an output into an output. This measurement times the LED from the DC2200's own "
            "engine; gate the bias generator's port only (leave trigger_states unset)."
        )

    # ------------------------------------------------------------------ constituent objects

    @property
    def led_stream(self) -> Optional[TimeStream]:
        """The acquired time stream, or ``None`` before :meth:`run` and after :meth:`load`."""
        return self._stream

    # ------------------------------------------------------------------ acquisition

    def run(
        self,
        bias: Optional[Agilent33220A] = None,
        led: Optional[DC2200] = None,
        *,
        presto_address: Optional[str] = None,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
        save_filename: Optional[str] = None,
        stream_filename: Optional[str] = None,
    ) -> str:
        """Acquire one record and save the derived measurement.

        Order of operations, every step recorded on the stream and on this object:

        1. LED output off; select and read back the terminal; check the current limit against
           *expected_led_limit_a* and the protection flags. Configure the pulse train with
           the output still off (a dark file skips this).
        2. Resolve the trigger routing from the gate generator, refuse it if it asserts the
           LED's port.
        3. Put the generator into the gated ramp and start the stream. Inside ``on_acquire``
           -- after the Presto is configured, immediately before ``get_pixels`` -- enable the
           LED output, which starts the train, and stamp the host clock either side of that
           write.
        4. Whatever happens, LED output off and gate output off on the way out.

        Both instruments are opened here when not passed in; a caller-supplied instrument is
        de-energised but never closed.

        :param bias: An open :class:`~daq.instruments.function_generator.Agilent33220A`.
        :param led: An open :class:`~daq.instruments.dc2200.DC2200`.
        :param presto_address: Presto address. Defaults to ``DAQ_PRESTO_ADDRESS``.
        :param presto_port: Presto port. Defaults to the presto default.
        :param ext_ref_clk: Whether to use an external reference clock.
        :param save_filename: Explicit path for this measurement's own HDF5 file.
        :param stream_filename: Explicit path for the time stream's HDF5 file.
        :raises ValueError: If the routing gates no port or asserts the LED's port.
        :raises RuntimeError: If the DC2200's current limit is not the expected one, or a
            protection flag has tripped.
        :returns: Path of this measurement's HDF5 file.

        """
        run_kwargs: Dict[str, Any] = dict(
            presto_address=presto_address,
            presto_port=presto_port,
            ext_ref_clk=ext_ref_clk,
            save_filename=stream_filename,
        )
        self.raw_file = None
        self.df_tuned = None
        self.n_samples = None
        self._stream = None

        with ExitStack() as stack:
            if bias is None:
                bias = stack.enter_context(Agilent33220A())
            else:
                stack.callback(setattr, bias, "output", False)
            if led is None:
                led = stack.enter_context(DC2200())
            else:
                stack.callback(setattr, led, "output", False)

            # --- 1. LED: off, checked, configured but not armed ------------------------
            led.output = False
            if self.led_terminal is not None:
                led.terminal = self.led_terminal
            self.led_terminal_read = int(led.terminal)
            limit = float(led.current_limit)
            if self.expected_led_limit_a is not None and not math.isclose(
                limit, self.expected_led_limit_a, rel_tol=0.0, abs_tol=1e-4
            ):
                raise RuntimeError(
                    f"The DC2200 current limit reads {limit:g} A; this measurement expects "
                    f"{self.expected_led_limit_a:g} A. Pulse amplitude is a percentage of that "
                    "limit, so set it on the front panel before running."
                )
            tripped = {name for name, flag in led.protection_status().items() if flag}
            if tripped:
                raise RuntimeError(f"DC2200 protection has tripped: {sorted(tripped)}")
            if not self.dark:
                led.configure_pulse(
                    on_time_s=self.led_on_s,
                    off_time_s=self.led_period_s - self.led_on_s,
                    current_a=self.led_current_a,
                    count=0,
                    output=False,
                )
            self.attach(led=led)

            # --- 2. routing: the gate's port, never the LED's ----------------------------
            self.trigger_states = self._resolve_run_trigger_states(bias)
            self._refuse_led_port(self.trigger_states, led)
            print(
                f"LED pulsed ramp: gating the ramp on Presto digital output "
                f"{describe_trigger_states(self.trigger_states)}; "
                + (
                    "dark (LED output stays off)"
                    if self.dark
                    else f"LED {self.led_on_s * 1e6:g} us every {self.led_period_s * 1e3:g} ms "
                    f"at {self.led_current_a * 1e3:g} mA, software-started at acquisition"
                )
            )

            # --- 3. the stream -----------------------------------------------------------
            signed_if = self.readout_freqs - self.lo_freq
            stream = self._make_timestream(
                self._stream_samples(self.duration_s),
                external_trigger=self.trigger_states,
                notes=(
                    "LED pulsed ramp: "
                    + (
                        "dark"
                        if self.dark
                        else f"{self.led_on_s * 1e6:g} us / "
                        f"{self.led_period_s * 1e3:g} ms at {self.led_current_a * 1e3:g} mA"
                    )
                ),
                lo_freq=self.lo_freq,
                if_freqs=np.abs(signed_if),
                is_usb=signed_if >= 0,
                amp=self.amp,
                save_arrays=self.save_arrays,
                save_dtype=self.save_dtype,
            )
            stream.attach(
                led=led,
                led_plan={
                    "on_s": self.led_on_s,
                    "period_s": self.led_period_s,
                    "current_a": self.led_current_a,
                    "dark": self.dark,
                    "terminal": self.led_terminal_read,
                    "scheme": (
                        "DC2200 internal pulse engine, software-started in on_acquire; "
                        "no Presto trigger on the LED"
                    ),
                },
                tones={"labels": ",".join(self.tone_labels)},
            )

            host: Dict[str, float] = {}

            def start_train() -> None:
                # Called by TimeStream.run after the Presto is configured and immediately
                # before get_pixels(). The write that enables the output starts the train.
                host["acquire_hook_unix"] = time.time()
                if not self.dark:
                    host["led_enable_before_unix"] = time.time()
                    led.output = True
                    host["led_enable_after_unix"] = time.time()
                    host["led_enable_write_s"] = (
                        host["led_enable_after_unix"] - host["led_enable_before_unix"]
                    )
                stream.attach(host=host)  # host stamps, not measured optical times

            run_kwargs["on_acquire"] = start_train
            path = self._run_gated_ramp(bias, stream, run_kwargs)
            led.output = False
            self.attach(bias=bias, host=host)

            self._stream = stream
            self.raw_file = path
            self.df_tuned = float(stream.df)
            self.n_samples = int(stream.signal.shape[0])
            self.led_flashes_expected = 0.0 if self.dark else self.duration_s / self.led_period_s

        return self.save(save_filename=save_filename)

    # ------------------------------------------------------------------ persistence

    def save(self, save_filename: Optional[str] = None) -> str:
        """Write this measurement's HDF5 file and MongoDB record.

        :param save_filename: Explicit path. Generated under ``DAQ_DATA_FOLDER`` when ``None``.
        :returns: Path of the written file.

        """
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "LEDPulsedRamp":
        """Rebuild a measurement from its saved HDF5 file.

        The record is restored; the time stream is not. Load it from :attr:`raw_file` with
        :meth:`TimeStream.load <daq.measurements.timestream.TimeStream.load>`.

        :param load_filename: Path of the HDF5 file to load.
        :returns: The reconstructed measurement.

        """

        def text(value: Any) -> Any:
            return value.decode() if isinstance(value, bytes) else value

        with h5py.File(load_filename, "r") as h5f:
            attrs = h5f.attrs
            with warnings.catch_warnings():
                # The saved ratios already warned once, when the record was taken.
                warnings.simplefilter("ignore")
                self = cls(
                    readout_freqs=h5f["readout_freqs"][()],  # type: ignore
                    amp=h5f["amp"][()],  # type: ignore
                    output_port=int(attrs["output_port"]),  # type: ignore
                    input_port=int(attrs["input_port"]),  # type: ignore
                    duration_s=float(attrs["duration_s"]),  # type: ignore
                    led_on_s=float(attrs["led_on_s"]),  # type: ignore
                    led_period_s=float(attrs["led_period_s"]),  # type: ignore
                    led_current_a=float(attrs["led_current_a"]),  # type: ignore
                    expected_led_limit_a=(
                        float(attrs["expected_led_limit_a"])
                        if "expected_led_limit_a" in attrs
                        else None
                    ),
                    led_terminal=int(attrs["led_terminal"]) if "led_terminal" in attrs else None,
                    ramp_vpp=float(attrs["ramp_vpp"]),  # type: ignore
                    ramp_freq_hz=float(attrs["ramp_freq_hz"]),  # type: ignore
                    ramp_offset_v=float(attrs["ramp_offset_v"]),  # type: ignore
                    ramp_symmetry_pct=float(attrs["ramp_symmetry_pct"]),  # type: ignore
                    sampling_frequency=float(attrs["sampling_frequency"]),  # type: ignore
                    discard_start_ms=float(attrs["discard_start_ms"]),  # type: ignore
                    dither=bool(attrs["dither"]),  # type: ignore
                    lo_freq=float(attrs["lo_freq"]),  # type: ignore
                    tone_labels=(
                        h5f["tone_labels"].asstr()[()].tolist() if "tone_labels" in h5f else None
                    ),
                    save_arrays=text(attrs["save_arrays"]) if "save_arrays" in attrs else None,
                    save_dtype=text(attrs["save_dtype"]) if "save_dtype" in attrs else None,
                    device=text(attrs.get("device", None)),
                    filter=text(attrs.get("filter", None)),
                    notes=text(attrs.get("notes", None)),
                )
            if "trigger_states" in h5f:
                self.trigger_states = resolve_trigger_states(h5f["trigger_states"][()])  # type: ignore
            for name in (
                "raw_file",
                "df_tuned",
                "n_samples",
                "led_flashes_expected",
                "led_terminal_read",
            ):
                if name in attrs:
                    setattr(self, name, text(attrs[name]))
            if self.n_samples is not None:
                self.n_samples = int(self.n_samples)
            # Attached instrument state (led_*, bias_*, host_*) comes back too, like TimeStream.
            consumed = {
                "output_port",
                "input_port",
                "duration_s",
                "led_on_s",
                "led_period_s",
                "led_current_a",
                "expected_led_limit_a",
                "led_terminal",
                "ramp_vpp",
                "ramp_freq_hz",
                "ramp_offset_v",
                "ramp_symmetry_pct",
                "sampling_frequency",
                "discard_start_ms",
                "dither",
                "lo_freq",
                "save_arrays",
                "save_dtype",
                "device",
                "filter",
                "notes",
                "raw_file",
                "df_tuned",
                "n_samples",
                "led_flashes_expected",
                "led_terminal_read",
                "dark",
                "num_periods",
                "n_tones",
            }
            for key, value in attrs.items():
                if str(key) not in consumed and not hasattr(self, str(key)):
                    setattr(self, str(key), text(value))
        return self

    # ------------------------------------------------------------------ analysis

    def analyze(
        self,
        *,
        tone: Optional[int] = None,
        window_s: Optional[float] = None,
        title: Optional[str] = None,
    ):
        """Quick look: each tone's principal-axis projection against time.

        A diagnostic for the bench, not the analysis -- the LED flashes should be visible on a
        KID tone at high current, and the gate ramp's QC structure on the QPD tones. Needs
        the live stream (after :meth:`run`); load it from :attr:`raw_file` otherwise.

        :param tone: A single tone to draw; ``None`` draws every tone.
        :param window_s: Seconds from the start of the record to show; ``None`` shows all.
        :param title: Figure title.
        :raises RuntimeError: If there is no stream to draw.
        :returns: The matplotlib figure.

        """
        import matplotlib.pyplot as plt

        from ..analysis.qc_periods import principal_axis, project

        stream = self._stream
        if stream is None or stream.signal is None:
            raise RuntimeError(
                "No time stream to draw: analyze() needs the live stream from run(). Load it "
                "with TimeStream.load(measurement.raw_file) and use its own analyze()."
            )
        tones = list(range(self.n_tones)) if tone is None else [int(tone)]
        n = (
            stream.signal.shape[0]
            if window_s is None
            else min(stream.signal.shape[0], int(round(window_s * stream.df)))
        )
        t = np.arange(n) / stream.df
        fig, axes = plt.subplots(
            len(tones), 1, figsize=(10, 1.8 * len(tones) + 1), sharex=True, squeeze=False
        )
        for ax, index in zip(axes[:, 0], tones):
            z = stream.signal[:n, index]
            axis, origin = principal_axis(z)
            ax.plot(t, project(z, axis, origin), lw=0.6)
            ax.set_ylabel(self.tone_labels[index], fontsize=8)
            ax.grid(alpha=0.2)
        axes[-1, 0].set_xlabel("time from first retained sample [s]")
        fig.suptitle(
            title
            or (
                "dark"
                if self.dark
                else f"LED {self.led_on_s * 1e6:g} us / {self.led_period_s * 1e3:g} ms at "
                f"{self.led_current_a * 1e3:g} mA"
            )
        )
        fig.tight_layout()
        return fig
