# -*- coding: utf-8 -*-
"""Thorlabs DC2200 LED driver."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from ..config import get_led_resource, get_led_trigger_port
from ._visa import InstrumentError, VisaInstrument


class DC2200(VisaInstrument):
    """Thorlabs DC2200 LED driver, controlled over USB with SCPI.

    Supports four modes, which differ in what generates the pulse timing:

    - :meth:`configure_cc` -- constant current, no timing structure.
    - :meth:`configure_pwm` / :meth:`pwm_train` -- a burst of pulses defined by a *duty
      cycle*. The duty floor of 0.1 % couples width to rate: the narrowest pulse available is
      ``0.001 / freq_hz``, so 10 us is only reachable at 100 Hz.
    - :meth:`configure_pulse` -- a burst of pulses defined by an explicit *width*, decoupling
      width from repetition rate. This is the mode for a short pulse at a slow rate, e.g.
      10 us at 2 Hz.
    - :meth:`configure_ttl` -- the LED follows an external logic signal, the only mode whose
      timing can be synchronised to an acquisition. Driven from a ``TimeStream`` that means a
      steady illumination window spanning the record, not pulses within it.

    Requested currents are validated against the instrument's own configured current limit,
    queried at first use, so a typo cannot drive the LED past what the head is set up for.

    Note two Thorlabs SCPI spellings that are easy to get wrong and both required verbatim:
    the constant-current header is ``SOURCE1:CCURENT`` (missing the second ``R``), and the PWM
    frequency is ``SOURCE1:PWM:FREQ`` -- the long form ``FREQUENCY`` is rejected with
    ``-113 Undefined header``.

    :param resource: Explicit VISA resource string. When ``None``, uses ``DAQ_LED_RESOURCE``
        and otherwise autodiscovers by ``*IDN?``.
    :param timeout_ms: VISA I/O timeout in milliseconds.
    :param backend: PyVISA backend spec. Defaults to ``DAQ_VISA_BACKEND``.
    :param transcript_path: Optional path to a SCPI transcript file.
    :param trigger_port: Presto digital output port wired to the rear-panel modulation input,
        which gates :meth:`configure_ttl`. Defaults to ``DAQ_LED_TRIGGER_PORT``, then to
        :attr:`TRIGGER_PORT`.

    """

    IDN_KEYWORDS = ("DC2200",)
    RESOURCE_HINTS = ("0x1313::0x80C8",)
    ENV_VAR = "DAQ_LED_RESOURCE"
    TRIGGER_PORT = 2
    """Presto digital output port wired to the modulation input in the lab's default setup.

    Only :meth:`configure_ttl` is gated by it -- PWM and pulse mode are timed by the
    instrument itself and start on the software write that enables the output.

    """
    TRIGGER_PORT_ENV_VAR = "DAQ_LED_TRIGGER_PORT"

    # --- SCPI headers, from the DC2200 Operation Manual v1.8 (29-Nov-2023), section 4.3.2.
    # Optional (bracketed) tree nodes are omitted: SOURce1:PULSe[:BRIGhtness][:LEVel]
    # [:AMPLitude] accepts the short form used here.
    PULSE_MODE_COMMAND: str = "SOURCE1:MODE PULS"
    PULSE_BRIGHTNESS_HEADER: str = "SOURCE1:PULSE:BRIGHTNESS:LEVEL:AMPLITUDE"
    PULSE_ONTIME_HEADER: str = "SOURCE1:PULSE:ONTIME"
    PULSE_OFFTIME_HEADER: str = "SOURCE1:PULSE:OFFTIME"
    PULSE_COUNT_HEADER: str = "SOURCE1:PULSE:COUNT"
    TTL_CURRENT_HEADER: str = "SOURCE1:TTL:CURRENT"

    PULSE_TIME_RANGE_S: Tuple[float, float] = (1e-6, 10.0)
    """Instrument limits on pulse ON and OFF time: 0.001 ms to 10 s."""
    PWM_DUTY_RANGE_PCT: Tuple[float, float] = (0.1, 99.9)
    """Instrument limits on PWM duty cycle. The 0.1 % floor is why pulse mode exists."""
    PWM_FREQ_RANGE_HZ: Tuple[float, float] = (0.1, 20e3)
    """Instrument limits on PWM modulation frequency."""
    MAX_COUNT: int = 1000
    """Maximum finite pulse count. ``0`` means infinite, in both PWM and pulse mode."""

    def __init__(
        self,
        resource: Optional[str] = None,
        *,
        timeout_ms: int = 5000,
        backend: Optional[str] = None,
        transcript_path: Optional[str] = None,
        trigger_port: Optional[int] = None,
    ) -> None:
        self._current_limit: Optional[float] = None
        super().__init__(
            resource,
            timeout_ms=timeout_ms,
            backend=backend,
            transcript_path=transcript_path,
            trigger_port=trigger_port,
        )

    @classmethod
    def env_resource(cls) -> Optional[str]:
        """Return the resource configured via ``DAQ_LED_RESOURCE``.

        :returns: The configured resource string, or ``None``.

        """
        return get_led_resource()

    @classmethod
    def env_trigger_port(cls) -> Optional[int]:
        """Return the trigger port configured via ``DAQ_LED_TRIGGER_PORT``.

        :returns: The configured port number, or ``None``.

        """
        return get_led_trigger_port()

    # ------------------------------------------------------------------ output state

    @property
    def current_limit(self) -> float:
        """The instrument's configured current limit in amperes.

        Queried once and cached, since it is a front-panel setting that does not change
        during a run.

        :returns: The current limit in amperes.

        """
        if self._current_limit is None:
            self._current_limit = self.query_float("SOURCE1:CURRENT:LIMIT?")
        return self._current_limit

    @property
    def output(self) -> bool:
        """Whether the LED output is enabled.

        :returns: ``True`` when the output is on.

        """
        return self.query("OUTPUT1:STATE?").strip() in ("1", "ON")

    @output.setter
    def output(self, enabled: bool) -> None:
        """Enable or disable the LED output.

        :param enabled: Desired output state.

        """
        self.write("OUTPUT1:STATE ON" if enabled else "OUTPUT1:STATE OFF")

    @property
    def measured_current(self) -> float:
        """The current the instrument is presently measuring, in amperes.

        :returns: The measured LED current in amperes.

        """
        return self.query_float("SENSE3:CURRENT:DATA?")

    @property
    def terminal(self) -> int:
        """Which output terminal is selected: ``1`` (10 A, 12-pin) or ``2`` (2 A, 4-pin).

        The DC2200 has two LED connectors and every source setting applies to the selected
        one. Left alone, the driver inherits whatever the front panel last selected, so read
        this back rather than assuming.

        :returns: The selected terminal number.

        """
        return int(self.query_float("OUTPUT1:TERMINAL?"))

    @terminal.setter
    def terminal(self, number: int) -> None:
        """Select the output terminal.

        :param number: ``1`` for the 10 A 12-pin connector, ``2`` for the 2 A 4-pin connector.
        :raises ValueError: If *number* is not 1 or 2.

        """
        if number not in (1, 2):
            raise ValueError(f"terminal must be 1 or 2, got {number}")
        self.write(f"OUTPUT1:TERMINAL {int(number)}")

    def protection_status(self) -> Dict[str, bool]:
        """Return the instrument's protection flags.

        Worth checking when the LED does not light or output is unexpectedly disabled: any of
        these tripping shuts the output down independently of the settings.

        :returns: Mapping of protection name to whether it has tripped.

        """
        return {
            "current_limit": bool(self.query_float("SOURCE1:CURRENT:LIMIT:TRIPPED?")),
            "interlock": bool(self.query_float("OUTPUT1:PROTECTION:INTLOCK:TRIPPED?")),
            "driver_over_temp": bool(
                self.query_float("OUTPUT1:PROTECTION:TEMPERATURE:DRIVER:TRIPPED?")
            ),
            "head_over_temp": bool(
                self.query_float("OUTPUT1:PROTECTION:TEMPERATURE:HEAD:TRIPPED?")
            ),
        }

    def safe_state(self) -> None:
        """Turn the LED off."""
        self.output = False

    # ------------------------------------------------------------------ validation

    def _check_current(self, current_a: float) -> None:
        """Validate a requested drive current against the instrument's current limit.

        :param current_a: Requested current in amperes.
        :raises ValueError: If the current is not strictly between zero and the limit.

        """
        limit = self.current_limit
        if not 0.0 < current_a < limit:
            raise ValueError(
                f"current_a must satisfy 0 < I < {limit} A (the DC2200 current limit), "
                f"got {current_a}"
            )

    # ------------------------------------------------------------------ output modes

    def configure_cc(self, current_a: float, *, output: bool = False) -> None:
        """Configure constant-current mode.

        :param current_a: Drive current in amperes.
        :param output: Whether to enable the output afterwards.
        :raises ValueError: If *current_a* exceeds the instrument's current limit.

        """
        self._check_current(current_a)
        self.write("SOURCE1:MODE CC")
        self.write(f"SOURCE1:CCURENT:CURRENT {current_a}")
        self.output = output

    def configure_ttl(self, current_a: float, *, output: bool = False) -> None:
        """Configure TTL modulation: the LED follows the rear-panel modulation input.

        "TTL" is the ordinary digital-logic sense -- transistor-transistor logic, a 0 V-low /
        ~5 V-high on/off signal. In this mode the LED drives at *current_a* while the SMA
        modulation input on the rear panel is high, and is off while it is low. Driving that
        input from the Presto's trigger output is how illumination is synchronised to an
        acquisition -- see ``daq/instruments/README.md`` for the wiring and the timing.

        The input takes TTL levels -- low 0-0.8 V, high 2.0-5.0 V -- into 10 kOhm. (The
        "0-5 V, 250 kHz" figure sometimes quoted for this connector is External Modulation
        mode's analog small-signal bandwidth, a different mode.)

        Note this yields an illumination **window** spanning the acquisition, not a pulse
        inside it: the Presto re-asserts its trigger every lock-in window, so the line is high
        from the first sample to the last. For a short flash see
        :meth:`configure_pulse` (instrument-timed, but not synchronised) or presto's
        ``Pulsed.output_digital_marker`` (synchronised, but a different measurement mode that
        this library does not wrap).

        Pair this with ``TimeStream(external_trigger=trigger_for(led))``, which asserts
        whichever port :attr:`~daq.instruments._visa.VisaInstrument.trigger_port` says the
        modulation input is wired to -- **digital output port 2** in the lab's default wiring,
        overridable per instrument or through ``DAQ_LED_TRIGGER_PORT``. Naming the port by
        hand is what goes wrong here: a plain ``external_trigger=True`` gates port 1 (the
        function generator) and leaves the LED dark for the whole run.

        Configuring the mode does **not** arm the LED. Enabling the output (``output=True``
        here, or ``led.output = True`` later) arms the output stage, after which the LED
        follows the modulation input. Whether it illuminates the instant you arm it therefore
        depends on the level the trigger line happens to be sitting at, so arm immediately
        before the acquisition rather than during setup -- ``TimeStream.run()`` spends a
        while connecting, configuring and tuning before it acquires, and an armed LED would
        be lit for all of it if the line idles high.

        :param current_a: Current applied while the modulation input is high, in amperes.
        :param output: Whether to arm the output afterwards. Defaults to ``False`` so that
            configuring never illuminates the LED as a side effect.
        :raises ValueError: If *current_a* exceeds the instrument's current limit.
        :raises InstrumentError: If the instrument rejects every candidate current header,
            in which case check the SCPI section of the DC2200 manual for your firmware.

        """
        self._check_current(current_a)
        self.write("SOURCE1:MODE TTL")
        self.write(f"{self.TTL_CURRENT_HEADER} {current_a}")
        self.output = output

    def configure_pulse(
        self,
        on_time_s: float,
        *,
        off_time_s: Optional[float] = None,
        freq_hz: Optional[float] = None,
        period_s: Optional[float] = None,
        brightness_pct: Optional[float] = None,
        current_a: Optional[float] = None,
        count: int = 1,
        output: bool = False,
    ) -> None:
        """Configure pulse mode: an explicit ON time and OFF time.

        Unlike :meth:`configure_pwm`, which takes a duty cycle and so couples pulse width to
        repetition rate, pulse mode sets the two times independently. That is what makes a
        short pulse at a slow rate possible -- PWM's 0.1 % duty floor puts its narrowest pulse
        at ``0.001 / freq_hz`` (10 us only at 100 Hz), whereas pulse mode reaches 1 us at any
        rate. ON and OFF time each span 0.001 ms to 10 s, giving 0.05 Hz to 500 Hz overall.

        Give the OFF time directly, or as a repetition rate via *freq_hz* / *period_s*, in
        which case ``off_time = period - on_time``.

        **Pulse mode is brightness-based, not current-based.** The instrument takes a
        percentage of the *currently configured current limit*, so 100 % means driving at the
        limit. Pass *brightness_pct* directly, or pass *current_a* to have it converted using
        the queried limit. Note the consequence: with a high limit, a small absolute current
        is a very small percentage, and the instrument's brightness resolution may not reach
        it -- lower the LED current limit to gain fine control near the bottom of the range.

        :param on_time_s: Pulse ON time in seconds, e.g. ``10e-6``.
        :param off_time_s: Pulse OFF time in seconds.
        :param freq_hz: Repetition frequency, used to derive the OFF time.
        :param period_s: Repetition period, used to derive the OFF time.
        :param brightness_pct: Pulse amplitude as a percentage of the current limit.
        :param current_a: Pulse amplitude in amperes, converted to a percentage of the limit.
        :param count: Number of pulses; ``0`` means run indefinitely until the output is
            disabled, which is what you usually want spanning an acquisition.
        :param output: Whether to enable the output afterwards, which starts the train.
        :raises ValueError: If arguments are missing, inconsistent, or outside the
            instrument's documented ranges.

        """
        rate_given = [x is not None for x in (off_time_s, freq_hz, period_s)]
        if sum(rate_given) != 1:
            raise ValueError("Specify exactly one of off_time_s, freq_hz or period_s")
        if freq_hz is not None:
            if freq_hz <= 0:
                raise ValueError(f"freq_hz must be positive, got {freq_hz}")
            period_s = 1.0 / freq_hz
        if period_s is not None:
            if period_s <= on_time_s:
                raise ValueError(
                    f"the repetition period {period_s} s must exceed on_time_s={on_time_s} s"
                )
            off_time_s = period_s - on_time_s

        if (brightness_pct is None) == (current_a is None):
            raise ValueError("Specify exactly one of brightness_pct or current_a")
        if current_a is not None:
            self._check_current(current_a)
            brightness_pct = current_a / self.current_limit * 100.0
        if not 0.0 < brightness_pct <= 100.0:
            raise ValueError(f"brightness_pct must satisfy 0 < B <= 100, got {brightness_pct}")

        low, high = self.PULSE_TIME_RANGE_S
        for name, value in (("on_time_s", on_time_s), ("off_time_s", off_time_s)):
            if not low <= value <= high:
                raise ValueError(
                    f"{name}={value} s is outside the DC2200's pulse timing range "
                    f"{low} s to {high} s"
                )
        if int(count) != count or count < 0 or count > self.MAX_COUNT:
            raise ValueError(
                f"count must be 0 (infinite) or a whole number from 1 to "
                f"{self.MAX_COUNT}, got {count}"
            )

        self.write(self.PULSE_MODE_COMMAND)
        self.write(f"{self.PULSE_BRIGHTNESS_HEADER} {brightness_pct}")
        self.write(f"{self.PULSE_ONTIME_HEADER} {on_time_s}")
        self.write(f"{self.PULSE_OFFTIME_HEADER} {off_time_s}")
        self.write(f"{self.PULSE_COUNT_HEADER} {int(count)}")
        self.output = output

    def configure_pwm(
        self,
        current_a: float,
        freq_hz: float,
        duty_pct: float,
        count: int,
        *,
        output: bool = False,
    ) -> None:
        """Configure a pulse-width-modulated pulse train.

        :param current_a: Peak drive current in amperes.
        :param freq_hz: Pulse repetition frequency in hertz.
        :param duty_pct: Duty cycle in percent, strictly between 0 and 100.
        :param count: Number of pulses to emit; must be a positive whole number.
        :param output: Whether to enable the output afterwards.
        :raises ValueError: If any parameter is out of range.

        """
        self._check_current(current_a)
        f_low, f_high = self.PWM_FREQ_RANGE_HZ
        if not f_low <= freq_hz <= f_high:
            raise ValueError(
                f"freq_hz={freq_hz} is outside the DC2200's PWM range {f_low} Hz to {f_high} Hz"
            )
        d_low, d_high = self.PWM_DUTY_RANGE_PCT
        if not d_low <= duty_pct <= d_high:
            raise ValueError(
                f"duty_pct={duty_pct} is outside the DC2200's range {d_low} % to {d_high} %. "
                f"Note the {d_low} % floor caps the narrowest PWM pulse at "
                f"{d_low / 100.0 / freq_hz * 1e6:.1f} us at {freq_hz} Hz; use configure_pulse() "
                f"for a shorter pulse at this rate."
            )
        if int(count) != count or count < 0 or count > self.MAX_COUNT:
            raise ValueError(
                f"count must be 0 (infinite) or a whole number from 1 to "
                f"{self.MAX_COUNT}, got {count}"
            )

        self.write("SOURCE1:MODE PWM")
        self.write(f"SOURCE1:PWM:CURRENT {current_a}")
        # FREQ, not FREQUENCY -- the long form is rejected with -113 Undefined header.
        self.write(f"SOURCE1:PWM:FREQ {freq_hz}")
        self.write(f"SOURCE1:PWM:DCYCLE {duty_pct}")
        self.write(f"SOURCE1:PWM:COUNT {int(count)}")
        self.output = output

    def pwm_train(
        self,
        current_a: float,
        freq_hz: float,
        duty_pct: float,
        count: int,
        *,
        settle_s: float = 0.5,
    ) -> None:
        """Configure and emit one **PWM** pulse train, blocking until it has finished.

        Named for the mode it uses: this is :meth:`configure_pwm` plus the wait, so it
        inherits PWM's 0.1 %% duty floor and *cannot* produce a short pulse at a slow
        rate. For that, use :meth:`configure_pulse`.

        Loads the PWM settings, enables the output, sleeps for the train's own duration
        (``count / freq_hz``) plus *settle_s*, then disables the output again. The DC2200's
        PWM engine generates the pulses and stops itself after *count* of them, so pulse
        timing is instrument-accurate; only the moment the train *starts* is software-timed,
        set by a USB write, with millisecond-scale jitter.

        For example ``pwm_train(current_a=0.01, freq_hz=100, duty_pct=50, count=10)`` emits
        ten 10 ms periods -- 5 ms on at 10 mA, 5 ms off -- so a 100 ms train carrying 50 ms of
        total LED-on time, and the call blocks for 0.6 s (0.1 s of train plus the 0.5 s
        settle margin).

        **This is not synchronised with anything.** Because the start is software-timed, it
        cannot be placed reliably within a :class:`~daq.measurements.timestream.TimeStream`
        acquisition. :meth:`configure_ttl` is what locks the LED to a time stream, but note it
        gives a steady illumination window spanning the acquisition rather than pulses within
        it -- the two modes are not interchangeable. Choose PWM when you need real pulses and
        can tolerate an unknown offset, TTL when you need the light to line up with the data.

        The output is turned off in a ``finally``, so it is disabled even if the wait is
        interrupted. The instrument is left in PWM mode with these settings, so a subsequent
        :meth:`settings` call (or :meth:`daq._base.Base.attach`) reports them.

        :param current_a: Peak drive current in amperes, applied during the on-phase of each
            pulse.
        :param freq_hz: Pulse repetition frequency in hertz; each period lasts ``1 / freq_hz``.
        :param duty_pct: Duty cycle in percent -- the fraction of each period the LED is on.
        :param count: Number of pulses to emit. The instrument stops after this many; very
            large counts are firmware-limited (100000 on firmware 1.3.0 and later).
        :param settle_s: Extra delay added to the computed train duration before switching the
            output off, so a slightly slow instrument is not cut off mid-train.
        :raises ValueError: If any parameter is out of range.

        """
        self.configure_pwm(current_a, freq_hz, duty_pct, count, output=False)
        self.output = True
        try:
            time.sleep(int(count) / freq_hz + settle_s)
        finally:
            self.output = False

    # ------------------------------------------------------------------ metadata

    def settings(self) -> Dict[str, Any]:
        """Return the driver's current state, read back from the instrument.

        Mode-specific fields are included only for the mode that is actually selected.

        :returns: Flat mapping of setting name to scalar value.

        """
        state: Dict[str, Any] = super().settings()
        mode = self.query("SOURCE1:MODE?").strip().upper()
        state["mode"] = mode
        state["current_limit_a"] = self.current_limit
        state["output"] = self.output
        if mode.startswith("PWM"):
            state["pwm_current_a"] = self.query_float("SOURCE1:PWM:CURRENT?")
            state["pwm_freq_hz"] = self.query_float("SOURCE1:PWM:FREQ?")
            state["pwm_duty_pct"] = self.query_float("SOURCE1:PWM:DCYCLE?")
            state["pwm_count"] = int(self.query_float("SOURCE1:PWM:COUNT?"))
        elif mode.startswith("CC"):
            state["cc_current_a"] = self.query_float("SOURCE1:CCURENT:CURRENT?")
        elif mode.startswith("TTL"):
            state["ttl_current_a"] = self.query_float(f"{self.TTL_CURRENT_HEADER}?")
        elif mode.startswith("PULS"):
            state["pulse_brightness_pct"] = self.query_float(f"{self.PULSE_BRIGHTNESS_HEADER}?")
            on_s = self.query_float(f"{self.PULSE_ONTIME_HEADER}?")
            off_s = self.query_float(f"{self.PULSE_OFFTIME_HEADER}?")
            state["pulse_on_time_s"] = on_s
            state["pulse_off_time_s"] = off_s
            state["pulse_count"] = int(self.query_float(f"{self.PULSE_COUNT_HEADER}?"))
            period = on_s + off_s
            if period > 0:
                state["pulse_freq_hz"] = 1.0 / period
            # Absolute amplitude is only meaningful against the limit the percentage refers to.
            state["pulse_current_a"] = state["pulse_brightness_pct"] / 100.0 * self.current_limit
        return state
