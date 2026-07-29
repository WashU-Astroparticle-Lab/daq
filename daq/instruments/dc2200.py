# -*- coding: utf-8 -*-
"""Thorlabs DC2200 LED driver."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..config import get_led_resource
from ._visa import InstrumentError, VisaInstrument


class DC2200(VisaInstrument):
    """Thorlabs DC2200 LED driver, controlled over USB with SCPI.

    Supports the two modes used on the bench:

    - :meth:`configure_cc` -- constant current.
    - :meth:`configure_pwm` / :meth:`pulse_train` -- a pulse-width-modulated burst of a given
      current, frequency, duty cycle and pulse count.

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

    """

    IDN_KEYWORDS = ("DC2200",)
    RESOURCE_HINTS = ("0x1313::0x80C8",)
    ENV_VAR = "DAQ_LED_RESOURCE"

    TTL_CURRENT_HEADERS = ("SOURCE1:TTL:CURRENT", "SOURCE1:CCURENT:CURRENT")
    """Candidate headers for the TTL-mode drive current, tried in order.

    Thorlabs documents TTL mode as having exactly one settable parameter -- the current
    applied while the modulation input is high -- but does not publish the SCPI header for it.
    :meth:`configure_ttl` tries these against the instrument and keeps whichever it accepts,
    rather than hardcoding a guess. Rejected headers raise ``-113 Undefined header``, so this
    resolves deterministically on the first call and cannot silently drive the wrong value.

    """

    def __init__(
        self,
        resource: Optional[str] = None,
        *,
        timeout_ms: int = 5000,
        backend: Optional[str] = None,
        transcript_path: Optional[str] = None,
    ) -> None:
        self._current_limit: Optional[float] = None
        self._ttl_current_header: Optional[str] = None
        super().__init__(
            resource,
            timeout_ms=timeout_ms,
            backend=backend,
            transcript_path=transcript_path,
        )

    @classmethod
    def env_resource(cls) -> Optional[str]:
        """Return the resource configured via ``DAQ_LED_RESOURCE``.

        :returns: The configured resource string, or ``None``.

        """
        return get_led_resource()

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

        In this mode the LED drives at *current_a* while the SMA modulation input on the rear
        panel is high, and is off while it is low. Driving that input from the Presto's
        trigger output is how an LED pulse is synchronised to an acquisition -- see
        ``daq/instruments/README.md`` for the wiring and the timing.

        The input accepts 0-5 V at up to 250 kHz.

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

        if self._ttl_current_header is not None:
            self.write(f"{self._ttl_current_header} {current_a}")
        else:
            rejected = []
            for header in self.TTL_CURRENT_HEADERS:
                try:
                    self.write(f"{header} {current_a}")
                except InstrumentError as exc:
                    rejected.append(f"{header} ({exc})")
                    continue
                self._ttl_current_header = header
                break
            else:
                raise InstrumentError(
                    "Could not set the TTL-mode drive current: the DC2200 rejected every "
                    f"candidate header. Tried: {'; '.join(rejected)}. Check the SCPI section "
                    "of the DC2200 manual for this firmware and add the correct header to "
                    "DC2200.TTL_CURRENT_HEADERS."
                )

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
        if freq_hz <= 0:
            raise ValueError(f"freq_hz must be positive, got {freq_hz}")
        if not 0.0 < duty_pct < 100.0:
            raise ValueError(f"duty_pct must satisfy 0 < D < 100, got {duty_pct}")
        if int(count) != count or count <= 0:
            raise ValueError(f"count must be a positive whole number, got {count}")

        self.write("SOURCE1:MODE PWM")
        self.write(f"SOURCE1:PWM:CURRENT {current_a}")
        # FREQ, not FREQUENCY -- the long form is rejected with -113 Undefined header.
        self.write(f"SOURCE1:PWM:FREQ {freq_hz}")
        self.write(f"SOURCE1:PWM:DCYCLE {duty_pct}")
        self.write(f"SOURCE1:PWM:COUNT {int(count)}")
        self.output = output

    def pulse_train(
        self,
        current_a: float,
        freq_hz: float,
        duty_pct: float,
        count: int,
        *,
        settle_s: float = 0.5,
    ) -> None:
        """Configure and emit one PWM pulse train, blocking until it has finished.

        Loads the PWM settings, enables the output, sleeps for the train's own duration
        (``count / freq_hz``) plus *settle_s*, then disables the output again. The DC2200's
        PWM engine generates the pulses and stops itself after *count* of them, so pulse
        timing is instrument-accurate; only the moment the train *starts* is software-timed,
        set by a USB write, with millisecond-scale jitter.

        For example ``pulse_train(current_a=0.01, freq_hz=100, duty_pct=50, count=10)`` emits
        ten 10 ms periods -- 5 ms on at 10 mA, 5 ms off -- so a 100 ms train carrying 50 ms of
        total LED-on time, and the call blocks for 0.6 s (0.1 s of train plus the 0.5 s
        settle margin).

        **This is not synchronised with anything.** Because the start is software-timed, it
        cannot be placed reliably within a :class:`~daq.measurements.timestream.TimeStream`
        acquisition. To pulse the LED at a known point in a time stream, use
        :meth:`configure_ttl` and gate it from the Presto trigger output instead.

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
        elif mode.startswith("TTL") and self._ttl_current_header is not None:
            state["ttl_current_a"] = self.query_float(f"{self._ttl_current_header}?")
        return state
