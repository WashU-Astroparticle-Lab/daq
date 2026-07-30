# -*- coding: utf-8 -*-
"""Function-generator drivers used as detector bias sources."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..config import get_fgen_resource
from ._visa import InstrumentError, VisaInstrument


class Agilent33220A(VisaInstrument):
    """Agilent/Keysight 33220A arbitrary waveform generator, used as a gate-bias source.

    Two output modes cover the parity/quasiparticle-tunnelling measurements:

    - :meth:`constant` -- a fixed DC bias.
    - :meth:`sawtooth` -- a repeating voltage ramp, normally *gated* so that it runs only
      while the Presto asserts its trigger output (``TimeStream(external_trigger=True)``,
      i.e. digital output port 1, where this generator's gate input is wired).

    Both setters are **hermetic**: each writes the full state its mode depends on, including
    explicitly disabling burst mode. This matters because the 33220A does not support burst
    with a DC carrier, so a ``FUNC DC`` written on top of a leftover ``BURS:STAT ON`` from an
    earlier ramp puts the instrument into an invalid combination. Bench scripts that set only
    the parameters they care about hit this whenever a constant-bias run follows a ramp.

    :param resource: Explicit VISA resource string. When ``None``, uses ``DAQ_FGEN_RESOURCE``
        and otherwise autodiscovers by ``*IDN?``.
    :param load: Output-load setting, ``"INF"`` for high impedance (the lab default) or a
        resistance in ohms such as ``"50"``. Fixes the minimum programmable amplitude.
    :param timeout_ms: VISA I/O timeout in milliseconds.
    :param backend: PyVISA backend spec. Defaults to ``DAQ_VISA_BACKEND``.
    :param transcript_path: Optional path to a SCPI transcript file.

    """

    IDN_KEYWORDS = ("33220A",)
    RESOURCE_HINTS = ("0x0957::0x0407",)
    """Agilent USB vendor/product ID for the 33220A.

    Narrows autodiscovery to the likely resource so unrelated instruments are not opened and
    interrogated. Purely an optimisation: if no visible resource matches the hint -- the unit
    is on GPIB, say -- discovery falls back to probing everything.

    """
    ENV_VAR = "DAQ_FGEN_RESOURCE"

    MIN_VPP_BY_LOAD: Dict[str, float] = {"INF": 0.02, "50": 0.01}
    """Minimum programmable peak-to-peak amplitude (V), per output-load setting."""
    DEFAULT_SETTLE_S: float = 0.5
    """Delay after reconfiguring, matching the settling time used on the bench."""

    def __init__(
        self,
        resource: Optional[str] = None,
        *,
        load: str = "INF",
        timeout_ms: int = 5000,
        backend: Optional[str] = None,
        transcript_path: Optional[str] = None,
    ) -> None:
        self.load = str(load).upper()
        # Last-configured ramp parameters, kept so samples_for_periods() and settings() can
        # report the ramp without re-querying every field.
        self.ramp_freq_hz: Optional[float] = None
        super().__init__(
            resource,
            timeout_ms=timeout_ms,
            backend=backend,
            transcript_path=transcript_path,
        )

    @classmethod
    def env_resource(cls) -> Optional[str]:
        """Return the resource configured via ``DAQ_FGEN_RESOURCE``.

        :returns: The configured resource string, or ``None``.

        """
        return get_fgen_resource()

    # ------------------------------------------------------------------ output state

    @property
    def output(self) -> bool:
        """Whether the front-panel output is enabled.

        :returns: ``True`` when the output is on.

        """
        return self.query("OUTP?").strip() in ("1", "ON")

    @output.setter
    def output(self, enabled: bool) -> None:
        """Enable or disable the front-panel output.

        :param enabled: Desired output state.

        """
        self.write("OUTP ON" if enabled else "OUTP OFF")

    def safe_state(self) -> None:
        """Turn the output off, so no bias is left applied to the device."""
        self.output = False

    # ------------------------------------------------------------------ validation

    def _min_vpp(self) -> float:
        """Return the minimum programmable amplitude for the configured output load.

        :returns: Minimum peak-to-peak amplitude in volts.

        """
        return self.MIN_VPP_BY_LOAD.get(self.load, 0.02)

    def _check_vpp(self, vpp: float) -> None:
        """Validate a requested peak-to-peak amplitude against the instrument's limits.

        :param vpp: Requested peak-to-peak amplitude in volts.
        :raises ValueError: If *vpp* is not positive, or is below what the 33220A can
            program into the configured load (it would silently output a different
            amplitude than requested).

        """
        if vpp <= 0:
            raise ValueError(f"vpp must be positive, got {vpp}")
        minimum = self._min_vpp()
        if vpp < minimum:
            raise ValueError(
                f"vpp={vpp} V is below the 33220A minimum of {minimum} V into a "
                f"{self.load} load; the instrument cannot program this amplitude."
            )

    # ------------------------------------------------------------------ output modes

    def constant(
        self,
        offset_v: float,
        *,
        output: bool = True,
        settle_s: Optional[float] = None,
    ) -> None:
        """Apply a constant DC bias.

        Burst mode is turned off first, before the carrier is switched to DC -- the 33220A
        rejects a DC carrier while burst is enabled.

        :param offset_v: DC offset in volts.
        :param output: Whether to enable the output afterwards.
        :param settle_s: Delay after configuring. Defaults to :attr:`DEFAULT_SETTLE_S`.

        """
        self.write("BURS:STAT OFF")
        self.write("FUNC DC")
        self.write(f"VOLT:OFFS {offset_v}")
        self.write(f"OUTP:LOAD {self.load}")
        self.ramp_freq_hz = None
        self.output = output
        time.sleep(self.DEFAULT_SETTLE_S if settle_s is None else settle_s)

    def sawtooth(
        self,
        vpp: float,
        freq_hz: float,
        *,
        symmetry_pct: float = 100.0,
        offset_v: Optional[float] = None,
        gated: bool = True,
        phase_deg: float = 180.0,
        gate_polarity: str = "NORM",
        output: bool = True,
        settle_s: Optional[float] = None,
    ) -> None:
        """Configure a repeating voltage ramp.

        With ``gated=True`` (the default) the generator runs in gated-burst mode and emits the
        ramp only while its external trigger input is asserted, which is how the Presto
        synchronises the bias to an acquisition -- pair it with
        ``TimeStream(external_trigger=True)``, which asserts digital output port 1 for the
        duration of the acquisition. Pass a per-port states list instead if the gate input is
        on another port. With ``gated=False`` the ramp free-runs.

        :param vpp: Peak-to-peak amplitude in volts.
        :param freq_hz: Ramp repetition frequency in hertz.
        :param symmetry_pct: Ramp symmetry in percent; ``100`` gives a ramp-up sawtooth.
        :param offset_v: DC offset in volts. Defaults to ``vpp / 2``, making the ramp
            unipolar-positive (the lab convention).
        :param gated: Whether to gate the ramp on the external trigger input.
        :param phase_deg: Burst start phase in degrees.
        :param gate_polarity: ``"NORM"`` to output while the gate is high, ``"INV"`` while low.
        :param output: Whether to enable the output afterwards.
        :param settle_s: Delay after configuring. Defaults to :attr:`DEFAULT_SETTLE_S`.
        :raises ValueError: If *vpp*, *freq_hz* or *symmetry_pct* is out of range.

        """
        self._check_vpp(vpp)
        if freq_hz <= 0:
            raise ValueError(f"freq_hz must be positive, got {freq_hz}")
        if not 0.0 <= symmetry_pct <= 100.0:
            raise ValueError(f"symmetry_pct must be between 0 and 100, got {symmetry_pct}")
        if offset_v is None:
            offset_v = vpp / 2.0

        self.write("FUNC RAMP")
        self.write(f"FREQ {freq_hz}")
        self.write(f"VOLT {vpp}")
        self.write(f"VOLT:OFFS {offset_v}")
        self.write(f"FUNC:RAMP:SYMM {symmetry_pct}")
        if gated:
            self.write("BURS:MODE GAT")
            self.write(f"BURS:GATE:POL {gate_polarity}")
            self.write(f"BURS:PHAS {phase_deg}")
            self.write("BURS:STAT ON")
        else:
            self.write("BURS:STAT OFF")
        self.write(f"OUTP:LOAD {self.load}")
        self.ramp_freq_hz = float(freq_hz)
        self.output = output
        time.sleep(self.DEFAULT_SETTLE_S if settle_s is None else settle_s)

    # ------------------------------------------------------------------ helpers

    def samples_for_periods(
        self,
        n_periods: int,
        sample_rate: float,
        *,
        freq_hz: Optional[float] = None,
        discard_ms: float = 25.0,
    ) -> int:
        """Return the ``pixel_counts`` covering *n_periods* whole ramp periods.

        A time stream that is block-averaged per ramp period must span a whole number of
        periods, and must additionally cover the samples that :class:`~daq.measurements.
        timestream.TimeStream` discards at the start of the acquisition. Deriving that count
        here keeps the generator and the acquisition from drifting apart::

            fgen.sawtooth(vpp=2.0, freq_hz=500)
            ts = TimeStream(..., df=5e4, pixel_counts=fgen.samples_for_periods(200, 5e4))

        :param n_periods: Number of whole ramp periods to acquire.
        :param sample_rate: Time-stream sample rate in hertz (``TimeStream.df``).
        :param freq_hz: Ramp frequency. Defaults to the frequency of the last
            :meth:`sawtooth` call.
        :param discard_ms: Leading milliseconds the time stream will discard; must match
            ``TimeStream(discard_start_ms=...)``.
        :raises ValueError: If *n_periods* or *sample_rate* is not positive.
        :raises InstrumentError: If no ramp frequency is known.
        :returns: The number of samples to request.

        """
        if n_periods <= 0:
            raise ValueError(f"n_periods must be positive, got {n_periods}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        ramp_freq = self.ramp_freq_hz if freq_hz is None else freq_hz
        if not ramp_freq:
            raise InstrumentError(
                "No ramp frequency is known; call sawtooth() first or pass freq_hz explicitly."
            )
        samples_to_discard = int(round(discard_ms * 1e-3 * sample_rate))
        samples_per_period = int(sample_rate / ramp_freq)
        if samples_per_period < 1:
            raise ValueError(
                f"sample_rate={sample_rate} Hz gives fewer than one sample per "
                f"{ramp_freq} Hz ramp period; increase the sample rate or slow the ramp."
            )
        return samples_to_discard + samples_per_period * int(n_periods)

    # ------------------------------------------------------------------ metadata

    def settings(self) -> Dict[str, Any]:
        """Return the generator's current state, read back from the instrument.

        Ramp-specific fields are included only when the carrier is actually a ramp.

        :returns: Flat mapping of setting name to scalar value.

        """
        state: Dict[str, Any] = super().settings()
        function = self.query("FUNC?").strip().upper()
        state["function"] = function
        state["offset_v"] = self.query_float("VOLT:OFFS?")
        state["load"] = self.query("OUTP:LOAD?").strip()
        state["output"] = self.output
        if function.startswith("RAMP"):
            state["vpp"] = self.query_float("VOLT?")
            state["freq_hz"] = self.query_float("FREQ?")
            state["symmetry_pct"] = self.query_float("FUNC:RAMP:SYMM?")
            burst_on = self.query("BURS:STAT?").strip() in ("1", "ON")
            state["burst"] = burst_on
            if burst_on:
                state["burst_mode"] = self.query("BURS:MODE?").strip()
                state["burst_phase_deg"] = self.query_float("BURS:PHAS?")
                state["gate_polarity"] = self.query("BURS:GATE:POL?").strip()
        return state
