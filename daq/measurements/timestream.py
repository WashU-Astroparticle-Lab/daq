# -*- coding: utf-8 -*-
"""
TimeStream measurement class for acquiring time-domain data with multiple frequencies.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt

from presto import lockin
from presto.utils import untwist_downconversion

from .._base import Base
from ..config import get_presto_address, get_presto_port
from ..triggers import (
    MAX_TRIGGER_PORTS,
    TriggerAny,
    describe_trigger_states,
    resolve_trigger_states,
)

FloatAny = Union[float, List[float], npt.NDArray[np.floating]]
BoolAny = Union[bool, List[bool], npt.NDArray[np.bool_]]

#: Suffix of the attribute :meth:`daq._base.Base.attach` writes for a function generator's
#: carrier waveform, e.g. ``bias_function``. Its prefix names the attachment.
BIAS_FUNCTION_SUFFIX = "_function"

#: Attachment prefix preferred when more than one instrument reports a carrier waveform.
#: ``QCTrace`` and ``BiasHunt`` both attach their generator as ``bias``.
DEFAULT_BIAS_PREFIX = "bias"

#: Reconstructions :meth:`TimeStream.analyze` can run, beyond ``"auto"``.
ANALYZE_MODES = ("raw", "sawtooth", "constant")

#: Projection of the complex readout the parity reconstructions run on by default. The
#: maximal-separation axis: parity moves the resonator between two points of the IQ plane, and
#: a fixed axis sees only the ``cos(angle)`` of the line joining them -- for a separation that
#: is mostly a phase shift, almost none of it. See :data:`daq.analysis.parity.PROJECTIONS`.
DEFAULT_QUANTITY = "proj"

#: Axis labels for the projections of the complex readout the parity spectrum can be taken of.
_QUANTITY_LABELS = {
    "proj": r"$S_\parallel$",
    "abs": r"$|S|$",
    "real": r"$\mathrm{Re}(S)$",
    "imag": r"$\mathrm{Im}(S)$",
}

#: Most individual switching events to draw as tick marks on a reconstructed trace. Past this
#: the marks merge into a solid band that hides the trace they annotate; the reconstructed
#: level curve still shows where the flips are, and the count is reported either way.
_MAX_FLIP_MARKS = 400

__all__ = [
    "ANALYZE_MODES",
    "BIAS_FUNCTION_SUFFIX",
    "BoolAny",
    "DEFAULT_BIAS_PREFIX",
    "DEFAULT_QUANTITY",
    "FloatAny",
    "MAX_TRIGGER_PORTS",
    "TimeStream",
    "TriggerAny",
    "resolve_trigger_states",
]


def _as_text(value: Any) -> str:
    """Render an HDF5-round-tripped scalar as text.

    h5py hands back ``bytes`` for some string vintages and ``numpy.str_`` for others, so a
    bare ``str()`` is not enough to compare a stored setting against a literal.

    :param value: The value to render.
    :returns: The value as a plain string.

    """
    return value.decode() if isinstance(value, bytes) else str(value)


class TimeStream(Base):
    _CONSTRUCTOR_ATTRS = frozenset({
        "lo_freq",
        "df",
        "pixel_counts",
        "output_port",
        "input_port",
        "dither",
        "discard_start_ms",
        "external_trigger",
    })
    """HDF5 attributes :meth:`load` consumes itself, and must not restore a second time.

    Everything *else* in a saved file's attributes belongs to something other than the
    acquisition parameters -- the instrument settings :meth:`~daq._base.Base.attach` flattened
    on, and the ``device``/``filter``/``notes`` metadata -- and :meth:`load` copies it back
    verbatim. Listing what is consumed rather than what is restored is deliberate: a new
    instrument driver, or a new ``settings()`` key on an existing one, then round-trips
    without an edit here.
    """

    TRIGGER_WIDTH_S: float = 0.03
    """s -- trigger-high duration handed to presto's ``set_trigger_out``.

    presto re-asserts the trigger at the start of **every** lock-in window, i.e. every
    ``1 / df``. This width is far longer than any lock-in window used here (20 μs at
    ``df = 50 kHz``), so the next rising edge always arrives before the falling edge is due
    and the line stays **high for the whole acquisition** -- which is what a gated bias ramp
    or a TTL-driven LED needs. It is not a one-shot pulse of this length. A width *below*
    ``1 / df`` would instead chop the line into a pulse train at the sample rate; see
    ``daq/instruments/README.md`` for why that is almost never what you want. The value also
    sits just under presto's 24-bit ceiling, ``(2**24 - 1) * CLK_T`` (33.5 ms at 2 ns).

    **Inferred, not scope-verified.** The continuous-high behaviour follows from the presto
    Python layer -- the per-window ``(start, stop)`` clock pair sent with ``df``, and the
    ``set_trigger_out`` docstring's "at the start of every demodulation window" -- and is
    consistent with the archived bench routine ``QCTrace`` packages having taken usable QC
    traces under this exact gated-ramp, 30 ms-width configuration over whole records.
    (``QCTrace`` itself has not yet been run on hardware, so it is not independent evidence.)
    How the FPGA actually responds to a width exceeding the window period has not been checked
    on an oscilloscope.
    """

    def __init__(
        self,
        lo_freq: float,
        if_freqs: FloatAny,
        df: float,
        pixel_counts: int,
        amp: FloatAny,
        output_port: int,
        input_port: int,
        is_usb: Optional[BoolAny] = None,
        dither: bool = True,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
        external_trigger: TriggerAny = False,
        discard_start_ms: float = 25.0,
    ) -> None:
        self.lo_freq = lo_freq
        self.if_freqs = np.asarray(if_freqs, dtype=np.float64)
        self.df = df  # modified after tuning
        self.pixel_counts = pixel_counts
        # Per-tone amplitudes. A single scalar is broadcast to every tone (equal
        # drive), matching the is_usb convention. This guards against a presto
        # footgun: OutputGroup.set_amplitudes drives ONLY tone 0 and silently
        # zeroes the rest when given fewer amplitudes than tones, so feeding one
        # amp for a multi-IF measurement would leave every tone but the first
        # unprobed.
        amp_arr = np.atleast_1d(np.asarray(amp, dtype=np.float64))
        if amp_arr.size == 1:
            self.amp = np.full(self.if_freqs.shape, amp_arr.item())
        else:
            self.amp = amp_arr
        # Per-tone sideband selection: True -> USB (LO + IF), False -> LSB (LO - IF).
        # Defaults to all-USB to preserve previous behaviour. A single bool is
        # broadcast to every tone.
        if is_usb is None:
            self.is_usb = np.ones_like(self.if_freqs, dtype=bool)
        else:
            is_usb_arr = np.atleast_1d(np.asarray(is_usb, dtype=bool))
            if is_usb_arr.size == 1:
                self.is_usb = np.full(self.if_freqs.shape, bool(is_usb_arr.item()))
            else:
                self.is_usb = is_usb_arr
        # Auto-calculate single-sideband phases: I = 0, and Q lags I by 90° for
        # USB (-π/2), leads I by 90° for LSB (+π/2). Users never set phases by hand.
        self.phases_i = np.zeros_like(self.if_freqs, dtype=np.float64)
        self.phases_q = self.phases_i + np.where(self.is_usb, -np.pi / 2, np.pi / 2)
        self.output_port = output_port
        self.input_port = input_port
        self.dither = dither
        self.device = device
        self.filter = filter
        self.notes = notes
        # Per-digital-output-port trigger states, resolved to the list presto's
        # set_trigger_out expects. Element i configures port i+1, so which
        # instrument is gated depends on what is wired where: [1] drives port 1
        # only, [0, 1] port 2 only, [1, 1] both. Prefer daq.triggers.trigger_for
        # (external_trigger=trigger_for(bias, led)) to naming ports here -- it
        # reads the wiring off the instruments. See resolve_trigger_states.
        self.external_trigger = self.resolve_trigger_states(external_trigger)
        # The first tens of milliseconds of an acquisition are typically startup
        # junk. discard_start_ms leading milliseconds are dropped from the
        # in-memory time-axis arrays after run()/load() (the saved HDF5 keeps the
        # full acquisition). Set to 0 to keep everything.
        self.discard_start_ms = float(discard_start_ms)

        # Data arrays - set by run method
        self.freq_arr = None
        self.pixel_i = None
        self.pixel_q = None
        self.lsb = None
        self.usb = None
        self.freqs_usb = None
        self.freqs_lsb = None
        # Per-tone selected sideband (set by run method):
        #   signal[:, i]      -> IQ timestream of tone i at its chosen sideband
        #   signal_freqs[i]   -> physical frequency (Hz) of tone i
        self.signal = None
        self.signal_freqs = None

        self.fit_results = None
        """Parity-model fit of this stream's spectrum, from :meth:`fit_parity`.

        Set only by the constant-bias reconstruction. Like every other ``fit_results`` in the
        repo it lives on the object alone -- ``Base`` skips the name in both the HDF5 and the
        MongoDB paths, and this one holds a live ``iminuit`` object besides.
        """

        self.check_amp()
        self.check_sideband()
        self.check_discard()

    def check_amp(self) -> None:
        assert self.amp.shape == self.if_freqs.shape, (
            "amp must be a scalar or have the same length as if_freqs "
            f"({self.amp.shape} != {self.if_freqs.shape})"
        )
        assert self.amp.sum() < 1.0, "Amplitude sum must be less than 1.0"

    def check_sideband(self) -> None:
        assert self.is_usb.shape == self.if_freqs.shape, (
            "is_usb must have the same length as if_freqs "
            f"({self.is_usb.shape} != {self.if_freqs.shape})"
        )

    @staticmethod
    def resolve_trigger_states(external_trigger: TriggerAny) -> npt.NDArray[np.int64]:
        """Normalise an ``external_trigger`` argument to presto's per-port states list.

        Thin alias for :func:`daq.triggers.resolve_trigger_states`, which owns the port
        semantics so instrument drivers can share them without importing ``presto``. See it
        for the full description of the states and their timing constraints.

        Because the resolved value is what :attr:`external_trigger` stores, that attribute is
        an array rather than the bool it was before per-port routing existed. Test it with
        ``.any()`` or ``.size``; plain ``if ts.external_trigger:`` raises on an empty or
        multi-element array.

        Prefer :func:`daq.triggers.trigger_for` over writing port numbers here: it derives the
        states from the instruments themselves, so a rig wired differently from the lab
        default gates the right hardware without editing the measurement.

        :param external_trigger: ``True``/``False``, or a sequence of per-port states.
        :return: The resolved states as an integer array; empty when no port is triggered.
        :raises ValueError: If a state is outside ``{0, 1, 2}``, is non-integral, or more than
            :data:`~daq.triggers.MAX_TRIGGER_PORTS` ports are addressed.

        """
        return resolve_trigger_states(external_trigger)

    def check_discard(self) -> None:
        if self.discard_start_ms < 0:
            raise ValueError(f"discard_start_ms must be non-negative, got {self.discard_start_ms}")
        # Rough guard against discarding ~everything, using the requested df (the
        # tuned rate is nearly identical), so we fail before running the hardware.
        if int(round(self.discard_start_ms * 1e-3 * self.df)) >= self.pixel_counts - 1:
            raise ValueError(
                f"discard_start_ms={self.discard_start_ms} ms discards ~all of the "
                f"{self.pixel_counts} samples at df={self.df} Hz; leaves fewer than 2 samples"
            )

    def _apply_discard_start(self) -> None:
        """Drop the leading ``discard_start_ms`` of startup junk in place.

        Trims the in-memory time-axis arrays (``signal``, ``usb``, ``lsb``,
        ``pixel_i``, ``pixel_q``) by ``round(discard_start_ms * 1e-3 * df)``
        samples using the tuned hardware sample rate :attr:`df`. Frequency-axis
        arrays (``freq_arr``, ``signal_freqs``) are left untouched. Only the
        in-memory arrays are trimmed -- the HDF5 file written by :meth:`run`
        keeps the full, untrimmed acquisition.
        """
        if self.discard_start_ms <= 0 or self.signal is None:
            return
        n_discard = int(round(self.discard_start_ms * 1e-3 * self.df))
        if n_discard <= 0:
            return
        n_samples = self.signal.shape[0]
        if n_discard >= n_samples - 1:
            raise ValueError(
                f"discard_start_ms={self.discard_start_ms} ms drops {n_discard} of "
                f"{n_samples} samples at df={self.df} Hz; leaves fewer than 2 samples"
            )
        for attr in ("signal", "usb", "lsb", "pixel_i", "pixel_q"):
            arr = getattr(self, attr, None)
            if arr is not None:
                setattr(self, attr, arr[n_discard:])

    def run(
        self,
        presto_address: Optional[str] = None,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
        save_filename: Optional[str] = None,
        on_acquire: Optional[Callable[[], None]] = None,
    ) -> str:
        """Run the acquisition, save it, and return the saved file's path.

        :param presto_address: Presto IP address. Defaults to ``DAQ_PRESTO_ADDRESS``.
        :param presto_port: Presto port. Defaults to ``DAQ_PRESTO_PORT``.
        :param ext_ref_clk: Lock the Presto to its external reference clock input.
        :param save_filename: Explicit output path; auto-generated when ``None``.
        :param on_acquire: Optional callable invoked **once, immediately before acquisition
            starts** -- after the seconds of connect/configure/tune work and after
            ``apply_settings()`` (so any configured trigger output is already asserted), just
            before ``get_pixels()``. This is the place to software-start side hardware whose
            timing should land close to sample zero, e.g. a DC2200 pulse train via
            ``on_acquire=lambda: setattr(led, "output", True)``: started *before* ``run()``
            the offset from sample zero is seconds and varies per run; started here it is set
            by a few SCPI round trips against the ``get_pixels`` start -- ms-scale, but its
            sign and size are **not yet bench-measured**, so read the actual offset off the
            data (the first recorded pulse position, modulo the pulse period). Keep the
            callable fast and side-effect-only; its return value is ignored. If it raises,
            the acquisition is abandoned and the exception propagates -- the Presto outputs
            are muted on the way out and the connection closes, but disarming any instrument
            the caller armed stays with the caller (e.g. its ``with`` block).
        :returns: Path of the saved HDF5 file.

        """
        if presto_address is None:
            presto_address = get_presto_address()
        if presto_port is None:
            presto_port = get_presto_port()
        with lockin.Lockin(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **self.DC_PARAMS,
        ) as lck:
            lck.hardware.set_adc_attenuation(self.input_port, self.ADC_ATTENUATION)
            lck.hardware.set_dac_current(self.output_port, self.DAC_CURRENT)
            lck.hardware.set_inv_sinc(self.output_port, 0)
            lck.hardware.configure_mixer(
                self.lo_freq, out_ports=self.output_port, in_ports=self.input_port
            )
            lck.set_dither(self.dither, self.output_port)

            _, self.df = lck.tune(0.0, self.df)
            lck.set_df(self.df)
            if np.any(~np.isclose(self.if_freqs, 0.0)):
                lck.set_phase_reset(False)

            # Configure output group
            og = lck.add_output_group(self.output_port, len(self.if_freqs))
            og.set_frequencies(self.if_freqs)
            og.set_amplitudes(self.amp)
            og.set_phases(self.phases_i, self.phases_q)

            # Configure input group
            ig = lck.add_input_group(self.input_port, len(self.if_freqs))
            ig.set_frequencies(self.if_freqs)

            # Re-resolve rather than trusting the stored array, so assigning
            # `ts.external_trigger = True` (or a states list) after construction
            # still does what it says.
            trigger_states = self.resolve_trigger_states(self.external_trigger)
            if trigger_states.any():
                # The trigger goes high as soon as "lck.apply_settings" is called,
                # and stays high for the acquisition -- see TRIGGER_WIDTH_S.
                lck.set_trigger_out(trigger_states.tolist(), width=self.TRIGGER_WIDTH_S)

            lck.apply_settings()

            try:
                if on_acquire is not None:
                    # Last stop before data: everything slow (connect, configure, tune) is
                    # done and any trigger output is asserted, so side hardware started here
                    # lands ms-scale from sample zero instead of a seconds-scale offset.
                    on_acquire()

                # Acquire data
                pixel_dict = lck.get_pixels(self.pixel_counts)
            finally:
                # Mute on the exception path too. Without this, a raising hook (arbitrary
                # user code, running at exactly the moment the outputs are live) would close
                # the connection with the tones still driving and any trigger re-asserting
                # every lock-in window -- a TTL-armed LED would stay lit indefinitely.
                # Best-effort: a mute failure must not mask the original exception.
                try:
                    if trigger_states.any():
                        # set_trigger_out rebuilds its control word from zero, so a single
                        # 0 clears every port, not just port 1.
                        lck.set_trigger_out([0])
                    og.set_amplitudes(0.0)
                    lck.apply_settings()
                except Exception as mute_err:
                    print(f"WARN: failed to mute Presto outputs during cleanup: {mute_err}")

            self.freq_arr, self.pixel_i, self.pixel_q = pixel_dict[self.input_port]
            self.lsb, self.usb = untwist_downconversion(self.pixel_i, self.pixel_q)

            # Calculate frequency arrays
            self.freqs_usb = self.lo_freq + self.if_freqs
            self.freqs_lsb = self.lo_freq - self.if_freqs

            # Select the driven sideband for each tone so users get the right
            # data directly, without remembering USB/LSB conventions.
            self.signal = np.where(self.is_usb[np.newaxis, :], self.usb, self.lsb)
            self.signal_freqs = np.where(self.is_usb, self.freqs_usb, self.freqs_lsb)

        # Save the full acquisition first, then drop the leading junk from the
        # in-memory arrays so the returned object matches the analysed window.
        save_path = self.save(save_filename=save_filename)
        self._apply_discard_start()
        return save_path

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "TimeStream":
        with h5py.File(load_filename, "r") as h5f:
            lo_freq = float(h5f.attrs["lo_freq"])  # type: ignore
            df = float(h5f.attrs["df"])  # type: ignore
            pixel_counts = int(h5f.attrs["pixel_counts"])  # type: ignore
            output_port = int(h5f.attrs["output_port"])  # type: ignore
            input_port = int(h5f.attrs["input_port"])  # type: ignore
            dither = bool(h5f.attrs["dither"])  # type: ignore
            # discard_start_ms is absent in files saved before this field existed;
            # default to the standard trim so old and new files behave the same.
            discard_start_ms = (
                float(h5f.attrs["discard_start_ms"]) if "discard_start_ms" in h5f.attrs else 25.0
            )
            # external_trigger has three vintages: absent (oldest files), a scalar
            # bool attribute (before per-port routing), and a per-port states
            # dataset (current). resolve_trigger_states maps the bool onto [1],
            # which is the port the bool always meant.
            if "external_trigger" in h5f:
                external_trigger = h5f["external_trigger"][()]  # type: ignore
            elif "external_trigger" in h5f.attrs:
                external_trigger = bool(h5f.attrs["external_trigger"])
            else:
                external_trigger = False

            if_freqs: npt.NDArray[np.float64] = h5f["if_freqs"][()]  # type: ignore
            amp: npt.NDArray[np.float64] = h5f["amp"][()]  # type: ignore
            # is_usb may be absent in files saved before sideband selection existed
            is_usb = h5f["is_usb"][()] if "is_usb" in h5f else None  # type: ignore

            # Load data arrays if they exist
            freq_arr = h5f["freq_arr"][()] if "freq_arr" in h5f else None  # type: ignore
            pixel_i = h5f["pixel_i"][()] if "pixel_i" in h5f else None  # type: ignore
            pixel_q = h5f["pixel_q"][()] if "pixel_q" in h5f else None  # type: ignore
            lsb = h5f["lsb"][()] if "lsb" in h5f else None  # type: ignore
            usb = h5f["usb"][()] if "usb" in h5f else None  # type: ignore
            freqs_usb = h5f["freqs_usb"][()] if "freqs_usb" in h5f else None  # type: ignore
            freqs_lsb = h5f["freqs_lsb"][()] if "freqs_lsb" in h5f else None  # type: ignore
            signal = h5f["signal"][()] if "signal" in h5f else None  # type: ignore
            signal_freqs = h5f["signal_freqs"][()] if "signal_freqs" in h5f else None  # type: ignore

            # Everything else in the attributes is state this class does not own: the
            # per-instrument settings attach() flattened onto the measurement (bias_function,
            # bias_freq_hz, led_mode, ...), plus device/filter/notes. Restoring them is what
            # lets a *loaded* stream still say how it was biased, which is what analyze()'s
            # reconstruction dispatch reads. Copied wholesale rather than by a fixed list, so a
            # new instrument driver's settings come back without touching this method.
            extra = {
                str(key): value
                for key, value in h5f.attrs.items()
                if str(key) not in cls._CONSTRUCTOR_ATTRS
            }

        # Legacy files (saved before scalar-amp broadcasting) stored a single scalar
        # amp for a multi-tone measurement. presto's set_amplitudes drove only the
        # first tone and left the rest unpowered, so those other tones contain only
        # the noise floor. Reconstruct amp as [amp, 0, ..., 0] to reflect what was
        # actually driven (this also keeps the full-scale sum check happy, since the
        # lone scalar already passed it when the file was written) and warn the user.
        amp_arr = np.atleast_1d(np.asarray(amp, dtype=np.float64))
        n_tones = np.atleast_1d(if_freqs).shape[0]
        if amp_arr.size == 1 and n_tones > 1:
            warnings.warn(
                f"{load_filename} was saved with a single scalar amp for {n_tones} "
                "tones (legacy pre-broadcast format). presto drove only the first tone "
                "and left the others unpowered, so only tone 0 carries meaningful "
                "signal; the remaining tones are just the noise floor. Reconstructing "
                "amp as [amp, 0, ...] to reflect what was actually driven.",
                stacklevel=2,
            )
            amp = np.concatenate([amp_arr, np.zeros(n_tones - 1, dtype=np.float64)])

        self = cls(
            lo_freq=lo_freq,
            if_freqs=if_freqs,
            df=df,
            pixel_counts=pixel_counts,
            amp=amp,
            output_port=output_port,
            input_port=input_port,
            is_usb=is_usb,
            dither=dither,
            external_trigger=external_trigger,
            discard_start_ms=discard_start_ms,
        )

        # Restore data arrays
        self.freq_arr = freq_arr
        self.pixel_i = pixel_i
        self.pixel_q = pixel_q
        self.lsb = lsb
        self.usb = usb
        self.freqs_usb = freqs_usb
        self.freqs_lsb = freqs_lsb
        self.signal = signal
        self.signal_freqs = signal_freqs

        # Reconstruct the per-tone selected sideband for files saved before
        # `signal`/`signal_freqs` existed (defaults to all-USB on those files).
        if self.signal is None and self.usb is not None and self.lsb is not None:
            self.signal = np.where(self.is_usb[np.newaxis, :], self.usb, self.lsb)
        if self.signal_freqs is None and freqs_usb is not None and freqs_lsb is not None:
            self.signal_freqs = np.where(self.is_usb, freqs_usb, freqs_lsb)

        # Attached instrument state and the database metadata, restored after the constructor
        # so nothing it set is clobbered by a stale attribute of the same name.
        for key, value in extra.items():
            setattr(self, key, _as_text(value) if isinstance(value, bytes) else value)

        # Files written by run() hold the full acquisition, so re-apply the trim
        # to match the in-memory state produced by a live run.
        self._apply_discard_start()

        return self

    # ------------------------------------------------------------------ how it was biased

    def _attached_bias(self) -> Tuple[Optional[str], Dict[str, Any]]:
        """Find the attached function generator's recorded settings.

        :meth:`~daq._base.Base.attach` flattens an instrument's ``settings()`` onto the
        measurement as ``<prefix>_<key>``, so a generator attached as ``bias`` leaves
        ``bias_function``, ``bias_freq_hz`` and the rest. The carrier-waveform key is what
        identifies a function generator among the attachments: the DC2200 reports ``mode``,
        not ``function``, so an attached LED is never mistaken for the gate bias.

        :returns: ``(prefix, settings)`` with the prefix stripped from the keys, or
            ``(None, {})`` when nothing that looks like a function generator is attached.

        """
        prefixes = sorted(
            name[: -len(BIAS_FUNCTION_SUFFIX)]
            for name, value in self.__dict__.items()
            if name.endswith(BIAS_FUNCTION_SUFFIX)
            and name != BIAS_FUNCTION_SUFFIX
            and isinstance(value, (str, bytes))
        )
        if not prefixes:
            return None, {}

        if len(prefixes) == 1:
            prefix = prefixes[0]
        else:
            prefix = DEFAULT_BIAS_PREFIX if DEFAULT_BIAS_PREFIX in prefixes else prefixes[0]
            warnings.warn(
                f"More than one attached instrument reports a carrier waveform ({prefixes}); "
                f"reading the bias off {prefix!r}. Attach the gate-bias generator as "
                f"{DEFAULT_BIAS_PREFIX!r} to make the choice explicit, or pass "
                "analyze(mode=...) to skip the detection.",
                stacklevel=3,
            )

        head = f"{prefix}_"
        settings = {
            name[len(head) :]: value
            for name, value in self.__dict__.items()
            if name.startswith(head)
        }
        return prefix, settings

    @property
    def bias_settings(self) -> Dict[str, Any]:
        """The attached gate-bias generator's recorded settings, prefix stripped.

        These are the values :meth:`~daq._base.Base.attach` read back from the instrument at
        the time of the call -- a snapshot of the hardware, not a live link.

        :returns: The settings mapping; empty when no generator is attached.

        """
        return self._attached_bias()[1]

    @property
    def bias_mode(self) -> str:
        """How the gate was biased during this acquisition.

        Read off the generator settings :meth:`~daq._base.Base.attach` recorded, which
        :meth:`load` restores, so it works on a reloaded file as well as a live run. It is the
        *configured* waveform: a gated ramp that nothing triggered still reports
        ``"sawtooth"`` (and :meth:`analyze` warns about it), because that is the measurement
        that was attempted.

        :returns: ``"sawtooth"`` for a ramp, ``"constant"`` for DC, and ``"unknown"`` when no
            generator was attached or its carrier is neither -- in which case
            :meth:`analyze` falls back to the plain time-stream plot.

        """
        settings = self.bias_settings
        if not settings:
            return "unknown"
        function = _as_text(settings.get("function", "")).strip().upper()
        if function.startswith("RAMP"):
            return "sawtooth"
        if function.startswith("DC"):
            return "constant"
        return "unknown"

    @property
    def bias_period_s(self) -> Optional[float]:
        """Period of the attached bias ramp in seconds.

        :returns: ``1 / freq_hz`` of the attached generator, or ``None`` unless the bias was a
            ramp with a usable frequency.

        """
        if self.bias_mode != "sawtooth":
            return None
        freq_hz = self.bias_settings.get("freq_hz")
        if freq_hz is None or float(freq_hz) <= 0:
            return None
        return 1.0 / float(freq_hz)

    def _warn_if_ramp_ungated(self) -> None:
        """Warn when the attached ramp was gated on a port this acquisition never asserted.

        The generator then sits at its burst start level for the whole record and the data is
        a static bias wearing a sawtooth's metadata -- the silent failure
        :class:`~daq.measurements.qc_trace.QCTrace` refuses outright. Folding it succeeds and
        returns a flat trace, so it is worth saying so before the plot rather than after.

        """
        settings = self.bias_settings
        if not settings.get("burst", False):
            # A free-running ramp needs no trigger; nothing to check.
            return
        port = settings.get("trigger_port")
        if port is None:
            return
        port = int(port)
        states = resolve_trigger_states(self.external_trigger)
        if port <= states.size and states[port - 1]:
            return
        warnings.warn(
            f"The attached bias generator was in gated-burst mode on trigger port {port}, but "
            f"this acquisition asserted {describe_trigger_states(states)}. The ramp would have "
            "waited on a gate that never came and held its burst start level, so the record is "
            "a static bias and the folded trace will be flat. Check the wiring "
            "(bias.trigger_port, or DAQ_FGEN_TRIGGER_PORT) against the measurement's "
            "external_trigger.",
            stacklevel=3,
        )

    # ------------------------------------------------------------------ reconstructions

    def fold(
        self,
        *,
        period_s: Optional[float] = None,
        n_periods: Optional[int] = None,
        tone: int = 0,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Block-average this stream into a single bias-ramp period.

        The sawtooth-bias reconstruction: the gate ramp repeats while the stream records
        continuously, so averaging in blocks of one period beats uncorrelated noise down by
        ``sqrt(n_periods)`` and leaves the device's response to one sweep of the gate.

        The period defaults to the attached generator's own, ``1 / bias_freq_hz``, and the
        fold runs at the **tuned** :attr:`df` rather than the requested sample rate -- a
        window off by a sample smears the average across blocks instead of dropping a
        leftover. Note that folding cuts the record into ``round(period_s * df)``-sample
        blocks, so a sample rate that is not a whole multiple of the ramp rate makes every
        block start a fraction of a sample late and the error accumulate; see
        :class:`~daq.measurements.qc_trace.QCTrace`, which warns about this up front.

        :param period_s: Fold on this period instead of the attached ramp's.
        :param n_periods: Fold on the record divided into this many periods instead. Mutually
            exclusive with *period_s*.
        :param tone: Which tone to fold, for a multi-tone stream.
        :raises RuntimeError: If there is no data, or no period is known and none was given.
        :raises ValueError: If both *period_s* and *n_periods* are given, or the record is
            shorter than one period.
        :returns: ``(time_ms, avg_iq)``, as
            :func:`~daq.analysis.folding.fold_timestream` -- ``avg_iq`` has shape
            ``(2, n_samples)``, row 0 I and row 1 Q.

        """
        from ..analysis.folding import fold_timestream

        if self.signal is None:
            raise RuntimeError("No data available. Run or load the measurement first.")
        if period_s is None and n_periods is None:
            period_s = self.bias_period_s
            if period_s is None:
                raise RuntimeError(
                    "No bias-ramp period is known for this stream, so there is nothing to fold "
                    "on. Attach the generator before running "
                    "(ts.attach(bias=fgen)) so the ramp frequency is recorded, or pass the "
                    "period explicitly: ts.fold(period_s=1 / ramp_freq_hz)."
                )
        return fold_timestream(self, self.df, period_s=period_s, n_periods=n_periods, tone=tone)

    def _projection(
        self, tone: int = 0, quantity: str = DEFAULT_QUANTITY
    ) -> npt.NDArray[np.floating]:
        """Return a real projection of the complex readout, as recorded.

        :param tone: Which tone to project.
        :param quantity: One of :data:`~daq.analysis.parity.PROJECTIONS`.
        :raises RuntimeError: If there is no data.
        :raises ValueError: If *quantity* is not a known projection.
        :returns: The projected series, operating point included.

        """
        from ..analysis.parity import project_readout

        if self.signal is None:
            raise RuntimeError("No data available. Run or load the measurement first.")
        return project_readout(np.asarray(self.signal)[:, tone], quantity)

    def _parity_series(
        self, tone: int = 0, quantity: str = DEFAULT_QUANTITY
    ) -> npt.NDArray[np.floating]:
        """Return the real-valued series whose spectrum is the parity spectrum.

        :param tone: Which tone to project.
        :param quantity: One of :data:`~daq.analysis.parity.PROJECTIONS`.
        :raises RuntimeError: If there is no data.
        :raises ValueError: If *quantity* is not a known projection.
        :returns: The mean-subtracted series -- the fluctuation, not the operating point.

        """
        series = self._projection(tone=tone, quantity=quantity)
        return series - series.mean()

    def _tones(self, tone: Optional[int]) -> List[int]:
        """Resolve a *tone* argument to the list of tones to work on.

        :param tone: A tone index, or ``None`` for every tone.
        :raises RuntimeError: If there is no data.
        :raises IndexError: If *tone* is out of range.
        :returns: The tone indices, in order.

        """
        if self.signal is None:
            raise RuntimeError("No data available. Run or load the measurement first.")
        n_tones = int(np.asarray(self.signal).shape[1])
        if tone is None:
            return list(range(n_tones))
        if not -n_tones <= tone < n_tones:
            raise IndexError(f"tone {tone} is out of range for a {n_tones}-tone stream")
        return [tone % n_tones]

    def parity_psd(
        self,
        *,
        tone: Optional[int] = None,
        quantity: str = DEFAULT_QUANTITY,
        welch: bool = False,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Noise spectrum of this stream, for the constant-bias reconstruction.

        The spectrum of the **mean-subtracted** projection of the readout, at the **tuned**
        :attr:`df` -- the parity signal is the fluctuation about the operating point, not the
        point itself, and the frequency axis follows the rate the hardware settled on rather
        than the one that was asked for. ``"abs"`` matches the metric
        :class:`~daq.measurements.bias_hunt.BiasHunt` ranks its tries by.

        Every tone is spectrated by default, one row per tone -- a multi-tone parity
        acquisition is normally a signal tone beside a reference, and taking only the first
        would quietly answer a question nobody asked.

        One stream's periodogram is noisy; ``BiasHunt.average_psd`` averages over the tries to
        beat that down before fitting.

        :param tone: Which tone to take the spectrum of. ``None`` (default) takes every tone.
        :param quantity: Projection of the complex readout -- ``"abs"``, ``"real"`` or
            ``"imag"``.
        :param welch: Use Welch's method instead of the bare periodogram.
        :param nperseg: Welch segment length. Ignored unless *welch*.
        :param noverlap: Welch segment overlap. Ignored unless *welch*.
        :returns: ``(f, psd)`` -- frequencies in Hz and the PSD in ``FS^2/Hz``. ``psd`` is 1-D
            for a single tone (whether named or because the stream has one) and 2-D, one row
            per tone, otherwise -- following
            :func:`~daq.analysis.noise.fit_parity_psd`'s dict-versus-list convention.

        """
        from ..analysis.noise import compute_psd

        tones = self._tones(tone)
        series = np.vstack([self._parity_series(tone=ii, quantity=quantity) for ii in tones])
        f, psd = compute_psd(
            series if len(tones) > 1 else series[0],
            self.df,
            welch=welch,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        return f, psd

    def fit_parity(
        self,
        *,
        tone: Optional[int] = None,
        quantity: str = DEFAULT_QUANTITY,
        welch: bool = False,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        **fit_kwargs: Any,
    ) -> Any:
        """Fit this stream's noise spectrum to the random-telegraph parity model.

        The sampling bandwidth is held at the tuned :attr:`df`, as
        :func:`~daq.analysis.noise.fit_parity_psd` expects. Check ``resid_dex_rms`` before
        believing the result -- ``~0.1`` is a good fit, ``~1`` means the model is a decade off
        across the band. Worth checking too against ``gamma_p_flips`` from
        :meth:`reconstruct_parity`, which measures the same rate in the time domain and knows
        nothing about this model.

        The result is stored on :attr:`fit_results` and reused by :meth:`analyze`.

        :param tone: Which tone to fit. ``None`` (default) fits every tone.
        :param quantity: Projection of the complex readout to fit.
        :param welch: Use Welch's method for the spectrum.
        :param nperseg: Welch segment length. Ignored unless *welch*.
        :param noverlap: Welch segment overlap. Ignored unless *welch*.
        :param fit_kwargs: Passed to :func:`~daq.analysis.noise.fit_parity_psd` --
            ``fit_onef=True`` for a ``1/f``-like low-frequency rise, ``n_bins``, and so on.
        :returns: The ``fit_results`` dict for a single tone, or a list of them, one per tone.

        """
        from ..analysis.noise import fit_parity_psd

        f, psd = self.parity_psd(
            tone=tone, quantity=quantity, welch=welch, nperseg=nperseg, noverlap=noverlap
        )
        self.fit_results = fit_parity_psd(f, psd, f_bw=self.df, **fit_kwargs)
        return self.fit_results

    def reconstruct_parity(
        self,
        *,
        tone: Optional[int] = None,
        ramped: Optional[bool] = None,
        bursts: bool = True,
        burst_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Recover the parity-flip times of this acquisition, via ``qpd``.

        The time-domain counterpart of :meth:`fit_parity`: where that fits a switching rate out
        of the spectrum, this recovers the individual tunnelling events. The method is
        :mod:`qpd.reconstruction`'s -- a blind two-blob emission model and a two-state HMM
        decode -- reached through :mod:`daq.analysis.parity`; this measurement supplies the
        trace, the tuned sample rate, and which reconstruction the bias calls for, and passes
        every algorithm parameter through on ``qpd``'s own defaults.

        Which reconstruction follows :attr:`bias_mode`, since it is the same distinction:
        a **constant** gate holds ``n_g`` fixed and gives two stationary blobs, a **sawtooth**
        sweeps it so the branches move, cross blind and reset with the ramp. Pass *ramped* to
        override.

        **Check ``degenerate`` and ``contrast`` on the result before using it** -- a model that
        latched onto noise fails quietly, and its fidelity estimate stays high while it does.
        The rate it reports (``rate_hz``) is worth comparing against the fitted ``gamma_p``
        from :meth:`fit_parity`: the two share no machinery, so agreement is evidence and
        disagreement means one of them is wrong.

        :param tone: Which tone to reconstruct. ``None`` (default) reconstructs every tone.
        :param ramped: Force the swept-gate (``True``) or fixed-gate (``False``)
            reconstruction. ``None`` takes it from :attr:`bias_mode`.
        :param bursts: Also cluster the flip train into rapid-switching bursts
            (:func:`~daq.analysis.parity.detect_bursts`), using the reconstruction's own rate
            as the Poisson background.
        :param burst_kwargs: Passed to that call (``max_gap``, ``min_flips``,
            ``max_p_value``); ``qpd``'s defaults otherwise. Note its ``max_p_value`` default is
            ``None``, i.e. every cluster is returned with its p-value rather than filtered --
            see :func:`~daq.analysis.parity.detect_bursts`.
        :param kwargs: Passed to the ``qpd`` reconstruction routine.
        :raises ImportError: If ``qpd`` is not installed.
        :returns: ``qpd``'s result for a single tone, or a list of them, one per tone. Each
            carries a ``bursts`` list and a ``tone`` index attached by this method.

        """
        from ..analysis.parity import detect_bursts, reconstruct_parity

        if ramped is None:
            ramped = self.bias_mode == "sawtooth"

        results = []
        for index in self._tones(tone):
            signal = np.asarray(self.signal)[:, index]
            result = reconstruct_parity(signal, self.df, ramped=ramped, **kwargs)
            found = []
            # A burst hunt over a degenerate model would be a hunt through noise crossings,
            # and it would find plenty.
            if bursts and not getattr(result, "degenerate", False):
                found = detect_bursts(
                    result.flip_times,
                    getattr(result, "rate_hz", 0.0),
                    signal.shape[0] / self.df,
                    **(burst_kwargs or {}),
                )
            result.bursts = found
            result.tone = index
            results.append(result)
        return results[0] if len(results) == 1 else results

    # ------------------------------------------------------------------ analysis

    def _resolve_analyze_mode(self, mode: Optional[str]) -> str:
        """Decide which reconstruction :meth:`analyze` runs.

        :param mode: The caller's *mode* argument.
        :raises ValueError: If *mode* is not a recognised reconstruction.
        :returns: One of :data:`ANALYZE_MODES`.

        """
        if mode is None or mode == "auto":
            detected = self.bias_mode
            return "raw" if detected == "unknown" else detected
        if mode not in ANALYZE_MODES:
            raise ValueError(f"mode must be 'auto' or one of {list(ANALYZE_MODES)}, got {mode!r}")
        return mode

    def _title(self, prefix: str, tone: int) -> str:
        """Compose a default figure title naming the device and the tone's frequency.

        :param prefix: Leading description of the plot.
        :param tone: Which tone the plot is of.
        :returns: The title.

        """
        parts = [prefix]
        if self.device is not None:
            parts.append(str(self.device))
        if self.signal_freqs is not None:
            parts.append(f"{np.atleast_1d(self.signal_freqs)[tone] / 1e9:.6f} GHz")
        return " -- ".join(parts)

    def analyze(
        self,
        num_samples: Optional[int] = None,
        title: Optional[str] = None,
        show_iq: bool = True,
        *,
        mode: Optional[str] = None,
        tone: Optional[int] = None,
        period_s: Optional[float] = None,
        raw: bool = True,
        fit: bool = True,
        reconstruct: bool = True,
        quantity: str = DEFAULT_QUANTITY,
        **fit_kwargs: Any,
    ):
        """Reconstruct and plot the acquisition according to how the gate was biased.

        The bias is not a property of the time stream the Presto took -- it is a property of
        what the function generator was doing while it took it -- so this reads the generator
        settings :meth:`~daq._base.Base.attach` recorded (and :meth:`load` restores) and runs
        the matching reconstruction:

        ==================  =======================================================
        :attr:`bias_mode`   what ``analyze()`` does
        ==================  =======================================================
        ``sawtooth``        folds the record into one ramp period (:meth:`fold`) and
                            plots the block-averaged I/Q trace -- a QC trace
        ``constant``        annotates the I/Q streams with the two-level parity
                            reconstruction (:meth:`reconstruct_parity`) -- levels,
                            switching events and any rapid-switching burst -- above
                            the noise spectra (:meth:`parity_psd`) fitted to the
                            random-telegraph parity model
        ``unknown``         the plain per-tone time-stream plot, as before
        ==================  =======================================================

        **Every reconstruction is per tone and shows all of them**, one row of the usual
        two-column grid each, with one spectrum per tone on the shared panel below. A
        multi-tone parity acquisition is normally a signal tone beside a reference; showing
        only the first would answer a question nobody asked.

        So ``ts.attach(bias=fgen)`` before ``run()`` is what makes a bare ``ts.analyze()``
        do the right thing afterwards; without it nothing about the samples says whether the
        gate was ramping, and the fallback is the raw plot. Pass *mode* to override the
        detection either way.

        This is the same reconstruction the composed measurements do --
        :class:`~daq.measurements.qc_trace.QCTrace` folds, and
        :class:`~daq.measurements.bias_hunt.BiasHunt` takes and fits the spectrum -- reached
        from the stream itself, so a single hand-rolled acquisition or a reloaded file gets it
        without wrapping it in a measurement class.

        :param num_samples: Limit the time-axis panels to this many samples. Applies to the
            ``raw`` and ``constant`` plots; the folded trace is one period long by
            construction. The reconstruction itself still runs on the whole record, so the
            rates and bursts reported describe the acquisition rather than the window.
        :param title: Figure title. Defaults to naming the reconstruction, the device and the
            tone's frequency.
        :param show_iq: Show I and Q rather than power and phase, in the ``raw`` and
            ``constant`` grids. The reconstructed levels are drawn only on I/Q.
        :param mode: Force a reconstruction: ``"sawtooth"``, ``"constant"`` or ``"raw"``.
            ``None`` (the default) or ``"auto"`` reads :attr:`bias_mode`.
        :param tone: Restrict the figure to one tone. ``None`` (the default) shows every tone,
            one row each.
        :param period_s: ``sawtooth`` mode only -- fold on this period instead of the attached
            ramp's. The way to fold a stream whose generator was never attached.
        :param raw: ``sawtooth`` mode only -- overlay one un-averaged period, showing what the
            averaging bought.
        :param fit: ``constant`` mode only -- overlay the parity-model fit. A fit that raises
            is reported on the panel rather than costing the spectrum.
        :param reconstruct: ``constant`` mode only -- mark the two-level reconstruction on the
            streams. ``False`` leaves the grid bare.
        :param quantity: ``constant`` mode only -- which projection of the complex readout to
            threshold and to take the spectrum of (``"abs"``, ``"real"``, ``"imag"``).
        :param fit_kwargs: ``constant`` mode only -- passed to
            :func:`~daq.analysis.noise.fit_parity_psd` (e.g. ``fit_onef=True``).
        :raises RuntimeError: If there is no data to analyse.
        :raises ValueError: If *mode* or *quantity* is not recognised.
        :raises IndexError: If *tone* is out of range.
        :returns: The created figure.

        """
        if self.signal is None:
            raise RuntimeError("No data available. Run the measurement first.")

        resolved = self._resolve_analyze_mode(mode)
        if fit_kwargs and resolved != "constant":
            # **fit_kwargs would otherwise swallow a misspelled argument without a word.
            warnings.warn(
                f"analyze() ignored {sorted(fit_kwargs)}: fit arguments belong to the "
                f"constant-bias spectrum, and this stream reconstructs as {resolved!r}.",
                stacklevel=2,
            )
        if mode is None or mode == "auto":
            print(f"TimeStream.analyze: bias_mode={self.bias_mode!r} -> {resolved} reconstruction")

        if resolved == "sawtooth":
            return self._analyze_sawtooth(title=title, tone=tone, period_s=period_s, raw=raw)
        if resolved == "constant":
            return self._analyze_constant(
                num_samples=num_samples,
                title=title,
                tone=tone,
                show_iq=show_iq,
                fit=fit,
                reconstruct=reconstruct,
                quantity=quantity,
                **fit_kwargs,
            )
        return self._analyze_raw(num_samples=num_samples, title=title, show_iq=show_iq)

    def _analyze_sawtooth(
        self,
        *,
        title: Optional[str],
        tone: Optional[int],
        period_s: Optional[float],
        raw: bool,
    ):
        """Fold the record into one bias-ramp period and plot it, one row per tone.

        :param title: Figure title, or ``None`` for the default.
        :param tone: Which tone to fold, or ``None`` for all of them.
        :param period_s: Fold period, or ``None`` for the attached ramp's.
        :param raw: Overlay one un-averaged period.
        :returns: The created figure.

        """
        import matplotlib.pyplot as plt

        from ..analysis.plotting import plot_qc_trace

        self._warn_if_ramp_ungated()
        tones = self._tones(tone)

        fig, axes = plt.subplots(
            len(tones),
            2,
            figsize=(12, 3 * len(tones)),
            sharex=True,
            tight_layout=True,
            squeeze=False,
        )
        n_folded = 0
        for row, index in enumerate(tones):
            time_ms, avg_iq = self.fold(period_s=period_s, tone=index)
            n_folded = int(np.asarray(self.signal).shape[0] // avg_iq.shape[1])
            plot_qc_trace(
                time_ms,
                avg_iq,
                raw=self if raw else None,
                ax=tuple(axes[row]),
                tone=index,
                title="",
            )
            for axis, name in zip(axes[row], ("I [FS]", "Q [FS]")):
                axis.set_ylabel(f"{name}\n{self._tone_label(index)}")
                # plot_qc_trace labels its own x-axis; only the bottom row should keep one.
                axis.set_xlabel("")

        for axis in axes[-1]:
            axis.set_xlabel("Time [ms]")
        if title is None:
            title = self._title(f"QC trace (block-averaged, {n_folded} periods)", tones[0])
        fig.suptitle(title)

        plt.show()
        return fig

    def _time_axis(self, n_samples: int) -> Tuple[npt.NDArray[np.floating], float, str]:
        """Return a time axis, its scale factor from seconds, and its unit.

        Scaled to the length of what is plotted: a parity record runs for seconds and a QC
        period for milliseconds, so a fixed unit labels one of them in millions.

        :param n_samples: Number of samples being plotted.
        :returns: ``(time, scale, unit)``, where ``seconds * scale`` is in the returned unit.

        """
        duration_s = n_samples / self.df
        if duration_s >= 1.0:
            scale, unit = 1.0, "s"
        elif duration_s >= 1e-3:
            scale, unit = 1e3, "ms"
        else:
            scale, unit = 1e6, "μs"
        return np.arange(n_samples) / self.df * scale, scale, unit

    def _tone_label(self, tone: int) -> str:
        """Return the sideband-annotated frequency label of a tone.

        :param tone: Which tone.
        :returns: E.g. ``"6.437 GHz (USB)"``.

        """
        sideband = "USB" if self.is_usb[tone] else "LSB"
        freqs = np.atleast_1d(self.signal_freqs)
        return f"{freqs[tone] / 1e9:.3f} GHz ({sideband})"

    def _draw_stream_grid(
        self,
        axes,
        *,
        tones: Optional[List[int]] = None,
        num_samples: Optional[int] = None,
        show_iq: bool = True,
    ) -> Tuple[npt.NDArray[np.floating], float, str]:
        """Draw the per-tone two-column stream grid: I and Q, or power and phase.

        The historical :meth:`analyze` plot, factored out so the bias reconstructions can
        annotate it rather than replace it.

        :param axes: Axes array of shape ``(len(tones), 2)`` to draw into.
        :param tones: Which tones to draw, one row each. ``None`` draws every tone.
        :param num_samples: Plot only this many leading samples.
        :param show_iq: Draw I and Q rather than power and phase.
        :returns: ``(time, scale, unit)`` -- the axis the grid was drawn against.

        """
        data = np.asarray(self.signal)
        if num_samples is not None:
            data = data[:num_samples]
        tones = list(range(data.shape[1])) if tones is None else list(tones)
        time_axis, scale, unit = self._time_axis(data.shape[0])

        for row, tone in enumerate(tones):
            label = self._tone_label(tone)
            if show_iq:
                columns = (
                    (np.real(data[:, tone]), "I [FS]"),
                    (np.imag(data[:, tone]), "Q [FS]"),
                )
            else:
                columns = (
                    (20.0 * np.log10(np.abs(data[:, tone])), "Power [dBFS]"),
                    (np.angle(data[:, tone]), "Phase [rad]"),
                )
            for axis, (values, name) in zip(axes[row], columns):
                axis.plot(time_axis, values, lw=0.6, color="tab:blue")
                axis.set_ylabel(f"{name}\n{label}")
                axis.grid(True, alpha=0.3)

        for axis in axes[-1]:
            axis.set_xlabel(f"Time [{unit}]")
        return time_axis, scale, unit

    def _overlay_telegraph(
        self,
        axes,
        result: Any,
        *,
        time_axis: npt.NDArray[np.floating],
        scale: float,
        show_iq: bool,
    ) -> None:
        """Mark one tone's reconstructed branches, flips and bursts on its row of the grid.

        :param axes: The ``(left, right)`` axes of this tone's row.
        :param result: One :meth:`reconstruct_parity` result, i.e. ``qpd``'s.
        :param time_axis: Time axis the row was drawn against.
        :param scale: Seconds-to-plot-unit factor for that axis.
        :param show_iq: Whether the row holds I/Q (branch levels are drawn) or power/phase.

        """
        from ..analysis.parity import summarize

        if time_axis.size == 0:
            return
        if result.degenerate:
            # qpd's own verdict that the fitted model latched onto noise. Marking its flips
            # would be marking noise crossings, and the fidelity estimate stays high while it
            # does -- which is exactly why the flag exists and is read here.
            axes[0].set_title(
                f"qpd reports a degenerate model (contrast = {result.contrast:.1f}); "
                "flips not marked",
                fontsize=8,
                color="tab:red",
            )
            return

        window = time_axis.shape[0]
        branch = np.asarray(result.branch)[:window] > 0
        data = np.asarray(self.signal)[:window, result.tone]

        if show_iq and branch.any() and (~branch).any():
            # The decoded branch sequence, drawn as each channel's mean within each state --
            # the reconstruction as the readout would have looked without noise.
            for axis, values in ((axes[0], np.real(data)), (axes[1], np.imag(data))):
                levels = np.where(branch, values[branch].mean(), values[~branch].mean())
                axis.plot(time_axis, levels, color="tab:orange", lw=1.2, drawstyle="steps-post")

        limit = float(time_axis[-1])
        flips = np.asarray(result.flip_times) * scale
        flips = flips[flips <= limit]
        for axis in axes:
            if flips.size <= _MAX_FLIP_MARKS:
                for flip in flips:
                    axis.axvline(flip, color="0.25", lw=0.7, alpha=0.65)
            for burst in result.bursts:
                start = burst.t_start * scale
                if start > limit:
                    continue
                axis.axvspan(start, min(burst.t_end * scale, limit), color="tab:red", alpha=0.15)

        info = summarize(result, result.bursts)
        summary = (
            f"{info['n_flips']} flips, rate = {info['rate_hz']:.3g} Hz\n"
            f"contrast = {info['contrast']:.1f}, fidelity = {info['decoded_fidelity']:.3f}"
        )
        if info["n_bursts"]:
            summary += f"\n{info['n_bursts']} burst(s), shaded"
        elif flips.size > _MAX_FLIP_MARKS:
            summary += f"\nflip marks omitted above {_MAX_FLIP_MARKS}"
        axes[0].legend([summary], loc="best", fontsize=7, handlelength=0)

    def _analyze_constant(
        self,
        *,
        num_samples: Optional[int],
        title: Optional[str],
        tone: Optional[int],
        show_iq: bool,
        fit: bool,
        reconstruct: bool,
        quantity: str,
        **fit_kwargs: Any,
    ):
        """Annotate the stream grid with the parity reconstruction, over the fitted spectra.

        :param num_samples: Limit the time panels to this many samples.
        :param title: Figure title, or ``None`` for the default.
        :param tone: Which tone to reconstruct, or ``None`` for all of them.
        :param show_iq: Draw I/Q rather than power/phase in the grid.
        :param fit: Overlay the parity-model fit on the spectra.
        :param reconstruct: Mark the two-level reconstruction on the grid.
        :param quantity: Projection of the readout to threshold and spectrate.
        :param fit_kwargs: Passed to :func:`~daq.analysis.noise.fit_parity_psd`.
        :returns: The created figure.

        """
        import matplotlib.pyplot as plt

        from ..analysis.plotting import plot_psd

        from ..analysis.parity import qpd_available

        tones = self._tones(tone)
        if quantity == "proj" and not qpd_available():
            # The separation axis is qpd's to fit. Without it, say so and fall back to the
            # magnitude rather than failing -- announced, and the panel labels record which
            # projection was actually taken.
            print(
                "INFO: qpd is not installed, so the spectrum falls back to quantity='abs' "
                "(the maximal-separation axis needs qpd.reconstruction)."
            )
            quantity = "abs"
        # Projected before drawing anything, so an unknown quantity costs no figure.
        for index in tones:
            self._projection(tone=index, quantity=quantity)

        n_rows = len(tones)
        fig = plt.figure(figsize=(12, 2.2 * n_rows + 4.5), tight_layout=True)
        grid = fig.add_gridspec(n_rows + 1, 2, height_ratios=[1.0] * n_rows + [2.2])
        axes = np.empty((n_rows, 2), dtype=object)
        shared = None
        for row in range(n_rows):
            for column in range(2):
                axis = fig.add_subplot(grid[row, column], sharex=shared)
                shared = shared if shared is not None else axis
                axes[row, column] = axis
        ax_psd = fig.add_subplot(grid[n_rows, :])

        time_axis, scale, _ = self._draw_stream_grid(
            axes, tones=tones, num_samples=num_samples, show_iq=show_iq
        )

        if reconstruct:
            # No quantity here: qpd fits its own discrimination axis to the cloud. The
            # quantity below is what the *spectrum* is taken of, which is a separate choice.
            try:
                results = self.reconstruct_parity(tone=tone)
            except ImportError as err:
                print(f"INFO: skipping the parity reconstruction: {err}")
                results = []
            else:
                results = [results] if not isinstance(results, list) else results
            for row, result in enumerate(results):
                self._overlay_telegraph(
                    axes[row], result, time_axis=time_axis, scale=scale, show_iq=show_iq
                )

        f, psd = self.parity_psd(tone=tone, quantity=quantity)
        _, fits = plot_psd(
            f,
            psd,
            basis="electronic",
            labels=(_QUANTITY_LABELS[quantity], ""),
            f_bw=self.df,
            fit=fit,
            tone_labels=[self._tone_label(index) for index in tones],
            ax=ax_psd,
            **fit_kwargs,
        )
        # plot_psd fits the arrays it was handed rather than reading one off this object, so
        # what lands here describes the spectra that were actually drawn. Only when a fit was
        # asked for: analyze(fit=False) must not wipe an earlier fit_parity() result.
        if fit:
            self.fit_results = fits["a"]

        fig.suptitle(
            self._title("Constant-bias parity stream", tones[0]) if title is None else title
        )

        plt.show()
        return fig

    def _analyze_raw(
        self,
        *,
        num_samples: Optional[int] = None,
        title: Optional[str] = None,
        show_iq: bool = True,
    ):
        """Plot the time-stream data, using each tone's selected sideband.

        The fallback reconstruction, and the plot every stream had before the bias dispatch
        existed: one row per tone, I and Q (or power and phase) side by side.

        :param num_samples: Plot only this many leading samples.
        :param title: Figure title, or ``None`` for the default.
        :param show_iq: Show I and Q rather than power and phase.
        :returns: The created figure.

        """
        import matplotlib.pyplot as plt

        n_tones = np.asarray(self.signal).shape[1]
        fig, axes = plt.subplots(
            n_tones, 2, figsize=(12, 2 * n_tones), tight_layout=True, sharex=True, squeeze=False
        )
        self._draw_stream_grid(axes, num_samples=num_samples, show_iq=show_iq)

        plot_type = "I/Q Streams" if show_iq else "Power/Phase"
        fig.suptitle(f"{'TimeStream' if title is None else title} ({plot_type})")

        plt.show()
        return fig
