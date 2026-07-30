# -*- coding: utf-8 -*-
"""
TimeStream measurement class for acquiring time-domain data with multiple frequencies.
"""

import warnings
from typing import Callable, List, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from presto import lockin
from presto.utils import untwist_downconversion

from .._base import Base
from ..config import get_presto_address, get_presto_port
from ..triggers import MAX_TRIGGER_PORTS, TriggerAny, resolve_trigger_states

FloatAny = Union[float, List[float], npt.NDArray[np.floating]]
BoolAny = Union[bool, List[bool], npt.NDArray[np.bool_]]

__all__ = ["MAX_TRIGGER_PORTS", "TimeStream", "TriggerAny"]


class TimeStream(Base):
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

        # Files written by run() hold the full acquisition, so re-apply the trim
        # to match the in-memory state produced by a live run.
        self._apply_discard_start()

        return self

    def analyze(
        self, num_samples: Optional[int] = None, title: Optional[str] = None, show_iq: bool = True
    ):
        """
        Plot the timestream data, using each tone's selected sideband.

        Parameters:
        -----------
        num_samples : int, optional
            Number of samples to plot. If None, plots all samples.
        title : str, optional
            Title for the plot.
        show_iq : bool, optional
            If True, show I and Q streams instead of phase and power. Default is True.
        """
        if self.signal is None:
            raise RuntimeError("No data available. Run the measurement first.")

        import matplotlib.pyplot as plt

        # Use the per-tone selected sideband
        data = self.signal
        freqs = self.signal_freqs

        # Limit number of samples if specified
        if num_samples is not None:
            data = data[:num_samples]

        # Create time axis
        time_axis = np.arange(data.shape[0]) / self.df * 1e6  # time in μs

        # Create figure with subplots for each frequency
        n_freqs = data.shape[1]
        fig, axes = plt.subplots(
            n_freqs, 2, figsize=(12, 2 * n_freqs), tight_layout=True, sharex=True
        )

        # Handle single frequency case
        if n_freqs == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_freqs):
            sb = "USB" if self.is_usb[i] else "LSB"
            freq_label = f"{freqs[i]/1e9:.3f} GHz ({sb})"
            if show_iq:
                # I stream plot
                i_stream = np.real(data[:, i])
                axes[i, 0].plot(time_axis, i_stream)
                axes[i, 0].set_ylabel(f"I [a.u.]\n{freq_label}")
                axes[i, 0].grid(True, alpha=0.3)

                # Q stream plot
                q_stream = np.imag(data[:, i])
                axes[i, 1].plot(time_axis, q_stream)
                axes[i, 1].set_ylabel(f"Q [a.u.]\n{freq_label}")
                axes[i, 1].grid(True, alpha=0.3)
            else:
                # Amplitude plot
                amplitudes = np.abs(data[:, i])
                power_db = 20.0 * np.log10(amplitudes)
                axes[i, 0].plot(time_axis, power_db)
                axes[i, 0].set_ylabel(f"Power [dBFS]\n{freq_label}")
                axes[i, 0].grid(True, alpha=0.3)

                # Phase plot
                phases = np.angle(data[:, i])
                axes[i, 1].plot(time_axis, phases)
                axes[i, 1].set_ylabel(f"Phase [rad]\n{freq_label}")
                axes[i, 1].grid(True, alpha=0.3)

        # Set x-labels for bottom plots
        axes[-1, 0].set_xlabel("Time [μs]")
        axes[-1, 1].set_xlabel("Time [μs]")

        # Set title
        plot_type = "I/Q Streams" if show_iq else "Power/Phase"
        if title is not None:
            fig.suptitle(f"{title} ({plot_type})")
        else:
            fig.suptitle(f"TimeStream ({plot_type})")

        plt.show()

        return fig
