# -*- coding: utf-8 -*-
"""Readout-frequency sweep of the quantum-capacitance trace's spread.

Single-tone or simultaneous multitone acquisition, ranked independently for each device by
the standard deviation of its folded trace along its principal axis.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt

from ..instruments import Agilent33220A
from ..triggers import TriggerAny, describe_trigger_states, resolve_trigger_states
from ._gate_bias import GateBiasMeasurement
from .qc_trace import QCTrace
from .timestream import TimeStream

TRACE_SOURCES = ("folded", "raw")
"""Which record the statistic is taken over: the folded period or the trimmed time stream."""

QUANTITIES = ("principal", "complex", "abs", "real", "imag")
"""Projections of the complex readout the standard deviation can be taken of."""


class StdDevSweep(GateBiasMeasurement):
    """Find the readout frequency at which the QC trace swings the most.

    For a 1-D sequence, a :class:`~daq.measurements.qc_trace.QCTrace` is taken at each entry
    of :attr:`readout_freqs` -- the same gate ramp, drive amplitude and sample rate every time --
    and each folded trace is reduced to one number, its standard deviation over the ramp
    period. The frequency with the largest spread is the operating point where the gate moves
    the resonator the most, i.e. the one to read the quantum capacitance out at.

    A 2-D frequency matrix acquires all columns simultaneously in one ``TimeStream`` per
    row. Each column follows one device and gets its own folded QC traces, statistics and
    optimum frequency. Existing 1-D calls retain flat curves and scalar winners.

    By default the spread is measured along the trace's **principal axis**: the folded I/Q
    samples are centred, their 2x2 covariance is diagonalised, and the standard deviation is
    the square root of the largest eigenvalue -- the spread along the direction the trace
    actually moves in. The axis is fitted afresh at each frequency, so a rotation of the I/Q
    plane (a cable-length change, a different LO phase) leaves the curve unchanged, where
    ``std(I)`` or ``std(Q)`` alone would not. This is the covariance's principal axis, not the
    two-blob discrimination axis :mod:`daq.analysis.parity` uses for a telegraph signal: a
    folded QC trace is a continuous curve, not two clusters. The separate I and Q spreads and
    the fitted axes are recorded alongside, whatever *quantity* is chosen.

    Like ``QCTrace``, this measurement does **not** locate the resonance: pass the candidate
    frequencies around a fitted ``fr``. It also does no calibration -- the statistic is in ADC
    full-scale units and compares frequencies against each other, not against a noise floor.

    Requires the Presto **and** the 33220A over VISA; it cannot run without hardware.

    :param readout_freqs: Readout frequencies in hertz, acquired in the order given, e.g.
        ``numpy.linspace(fr - 250e3, fr + 250e3, 51)``. A 2-D array with shape
        ``(n_steps, n_tones)`` enables simultaneous multitone acquisition: rows are acquired
        in order and each column follows one device. For common detunings use
        ``centers[None, :] + offsets[:, None]``. All tones share one bias ramp and RF port pair.
    :param amp: Drive amplitude in DAC full scale. For a multitone matrix, a scalar applies
        to EACH tone; alternatively pass one amplitude per column. The sum must be less than
        one; amplitudes are never split or rescaled. Convert from dBm with
        :func:`~daq.calibrations.power_dbm_to_amp`. Drive amplitudes stay fixed across rows.
    :param output_port: Presto output port.
    :param input_port: Presto input port.
    :param ramp_vpp: Ramp peak-to-peak amplitude in volts.
    :param ramp_freq_hz: Ramp repetition frequency in hertz.
    :param ramp_offset_v: Ramp DC offset in volts. Defaults to ``ramp_vpp / 2``, making the
        ramp unipolar-positive (the lab convention).
    :param ramp_symmetry_pct: Ramp symmetry in percent; ``100`` gives a ramp-up sawtooth.
    :param sampling_frequency: Time-stream sample rate in hertz. Pick a whole multiple of
        *ramp_freq_hz*, for the reason ``QCTrace`` warns about.
    :param num_periods: Number of whole ramp periods each point spans and averages over.
    :param discard_start_ms: Leading milliseconds of start-up junk each time stream drops from
        its in-memory arrays.
    :param trigger_states: Which Presto digital output ports gate each acquisition, as for
        :class:`~daq.measurements.qc_trace.QCTrace`. ``None`` (the default) reads the port off
        the bias generator on every run; an explicit routing is validated here, before any
        hardware is touched.
    :param dither: Whether to dither the Presto output.
    :param trace_source: ``"folded"`` (default) takes the statistic over the block-averaged
        ramp period; ``"raw"`` takes it over the trimmed time stream instead, which includes
        the noise the folding averages away.
    :param quantity: Which projection of the complex readout to take the spread of:
        ``"principal"`` (default), ``"complex"`` (``sqrt(var(I) + var(Q))``, the spread along
        both axes together), ``"abs"``, ``"real"`` (I) or ``"imag"`` (Q). For ``"abs"`` on a
        folded trace the I/Q average is taken first and the magnitude afterwards.
    :param ddof: Delta degrees of freedom of the standard deviation; ``0`` divides by ``N``.
    :param device: Device name, required for database logging.
    :param filter: Filter / amplifier chain description, for database logging.
    :param notes: Free-text note. Also prefixed onto each QC trace's own note.
    :param lo_freq: Fixed LO in hertz for multitone acquisition. Defaults to the midpoint
        of the entire frequency grid. Every tone must have ``abs(freq - lo_freq) < 500e6``.
        USB/LSB selection is automatic. Not used for a 1-D single-tone sweep.
    :param tone_labels: Optional display labels in column order; defaults to ``Tone 0``, etc.
        ``device`` remains the shared sample identifier for database logging.
    :raises ValueError: If any parameter is out of range.

    """

    def __init__(
        self,
        readout_freqs: Union[Sequence[float], Sequence[Sequence[float]]],
        amp: Union[float, Sequence[float]],
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
        trace_source: str = "folded",
        quantity: str = "principal",
        ddof: int = 0,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
        *,
        lo_freq: Optional[float] = None,
        tone_labels: Optional[Sequence[str]] = None,
    ) -> None:
        self.readout_freqs = np.array(readout_freqs, dtype=np.float64, copy=True)
        """Requested frequencies in Hz: ``(n_steps,)`` or ``(n_steps, n_tones)``."""
        if (
            self.readout_freqs.ndim not in (1, 2)
            or self.readout_freqs.size == 0
            or not np.all(np.isfinite(self.readout_freqs))
            or np.any(self.readout_freqs <= 0)
        ):
            raise ValueError("readout_freqs must be a non-empty 1-D or 2-D array of positive hertz")

        self.multitone = self.readout_freqs.ndim == 2
        self.n_tones = self.readout_freqs.shape[1] if self.multitone else 1
        amplitudes = np.asarray(amp, dtype=float)
        if self.multitone:
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
            if self.n_tones > 192:
                raise ValueError("Presto-8 supports at most 192 simultaneous tones")
            if any(np.unique(row).size != self.n_tones for row in self.readout_freqs):
                raise ValueError("Each acquisition row must contain distinct readout frequencies")
            self.lo_freq = float(
                (self.readout_freqs.min() + self.readout_freqs.max()) / 2
                if lo_freq is None
                else lo_freq
            )
            if not np.isfinite(self.lo_freq) or self.lo_freq <= 0:
                raise ValueError("lo_freq must be finite and positive")
            if np.any(np.abs(self.readout_freqs - self.lo_freq) >= 500e6):
                raise ValueError("Every tone must be within 500 MHz of lo_freq (exclusive)")
        elif amplitudes.ndim != 0 or lo_freq is not None:
            raise ValueError("A 1-D sweep requires scalar amp and no lo_freq override")
        if tone_labels is not None:
            if isinstance(tone_labels, str) or len(tone_labels) != self.n_tones:
                raise ValueError("tone_labels must contain one label per tone")
            if any(not isinstance(label, str) or not label for label in tone_labels):
                raise ValueError("tone_labels must be non-empty strings")
        self.tone_labels = (
            list(tone_labels)
            if tone_labels is not None
            else [f"Tone {tone}" for tone in range(self.n_tones)]
        )

        self._init_readout(
            readout_freq=float(self.readout_freqs.flat[0]),
            amp=float(amplitudes.flat[0]),
            output_port=output_port,
            input_port=input_port,
            sampling_frequency=sampling_frequency,
            discard_start_ms=discard_start_ms,
            dither=dither,
            device=device,
            filter=filter,
            notes=notes,
        )
        # A sweep has no single readout frequency. Each point's QCTrace owns the scalar, and
        # leaving one here would be saved -- and turned into a power_dbm -- as if it meant
        # something.
        del self.readout_freq
        if self.multitone:
            self.amp = amplitudes.copy()

        self._init_ramp(
            ramp_vpp=ramp_vpp,
            ramp_freq_hz=ramp_freq_hz,
            ramp_offset_v=ramp_offset_v,
            ramp_symmetry_pct=ramp_symmetry_pct,
            num_periods=num_periods,
        )

        if trace_source not in TRACE_SOURCES:
            raise ValueError(f"trace_source must be one of {TRACE_SOURCES}, got {trace_source!r}")
        if quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}, got {quantity!r}")
        if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
            raise ValueError(f"ddof must be a non-negative integer, got {ddof!r}")
        self.trace_source = trace_source
        self.quantity = quantity
        self.ddof = int(ddof)

        # The statistic needs at least two samples, and more than ddof of them. One folded
        # period holds round(fs / ramp) samples; the raw record holds num_periods of those.
        samples = int(round(self.sampling_frequency / self.ramp_freq_hz))
        if trace_source == "raw":
            samples *= self.num_periods
        if samples < 2 or samples <= self.ddof:
            raise ValueError(
                f"A {trace_source} trace holds only {samples} samples at "
                f"sampling_frequency={self.sampling_frequency:g} Hz and "
                f"ramp_freq_hz={self.ramp_freq_hz:g} Hz; the standard deviation needs at least "
                f"two and more than ddof={self.ddof}."
            )

        # As in QCTrace: an explicit routing is validated here and kept privately, so that a
        # re-run reads the caller's argument rather than the states a previous run resolved.
        self._trigger_states_arg = (
            None if trigger_states is None else QCTrace._check_trigger_states(trigger_states)
        )
        self.trigger_states = self._trigger_states_arg

        # Results - replaced by run()
        self.std_arr = None
        """Standard deviation of the chosen *quantity* at each of :attr:`readout_freqs`."""
        self.std_i_arr = None
        """``std(I)`` at each frequency, whatever *quantity* was chosen."""
        self.std_q_arr = None
        """``std(Q)`` at each frequency, whatever *quantity* was chosen."""
        self.std_principal_arr = None
        """Principal-axis standard deviation at each frequency, whatever *quantity* was chosen.

        Equal to :attr:`std_arr` for the default ``quantity="principal"``.
        """
        self.principal_axes = None
        """Unit ``[I, Q]`` principal axis at each frequency, shape ``(*readout_freqs.shape, 2)``.

        The sign is fixed so the larger component is positive; it carries no information.
        """
        self.sample_counts = None
        """Number of samples each standard deviation was taken over."""
        self.sampling_frequencies = None
        """The *tuned* sample rate each point was acquired at, in hertz."""
        self.best_freq = None
        """Winning frequency: scalar for 1-D input, array per column for 2-D input; first on ties."""
        self.best_std = None
        """Largest spread: scalar for 1-D input, array per column for 2-D input."""
        self.best_qc_file = None
        """Winning QC path: string for 1-D input, list per column for 2-D input."""
        self.qc_files = None
        """QC paths with the shape/order of :attr:`readout_freqs`; nested lists for multitone."""
        self.raw_files = None
        """One raw HDF5 path per acquisition row, shared by that row's tones."""

    def _reset_results(self) -> None:
        """Drop every result of a previous run, so a failed re-run leaves nothing stale.

        :attr:`std_arr` and :attr:`qc_files` are read together by :meth:`_select_best`, so
        they must never describe different runs.

        """
        self.std_arr = None
        self.std_i_arr = None
        self.std_q_arr = None
        self.std_principal_arr = None
        self.principal_axes = None
        self.sample_counts = None
        self.sampling_frequencies = None
        self.best_freq = None
        self.best_std = None
        self.best_qc_file = None
        self.qc_files = None
        self.raw_files = None

    # ------------------------------------------------------------------ helpers

    def statistics(self, trace: QCTrace) -> Dict[str, Any]:
        """Reduce one QC trace to the numbers :meth:`run` records for it.

        Everything is computed in one pass from the same complex series, so the ranked
        statistic and the diagnostic curves can never disagree about which samples they saw.

        :param trace: A run :class:`~daq.measurements.qc_trace.QCTrace`. A loaded one
            suffices for ``trace_source="folded"``; ``"raw"`` needs the live stream.
        :raises RuntimeError: If the trace does not carry the chosen record.
        :raises ValueError: If the record has the wrong shape, too few samples, or non-finite
            values.
        :returns: ``std`` (the chosen *quantity*), ``std_i``, ``std_q``, ``std_principal``,
            ``axis`` (the unit principal axis) and ``n_samples``.

        """
        z = self._complex_series(trace)
        axis = self._principal_axis(z)
        centered = z - z.mean()
        principal = axis[0] * centered.real + axis[1] * centered.imag

        if self.quantity == "principal":
            series = principal
        elif self.quantity == "abs":
            series = np.abs(z)
        elif self.quantity == "real":
            series = z.real
        elif self.quantity == "imag":
            series = z.imag
        else:  # "complex": numpy's std of a complex array is sqrt(var(I) + var(Q))
            series = z
        std = float(np.std(series, ddof=self.ddof))
        if not np.isfinite(std):
            raise ValueError("The standard deviation is not finite")

        return {
            "std": std,
            "std_i": float(np.std(z.real, ddof=self.ddof)),
            "std_q": float(np.std(z.imag, ddof=self.ddof)),
            "std_principal": float(np.std(principal, ddof=self.ddof)),
            "axis": axis,
            "n_samples": int(series.size),
        }

    def standard_deviation(self, trace: QCTrace) -> float:
        """Return the ranked statistic of one QC trace: ``std`` of :meth:`statistics`.

        :param trace: A run :class:`~daq.measurements.qc_trace.QCTrace`.
        :returns: The standard deviation of the chosen *quantity*, in ADC full scale.

        """
        return self.statistics(trace)["std"]

    def _complex_series(self, trace: QCTrace) -> npt.NDArray[np.complexfloating]:
        """Return the complex I/Q record the statistic is taken over.

        The folded trace is stored as two real rows (``avg_iq``) and the raw stream as a
        ``(n_samples, n_tones)`` complex array. The QC trace's stored ``tone`` selects
        the column; both come back as one complex vector.

        :param trace: A run QC trace.
        :raises RuntimeError: If the trace does not carry the chosen record.
        :raises ValueError: If the record has the wrong shape, fewer than two samples (or no
            more than *ddof*), or non-finite values.
        :returns: The complex samples.

        """
        if self.trace_source == "folded":
            if trace.avg_iq is None:
                raise RuntimeError("The QC trace has not been folded; run or load it first.")
            iq = np.asarray(trace.avg_iq, dtype=np.float64)
            if iq.ndim != 2 or iq.shape[0] != 2:
                raise ValueError(f"avg_iq must have shape (2, n_samples), got {iq.shape}")
            z = iq[0] + 1j * iq[1]
        else:
            stream = trace.qc_stream
            if stream is None or stream.signal is None:
                raise RuntimeError(
                    "The QC trace carries no time stream: load() does not restore it, so "
                    'trace_source="raw" needs a trace that has just been run.'
                )
            signal = np.asarray(stream.signal, dtype=np.complex128)
            tone = getattr(trace, "tone", 0)
            if signal.ndim != 2 or not 0 <= tone < signal.shape[1]:
                raise ValueError(
                    f"The QC time stream must have shape (n_samples, n_tones) with tone "
                    f"{tone} present, got {signal.shape}"
                )
            z = signal[:, tone]
        if z.size < 2 or z.size <= self.ddof:
            raise ValueError(
                f"The trace holds {z.size} samples; the standard deviation needs at least two "
                f"and more than ddof={self.ddof}."
            )
        if not np.all(np.isfinite(z)):
            raise ValueError("The trace contains non-finite samples")
        return z

    @staticmethod
    def _principal_axis(z: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.floating]:
        """Return the unit direction in the I/Q plane along which *z* spreads the most.

        The largest-eigenvalue eigenvector of the *centred* I/Q covariance. The divisor of the
        covariance (``N`` or ``N - ddof``) scales both eigenvalues alike and so does not move
        the eigenvector, which is why *ddof* plays no part here. An eigenvector's sign is
        arbitrary; it is fixed so the larger component is positive, purely so the saved axis is
        deterministic. When the two eigenvalues are equal the direction is ambiguous, but the
        spread along it is not.

        :param z: Complex samples.
        :raises ValueError: If the covariance is not finite.
        :returns: ``[I, Q]`` unit vector.

        """
        centered = z - z.mean()
        iq = np.vstack((centered.real, centered.imag))
        covariance = (iq @ iq.T) / z.size
        if not np.all(np.isfinite(covariance)):
            raise ValueError("The I/Q covariance is not finite")
        _, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, -1]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        return axis

    def _select_best(self) -> None:
        """Pick the frequency with the largest :attr:`std_arr`; the first one on an exact tie.

        Always picks one, as :class:`~daq.measurements.bias_hunt.BiasHunt` does: a curve that
        is zero everywhere names its first frequency with ``best_std = 0``, which
        :meth:`analyze` flags. Leaving the optimum unset would only turn into a save warning,
        since ``None`` has no HDF5 representation.

        """
        if self.multitone:
            best = np.argmax(self.std_arr, axis=0)
            tones = np.arange(self.n_tones)
            self.best_freq = self.readout_freqs[best, tones].copy()
            self.best_std = self.std_arr[best, tones].copy()
            self.best_qc_file = [self.qc_files[row][tone] for tone, row in enumerate(best)]
            return
        best = int(np.argmax(self.std_arr))
        self.best_freq = float(self.readout_freqs[best])
        self.best_std = float(self.std_arr[best])
        self.best_qc_file = self.qc_files[best]

    def _require_complete(self) -> None:
        """Raise unless every point of the sweep has a statistic and a file.

        :raises RuntimeError: If the sweep has not been run to completion, or loaded.

        """
        shape = self.readout_freqs.shape
        if (
            self.std_arr is None
            or np.shape(self.std_arr) != shape
            or not np.all(np.isfinite(self.std_arr))
            or self.qc_files is None
            or np.shape(self.qc_files) != shape
            or self.raw_files is None
            or len(self.raw_files) != shape[0]
        ):
            raise RuntimeError("No completed sweep available. Run or load the measurement first.")

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
        """Take one QC trace per readout frequency, rank them, and save the derived record.

        A 1-D sweep runs a :class:`~daq.measurements.qc_trace.QCTrace` at each frequency.
        A 2-D sweep takes one shared ``TimeStream`` per row and saves one folded ``QCTrace``
        per column, with its source tone index. All records use the normal save paths; this adds
        one summary record holding the curves, the winner and every constituent path. The
        gate-bias generator is opened once for the whole sweep and its output forced off at the
        end -- including on exception -- as ``QCTrace`` does. A point that fails aborts the
        sweep: the completed points' files remain on disk, but no winner is named.

        As for the other gate-bias measurements, *bias* is the only positional argument and
        the Presto connection parameters are keyword-only.

        :param bias: An open :class:`~daq.instruments.function_generator.Agilent33220A`. When
            ``None``, one is discovered and opened for the duration of the run.
        :param presto_address: Presto address. Defaults to ``DAQ_PRESTO_ADDRESS``.
        :param presto_port: Presto port. Defaults to the presto default.
        :param ext_ref_clk: Whether to use an external reference clock.
        :param save_filename: Explicit path for this measurement's own HDF5 file; the
            constituent files are always named automatically.
        :raises ValueError: If *device* is unset, checked before any hardware is touched, or a
            point's trigger routing gates no port.
        :returns: Path of this measurement's HDF5 file.

        """
        if self.device is None:
            # Base._save would raise the same after the last point; fail before the first.
            raise ValueError("device parameter is required for database logging")

        self._reset_results()
        run_kwargs: Dict[str, Any] = dict(
            presto_address=presto_address,
            presto_port=presto_port,
            ext_ref_clk=ext_ref_clk,
        )
        shape = self.readout_freqs.shape
        n = shape[0]
        std_arr = np.full(shape, np.nan)
        std_i_arr = np.full(shape, np.nan)
        std_q_arr = np.full(shape, np.nan)
        std_principal_arr = np.full(shape, np.nan)
        principal_axes = np.full((*shape, 2), np.nan)
        sample_counts = np.zeros(shape, dtype=np.int64)
        sampling_frequencies = np.full(n, np.nan)
        qc_files = []
        raw_files: List[str] = []

        with ExitStack() as stack:
            if bias is None:
                bias = stack.enter_context(Agilent33220A())
            else:
                # The caller owns the session, but never leave a bias on the gate.
                stack.callback(setattr, bias, "output", False)

            print(
                f"Std dev sweep: {n} acquisitions, {self.n_tones} tone(s), "
                f"{self.quantity} std of the {self.trace_source} trace"
            )
            for ii, freqs in enumerate(self.readout_freqs):
                traces, paths, raw_path = self._acquire_row(ii, freqs, bias, run_kwargs)
                qc_files.append(paths if self.multitone else paths[0])
                raw_files.append(raw_path)
                for tone, trace in enumerate(traces):
                    stats = self.statistics(trace)
                    index = (ii, tone) if self.multitone else ii
                    std_arr[index] = stats["std"]
                    std_i_arr[index] = stats["std_i"]
                    std_q_arr[index] = stats["std_q"]
                    std_principal_arr[index] = stats["std_principal"]
                    principal_axes[index] = stats["axis"]
                    sample_counts[index] = stats["n_samples"]
                sampling_frequencies[ii] = trace.qc_stream.df
                self.trigger_states = resolve_trigger_states(trace.trigger_states)
                print(f"Std dev sweep point {ii + 1}/{n}: std = {std_arr[ii]} FS")
                # Release the shared record before acquiring the next row.
                del trace, traces

            self.std_arr = std_arr
            self.std_i_arr = std_i_arr
            self.std_q_arr = std_q_arr
            self.std_principal_arr = std_principal_arr
            self.principal_axes = principal_axes
            self.sample_counts = sample_counts
            self.sampling_frequencies = sampling_frequencies
            self.qc_files = qc_files
            self.raw_files = raw_files
            self._select_best()
            print(
                f"Max std dev at {np.asarray(self.best_freq) / 1e9} GHz, "
                f"std = {self.best_std} FS "
                f"(gated on Presto digital output {describe_trigger_states(self.trigger_states)})"
            )

        # Saved after the bias is de-energised, so a failure here cannot leave it applied.
        return self.save(save_filename=save_filename)

    def _acquire_row(
        self,
        step: int,
        freqs: npt.ArrayLike,
        bias: Agilent33220A,
        run_kwargs: Dict[str, Any],
    ) -> Tuple[List[QCTrace], List[str], str]:
        """Acquire one sweep row and save a folded QC record for each tone.

        Multitone rows share one raw file; the QC records retain its column index so
        loading and re-folding a winning trace selects the same device.
        """
        traces = []
        amplitudes = np.atleast_1d(self.amp)
        for tone, freq in enumerate(np.atleast_1d(freqs)):
            traces.append(
                QCTrace(
                    readout_freq=float(freq),
                    amp=float(amplitudes[tone]),
                    output_port=self.output_port,
                    input_port=self.input_port,
                    ramp_vpp=self.ramp_vpp,
                    ramp_freq_hz=self.ramp_freq_hz,
                    ramp_offset_v=self.ramp_offset_v,
                    ramp_symmetry_pct=self.ramp_symmetry_pct,
                    sampling_frequency=self.sampling_frequency,
                    num_periods=self.num_periods,
                    discard_start_ms=self.discard_start_ms,
                    trigger_states=self._trigger_states_arg,
                    dither=self.dither,
                    device=self.device,
                    filter=self.filter,
                    notes=self._notes(
                        f"Std dev sweep point {step + 1}/{len(self.readout_freqs)}"
                        + (f", {self.tone_labels[tone]}" if self.multitone else "")
                    ),
                )
            )
        if not self.multitone:
            path = traces[0].run(bias, **run_kwargs)
            return traces, [path], traces[0].qc_file

        states = traces[0]._resolve_run_trigger_states(bias)
        signed_if = np.asarray(freqs) - self.lo_freq
        stream = TimeStream(
            lo_freq=self.lo_freq,
            if_freqs=np.abs(signed_if),
            is_usb=signed_if >= 0,
            amp=self.amp,
            df=self.sampling_frequency,
            pixel_counts=bias.samples_for_periods(
                self.num_periods,
                self.sampling_frequency,
                freq_hz=self.ramp_freq_hz,
                discard_ms=self.discard_start_ms,
            ),
            output_port=self.output_port,
            input_port=self.input_port,
            dither=self.dither,
            device=self.device,
            filter=self.filter,
            notes=self._notes(f"Multitone std dev sweep point {step + 1}"),
            external_trigger=states,
            discard_start_ms=self.discard_start_ms,
        )
        try:
            bias.sawtooth(
                vpp=self.ramp_vpp,
                freq_hz=self.ramp_freq_hz,
                offset_v=self.ramp_offset_v,
                symmetry_pct=self.ramp_symmetry_pct,
                gated=True,
            )
            stream.attach(bias=bias)
            raw_path = stream.run(**run_kwargs)
        finally:
            bias.output = False
        paths = []
        for tone, trace in enumerate(traces):
            trace._qc_stream = stream
            trace.qc_file = raw_path
            trace.trigger_states = states.copy()
            trace.fold(tone=tone)
            paths.append(trace.save())
        return traces, paths, raw_path

    def save(self, save_filename: Optional[str] = None) -> str:
        """Write this measurement's HDF5 file and MongoDB record.

        :param save_filename: Explicit path. Generated under ``DAQ_DATA_FOLDER`` when ``None``.
        :raises RuntimeError: If the sweep has not been run to completion, or loaded.
        :returns: Path of the written file.

        """
        self._require_complete()
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "StdDevSweep":
        """Rebuild a measurement from its saved HDF5 file.

        The curves, the winner and the constituent paths are restored; the QC traces
        themselves are not. Load one from :attr:`qc_files` with
        :meth:`QCTrace.load <daq.measurements.qc_trace.QCTrace.load>` when you need it. As for
        ``QCTrace``, the stored trigger routing is restored for inspection but not pinned:
        re-running a loaded sweep reads the generator in front of you now.

        :param load_filename: Path of the HDF5 file to load.
        :returns: The reconstructed measurement.

        """
        with h5py.File(load_filename, "r") as h5f:
            attrs = h5f.attrs

            # Files written by the first draft of this class stored the axis as freq_arr; it
            # was renamed because Base._build_document skips that name as Sweep's data array.
            axis_name = "readout_freqs" if "readout_freqs" in h5f else "freq_arr"
            self = cls(
                readout_freqs=h5f[axis_name][()],  # type: ignore
                amp=h5f["amp"][()] if "amp" in h5f else float(attrs["amp"]),  # type: ignore
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
                trace_source=str(attrs["trace_source"]),
                quantity=str(attrs["quantity"]),
                ddof=int(attrs["ddof"]),  # type: ignore
                device=attrs.get("device", None),
                filter=attrs.get("filter", None),
                notes=attrs.get("notes", None),
                lo_freq=attrs.get("lo_freq", None),
                tone_labels=(
                    h5f["tone_labels"].asstr()[()].tolist() if "tone_labels" in h5f else None
                ),
            )

            if "trigger_states" in h5f:
                self.trigger_states = resolve_trigger_states(h5f["trigger_states"][()])  # type: ignore
            for name in (
                "std_arr",
                "std_i_arr",
                "std_q_arr",
                "std_principal_arr",
                "principal_axes",
                "sample_counts",
                "sampling_frequencies",
            ):
                if name in h5f:
                    setattr(self, name, h5f[name][()])  # type: ignore
            for name in ("qc_files", "raw_files"):
                if name in h5f:
                    setattr(self, name, h5f[name].asstr()[()].tolist())

        # The winner is a function of the curve, so it is recomputed rather than trusted.
        self._require_complete()
        self._select_best()
        return self

    # ------------------------------------------------------------------ analysis

    def analyze(self, title: Optional[str] = None):
        """Plot the standard deviation against readout frequency, with the winner marked.

        The I, Q and principal-axis curves are always drawn; the ranked curve is drawn on top
        when it is neither of those (``quantity="complex"`` or ``"abs"``). Frequencies are
        plotted in ascending order whatever order they were acquired in. Multitone scans
        have one labeled panel per column, with an independently selected maximum.

        :param title: Figure title. Defaults to naming the device and the statistic.
        :raises RuntimeError: If the sweep has not been run to completion, or loaded.
        :returns: The created figure.

        """
        self._require_complete()

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            self.n_tones,
            1,
            squeeze=False,
            figsize=(8, 4 * self.n_tones),
            tight_layout=True,
        )
        for tone, ax in enumerate(axes[:, 0]):
            freqs = self.readout_freqs[:, tone] if self.multitone else self.readout_freqs
            curves = {
                name: getattr(self, name)[:, tone] if self.multitone else getattr(self, name)
                for name in ("std_i_arr", "std_q_arr", "std_principal_arr", "std_arr")
            }
            best_freq = self.best_freq[tone] if self.multitone else self.best_freq
            best_std = self.best_std[tone] if self.multitone else self.best_std
            order = np.argsort(freqs, kind="stable")
            freqs_ghz = freqs[order] * 1e-9
            ax.plot(freqs_ghz, curves["std_i_arr"][order], ".-", label="I")
            ax.plot(freqs_ghz, curves["std_q_arr"][order], ".-", label="Q")
            ax.plot(
                freqs_ghz, curves["std_principal_arr"][order], ".-", label="Principal axis", lw=2
            )
            if self.quantity in ("complex", "abs"):
                label = "Combined I/Q" if self.quantity == "complex" else "Magnitude"
                ax.plot(freqs_ghz, curves["std_arr"][order], ".-", label=label, lw=2)
            ax.axvline(
                best_freq * 1e-9,
                color="black",
                linestyle="--",
                label=f"Maximum: {best_freq / 1e9:.6f} GHz",
            )
            ax.plot(best_freq * 1e-9, best_std, "o", color="black")
            if best_std == 0:
                ax.text(
                    0.5,
                    0.95,
                    "Zero variation everywhere: the maximum is not meaningful",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                )
            ax.legend()
            ax.set_xlabel("Readout frequency [GHz]")
            ax.set_ylabel("QC trace standard deviation [FS]")
            if self.multitone:
                ax.set_title(self.tone_labels[tone])

        if title is None:
            parts = [f"QC trace std dev ({self.quantity}, {self.trace_source}, ddof={self.ddof})"]
            if self.device is not None:
                parts.append(str(self.device))
            title = " -- ".join(parts)
        if self.multitone:
            fig.suptitle(title)
        else:
            axes[0, 0].set_title(title)

        plt.show()
        return fig
