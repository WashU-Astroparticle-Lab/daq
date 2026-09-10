"""Sweep readout frequency to measure the spread of a QPD's QC trace."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Optional, Sequence

import h5py
import numpy as np

from .._base import Base
from ..instruments import Agilent33220A
from ..triggers import TriggerAny, resolve_trigger_states
from .qc_trace import QCTrace

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class StdDevSweep(Base):
    """Acquire a QC trace at each frequency and select the largest standard deviation.

    Each point uses :class:`QCTrace`, with the same gate ramp, drive amplitude and
    requested sample rate. Frequency spacing is independent of sample rate. The default
    statistic is the standard deviation along the principal axis of the folded,
    period-averaged I/Q trace. The axis is the eigenvector of the centered 2x2 I/Q
    covariance matrix with the largest eigenvalue; the statistic is the square root of
    that eigenvalue. The axis is fitted independently at each frequency, so rotating
    each trace's I/Q coordinates leaves the curve unchanged. These are ADC full-scale
    readout units, not farads. Separate I and Q standard deviations are also saved.

    ``trace_source="raw"`` instead uses the startup-trimmed time stream. ``quantity``
    selects the principal projection, complex IQ, magnitude, I or Q. In folded
    magnitude mode, I/Q is averaged first and the magnitude is taken afterwards.
    ``quantity="complex"`` measures ``sqrt(var(I) + var(Q))``: total spread across
    both axes, whereas ``"principal"`` measures only the direction of greatest spread.
    No resonator fit or parity reconstruction is required.

    :param freq_arr: Nonempty one-dimensional sequence of positive frequencies in Hz,
        acquired in the supplied order (e.g. ``np.linspace(fr - 1e6, fr + 1e6, 41)``).
    :param amp: Constant drive amplitude in DAC full scale, strictly between 0 and 1.
    :param output_port: Presto output port.
    :param input_port: Presto input port.
    :param ramp_vpp: Gate ramp peak-to-peak amplitude in volts.
    :param ramp_freq_hz: Gate ramp repetition frequency in Hz.
    :param ramp_offset_v: Gate ramp offset in volts; defaults to half ``ramp_vpp``.
    :param ramp_symmetry_pct: Ramp symmetry in percent, as in :class:`QCTrace`.
    :param sampling_frequency: Requested time-stream sample rate in Hz.
    :param num_periods: Number of ramp periods to acquire per frequency.
    :param discard_start_ms: Startup interval discarded by each time stream, in ms.
    :param trigger_states: Explicit Presto digital routing, or ``None`` to read it from
        the bias generator on every run, as in :class:`QCTrace`.
    :param dither: Whether to dither the Presto output.
    :param trace_source: ``"folded"`` (default) or ``"raw"``.
    :param quantity: ``"principal"`` (default), ``"complex"``, ``"abs"``, ``"real"``
        or ``"imag"``.
    :param ddof: Nonnegative integer subtracted from sample count in the variance divisor.
    :param device: Device name, required when running/saving for database logging.
    :param filter: Filter / amplifier chain description.
    :param notes: Free-text note, also included on each constituent QC trace.
    :raises ValueError: If the configuration is invalid.
    """

    _QC_PARAMETERS = (
        "amp",
        "output_port",
        "input_port",
        "ramp_vpp",
        "ramp_freq_hz",
        "ramp_offset_v",
        "ramp_symmetry_pct",
        "sampling_frequency",
        "num_periods",
        "discard_start_ms",
        "dither",
        "device",
        "filter",
        "notes",
    )

    def __init__(
        self,
        freq_arr: Sequence[float],
        amp: float,
        output_port: int,
        input_port: int,
        *,
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
    ) -> None:
        self.freq_arr = np.array(freq_arr, dtype=np.float64, copy=True)
        if (
            self.freq_arr.ndim != 1
            or not self.freq_arr.size
            or not np.all(np.isfinite(self.freq_arr))
            or np.any(self.freq_arr <= 0)
        ):
            raise ValueError("freq_arr must be a nonempty 1D sequence of positive finite Hz")
        if trace_source not in ("folded", "raw"):
            raise ValueError("trace_source must be 'folded' or 'raw'")
        if quantity not in ("principal", "complex", "abs", "real", "imag"):
            raise ValueError("quantity must be 'principal', 'complex', 'abs', 'real' or 'imag'")
        for name, value, minimum in (("ddof", ddof, 0), ("num_periods", num_periods, 1)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("amp", amp),
            ("ramp_vpp", ramp_vpp),
            ("ramp_freq_hz", ramp_freq_hz),
            ("ramp_offset_v", ramp_offset_v),
            ("ramp_symmetry_pct", ramp_symmetry_pct),
            ("sampling_frequency", sampling_frequency),
            ("discard_start_ms", discard_start_ms),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

        # Reuse QCTrace's configuration validation and defaults without touching hardware.
        template = QCTrace(
            readout_freq=float(self.freq_arr[0]),
            amp=amp,
            output_port=output_port,
            input_port=input_port,
            ramp_vpp=ramp_vpp,
            ramp_freq_hz=ramp_freq_hz,
            ramp_offset_v=ramp_offset_v,
            ramp_symmetry_pct=ramp_symmetry_pct,
            sampling_frequency=sampling_frequency,
            num_periods=int(num_periods),
            discard_start_ms=discard_start_ms,
            trigger_states=trigger_states,
            dither=dither,
            device=device,
            filter=filter,
            notes=notes,
        )
        for name in self._QC_PARAMETERS:
            setattr(self, name, getattr(template, name))
        self.trace_source = trace_source
        self.quantity = quantity
        self.ddof = int(ddof)
        samples = int(round(sampling_frequency / ramp_freq_hz))
        if trace_source == "raw":
            samples *= num_periods
        if samples < 2 or samples <= ddof:
            raise ValueError("The selected trace must contain at least two samples and exceed ddof")
        self._trigger_states_arg = (
            None if trigger_states is None else resolve_trigger_states(trigger_states)
        )
        self._reset_results()

    def _reset_results(self) -> None:
        self.std_arr = None
        self.std_i_arr = None
        self.std_q_arr = None
        self.std_principal_arr = None
        self.principal_axes = None
        self.best_freq = None
        self.best_std = None
        self.best_qc_file = None
        self.qc_files = []
        self.raw_files = []
        self.sample_counts = None
        self.sampling_frequencies = None
        self.trigger_states = None

    def standard_deviation(self, trace: QCTrace) -> float:
        """Evaluate the configured statistic on an acquired QC trace.

        :param trace: A run QCTrace; a loaded one suffices for ``trace_source="folded"``.
        :raises ValueError: If samples are nonfinite, too few, or have an invalid shape.
        :raises RuntimeError: If the chosen trace is unavailable.
        :returns: Standard deviation in ADC full-scale units.
        """
        value = float(np.std(self._series(trace), ddof=self.ddof))
        if not np.isfinite(value):
            raise ValueError("Nonfinite standard deviation")
        return value

    def _series(self, trace: QCTrace) -> np.ndarray:
        z = self._complex_series(trace)
        if self.quantity == "principal":
            axis = self._principal_axis(z)
            centered = z - z.mean()
            return axis[0] * centered.real + axis[1] * centered.imag
        if self.quantity == "abs":
            return np.abs(z)
        if self.quantity == "real":
            return z.real
        if self.quantity == "imag":
            return z.imag
        return z

    def _complex_series(self, trace: QCTrace) -> np.ndarray:
        if self.trace_source == "folded":
            if trace.avg_iq is None:
                raise RuntimeError("No folded QC trace available")
            iq = np.asarray(trace.avg_iq, dtype=np.float64)
            if iq.ndim != 2 or iq.shape[0] != 2 or not np.all(np.isfinite(iq)):
                raise ValueError("avg_iq must have shape (2, n_samples) and contain finite values")
            z = iq[0] + 1j * iq[1]
        else:
            if trace.qc_stream is None or trace.qc_stream.signal is None:
                raise RuntimeError("No raw QC time stream available")
            signal = np.asarray(trace.qc_stream.signal, dtype=np.complex128)
            if signal.ndim != 2 or signal.shape[1] != 1:
                raise ValueError("QC time stream must have shape (n_samples, 1)")
            z = signal[:, 0]
        if z.size < 2 or z.size <= self.ddof or not np.all(np.isfinite(z)):
            raise ValueError(
                "QC trace must contain finite samples, at least two and more than ddof"
            )
        return z

    @staticmethod
    def _principal_axis(z: np.ndarray) -> np.ndarray:
        centered = z - z.mean()
        iq = np.vstack((centered.real, centered.imag))
        # Scaling by N or N-ddof does not change the eigenvectors.
        covariance = (iq @ iq.T) / z.size
        if not np.all(np.isfinite(covariance)):
            raise ValueError("Nonfinite I/Q covariance")
        _, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, -1]
        # The sign has no effect on std; keep the saved direction deterministic.
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        return axis

    def run(
        self,
        bias: Optional[Agilent33220A] = None,
        *,
        presto_address: Optional[str] = None,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
        save_filename: Optional[str] = None,
    ) -> str:
        """Acquire every point, select the maximum, and save the sweep record.

        Each QCTrace saves its own folded record and raw TimeStream through their normal
        HDF5/MongoDB paths. Only one bias session is opened for the sweep. Its output is
        turned off on success or failure; caller-owned sessions remain open. A failed
        run leaves completed point files available, but no winning frequency. Ties choose
        the first frequency in acquisition order; an all-zero curve has no optimum.

        :param bias: Open bias generator, or ``None`` to discover/open one for this run.
        :param presto_address: Presto address, forwarded to each QCTrace.
        :param presto_port: Presto port, forwarded to each QCTrace.
        :param ext_ref_clk: Use the Presto external reference clock.
        :param save_filename: Path for the sweep summary only; generated if omitted.
        :returns: Saved sweep HDF5 path.
        :raises ValueError: If device is missing or a trace cannot yield a valid statistic.
        """
        self._reset_results()
        if self.device is None:
            raise ValueError("device parameter is required for database logging")
        n = len(self.freq_arr)
        self.std_arr = np.full(n, np.nan)
        self.std_i_arr = np.full(n, np.nan)
        self.std_q_arr = np.full(n, np.nan)
        self.std_principal_arr = np.full(n, np.nan)
        self.principal_axes = np.full((n, 2), np.nan)
        self.sample_counts = np.zeros(n, dtype=np.int64)
        self.sampling_frequencies = np.full(n, np.nan)
        with ExitStack() as stack:
            if bias is None:
                bias = stack.enter_context(Agilent33220A())
            stack.callback(setattr, bias, "output", False)
            for i, freq in enumerate(self.freq_arr):
                params = {name: getattr(self, name) for name in self._QC_PARAMETERS}
                step = f"Std dev sweep point {i + 1}/{n}, {freq:g} Hz"
                params["notes"] = step if self.notes is None else f"{self.notes} -- {step}"
                trace = QCTrace(
                    readout_freq=float(freq), trigger_states=self._trigger_states_arg, **params
                )
                path = trace.run(
                    bias,
                    presto_address=presto_address,
                    presto_port=presto_port,
                    ext_ref_clk=ext_ref_clk,
                )
                self.qc_files.append(path)
                self.raw_files.append(trace.qc_file)
                series = self._series(trace)
                value = float(np.std(series, ddof=self.ddof))
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite standard deviation at {freq:g} Hz")
                self.std_arr[i] = value
                z = self._complex_series(trace)
                self.std_i_arr[i] = np.std(z.real, ddof=self.ddof)
                self.std_q_arr[i] = np.std(z.imag, ddof=self.ddof)
                axis = self._principal_axis(z)
                centered = z - z.mean()
                self.std_principal_arr[i] = np.std(
                    axis[0] * centered.real + axis[1] * centered.imag, ddof=self.ddof
                )
                self.principal_axes[i] = axis
                self.sample_counts[i] = series.size
                self.sampling_frequencies[i] = trace.qc_stream.df
                self.trigger_states = np.array(trace.trigger_states, copy=True)

        self._select_best()
        return self.save(save_filename=save_filename)

    def _select_best(self) -> None:
        self.best_freq = self.best_std = self.best_qc_file = None
        if self.std_arr is None or not np.all(np.isfinite(self.std_arr)):
            return
        i = int(np.argmax(self.std_arr))
        if self.std_arr[i] > 0:
            self.best_freq = float(self.freq_arr[i])
            self.best_std = float(self.std_arr[i])
            self.best_qc_file = self.qc_files[i]

    def save(self, save_filename: Optional[str] = None) -> str:
        """Save a completed sweep via the normal HDF5/MongoDB path.

        :param save_filename: Explicit summary path, or ``None`` to generate one.
        :raises RuntimeError: If the sweep has not completed.
        :returns: Saved file path.
        """
        self._require_complete()
        return super()._save(__file__, save_filename=save_filename)

    def _require_complete(self) -> None:
        if (
            self.std_arr is None
            or self.std_arr.shape != self.freq_arr.shape
            or not np.all(np.isfinite(self.std_arr))
            or len(self.qc_files) != len(self.freq_arr)
        ):
            raise RuntimeError("No completed sweep available. Run or load the measurement first.")

    @classmethod
    def load(cls, load_filename: str) -> StdDevSweep:
        """Restore the summary without opening hardware or loading constituent files.

        Stored trigger routing describes the old acquisition; re-running reads the current
        generator's wiring, just as QCTrace.load() does.

        :param load_filename: Sweep HDF5 path.
        :returns: Sweep with its curve, optimum and constituent paths restored.
        """
        with h5py.File(load_filename, "r") as h5f:
            params = {name: h5f.attrs[name] for name in cls._QC_PARAMETERS if name in h5f.attrs}
            self = cls(
                freq_arr=h5f["freq_arr"][()],
                trace_source=h5f.attrs["trace_source"],
                quantity=h5f.attrs["quantity"],
                ddof=int(h5f.attrs["ddof"]),
                **params,
            )
            for name in (
                "std_arr",
                "std_i_arr",
                "std_q_arr",
                "std_principal_arr",
                "principal_axes",
                "sample_counts",
                "sampling_frequencies",
                "trigger_states",
            ):
                setattr(self, name, h5f[name][()])
            for name in ("qc_files", "raw_files"):
                setattr(self, name, h5f[name].asstr()[()].tolist())
        self._require_complete()
        self._select_best()
        return self

    def analyze(self, title: Optional[str] = None) -> Figure:
        """Plot frequency versus QC-trace standard deviation and mark the maximum.

        :param title: Optional figure title.
        :raises RuntimeError: If the sweep has not completed.
        :returns: Matplotlib figure; also displayed with ``plt.show()``.
        """
        self._require_complete()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(tight_layout=True)
        order = np.argsort(self.freq_arr, kind="stable")
        freqs = self.freq_arr[order] * 1e-9
        ax.plot(freqs, self.std_i_arr[order], ".-", label="I")
        ax.plot(freqs, self.std_q_arr[order], ".-", label="Q")
        ax.plot(freqs, self.std_principal_arr[order], ".-", label="Principal axis", linewidth=2)
        if self.quantity in ("complex", "abs"):
            label = "Combined I/Q" if self.quantity == "complex" else "Magnitude"
            ax.plot(freqs, self.std_arr[order], ".-", label=label, linewidth=2)
        if self.best_freq is not None:
            ax.axvline(
                self.best_freq * 1e-9,
                color="black",
                linestyle="--",
                label=f"Maximum: {self.best_freq / 1e9:.9g} GHz",
            )
            ax.plot(self.best_freq * 1e-9, self.best_std, "o", color="black")
        else:
            ax.text(
                0.5,
                0.95,
                "Zero variation: no optimum",
                transform=ax.transAxes,
                ha="center",
                va="top",
            )
        ax.legend()
        ax.set_xlabel("Readout frequency [GHz]")
        ax.set_ylabel("QC trace standard deviation [FS]")
        ax.set_title(
            title or f"QC trace std dev ({self.trace_source}, {self.quantity}, ddof={self.ddof})"
        )
        plt.show()
        return fig
