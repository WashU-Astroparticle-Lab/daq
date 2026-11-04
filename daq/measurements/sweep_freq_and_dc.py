"""Frequency × chip DC bias maps using Presto's hardware mean/std summaries."""

from __future__ import annotations

import math
import warnings
from contextlib import ExitStack
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
from presto import lockin
from presto.utils import recommended_dac_config

from .._base import Base
from ..config import get_presto_address, get_presto_port
from ..instruments import Agilent33220A


class SweepFreqAndDC(Base):
    """Map mean response and sigma_IQ versus probe frequency and constant chip bias.

    ``bias_arr`` is in **chip volts**. The required ``voltage_divider`` converts these to
    Agilent settings: ``generator_bias_arr = bias_arr * voltage_divider``. Explicitly pass
    ``455`` for the usual 455:1 wiring. No Presto DC output is used and no trigger is asserted.

    ``freq_step`` sets the RF grid independently of ``sampling_frequency`` (default 10 kHz).
    ``acquisition_duration_s`` is the retained observation per point (default 0.5 s).
    ``discard_start_ms`` defaults to 25 ms; both durations round UP to whole summary blocks
    at the tuned rate. Presto's SDK already discards one block (2.16.0 and 2.17.1, no FIR),
    which counts towards that exclusion even when ``discard_start_ms=0``. Only summaries
    cross the connection; only one mean and three standard deviations per point are saved.

    At zero IF the readout is ``I_port.real + 1j * Q_port.real``, as in the original sweep.
    Variances include both within-block spread and movement of block means. ``summary_ddof``
    describes the HARDWARE block variance normalization (0: population, 1: sample); pooled
    results always use population normalization. Default 0 is an explicit assumption pending
    bench verification: the public Presto manual does not specify the variance divisor.

    ``bias_settle_s`` defaults to the Agilent driver's 0.5 s, once per bias row. This is
    separate from the startup exclusion at every frequency. RF and bias outputs are turned
    off on exit, including failure. ``device`` is required by the normal database save path.

    Migration: ``df`` remains a deprecated alias for ``freq_step`` only. Old ``bias_port``,
    ``bias_ramp_rate``, ``num_skip`` and ``num_averages`` arguments are replaced by the Agilent
    connection, chip voltages/divider, discard interval and retained duration. Old files can
    still be loaded/plotted, but cannot be run as an Agilent measurement without constructing
    a new object with explicit chip voltages and divider.
    """

    MAX_SUM_SAMPLES = 1023
    SDK_DISCARD_BLOCKS = 1  # get_pixels(summed=True, fir_coeffs=None), Presto 2.16/2.17

    def __init__(
        self,
        freq_center: float,
        freq_span: float,
        *,
        amp: float,
        bias_arr: npt.ArrayLike,
        output_port: int,
        input_port: int,
        voltage_divider: float,
        freq_step: Optional[float] = None,
        sampling_frequency: float = 10_000.0,
        acquisition_duration_s: float = 0.5,
        discard_start_ms: float = 25.0,
        bias_settle_s: float = Agilent33220A.DEFAULT_SETTLE_S,
        summary_ddof: int = 0,
        dither: bool = True,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
        df: Optional[float] = None,
    ) -> None:
        if df is not None:
            if freq_step is not None:
                raise ValueError("Specify freq_step or the legacy df alias, not both")
            warnings.warn(
                "df is deprecated: use freq_step for RF spacing and sampling_frequency for "
                "the lock-in sample rate",
                DeprecationWarning,
                stacklevel=2,
            )
            freq_step = df
        if freq_step is None:
            raise ValueError("freq_step (RF frequency spacing in Hz) is required")
        for name, value in (
            ("freq_center", freq_center),
            ("freq_step", freq_step),
            ("sampling_frequency", sampling_frequency),
            ("acquisition_duration_s", acquisition_duration_s),
            ("voltage_divider", voltage_divider),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("freq_span", freq_span),
            ("discard_start_ms", discard_start_ms),
            ("bias_settle_s", bias_settle_s),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if freq_center - freq_span / 2 <= 0:
            raise ValueError("The entire RF frequency range must be positive")
        if not np.isfinite(amp) or not 0 < amp < 1:
            raise ValueError("amp must be between 0 and 1 DAC full scale")
        if not 1 <= sampling_frequency <= 1e6:
            raise ValueError("sampling_frequency must be within Presto's 1 Hz–1 MHz range")
        if summary_ddof not in (0, 1):
            raise ValueError("summary_ddof must be 0 (population) or 1 (sample)")
        voltages = np.atleast_1d(np.asarray(bias_arr, dtype=float))
        if voltages.ndim != 1 or not voltages.size or not np.all(np.isfinite(voltages)):
            raise ValueError("bias_arr must be a nonempty finite 1-D array of CHIP voltages")
        if not np.all(np.isfinite(voltages * voltage_divider)):
            raise ValueError("Chip voltages times voltage_divider must be finite")

        self.freq_center = float(freq_center)
        self.freq_span = float(freq_span)
        self.freq_step = float(freq_step)
        self.amp = float(amp)
        self.bias_arr = voltages.copy()
        self.voltage_divider = float(voltage_divider)
        self.generator_bias_arr = self.bias_arr * self.voltage_divider
        self.bias_units = "chip V"
        self.bias_source = "Agilent33220A"
        self.output_port = output_port
        self.input_port = input_port
        self.sampling_frequency = float(sampling_frequency)
        self.acquisition_duration_s = float(acquisition_duration_s)
        self.discard_start_ms = float(discard_start_ms)
        self.bias_settle_s = float(bias_settle_s)
        self.summary_ddof = int(summary_ddof)
        self.dither = dither
        self.device, self.filter, self.notes = device, filter, notes
        self.schema_version = 2
        self._reset_results()
        self._set_timing(self.sampling_frequency)

    def _reset_results(self) -> None:
        self.freq_arr = None
        self.resp_arr = None
        self.std_i_arr = None
        self.std_q_arr = None
        self.std_iq_arr = None
        self.generator_bias_readback_arr = None

    def _set_timing(self, sample_rate: float) -> None:
        """Plan using the tuned rate, accounting for the SDK's discarded initial block."""
        self.actual_sampling_frequency = float(sample_rate)
        # Aim for one 25-ms block by default; for longer exclusions/rates, use multiple
        # blocks at the hardware limit. Even with discard=0 the SDK drops one block.
        target = max(self.discard_start_ms * 1e-3, 1 / sample_rate)
        self.samples_per_block = min(self.MAX_SUM_SAMPLES, max(2, math.ceil(target * sample_rate)))
        block_s = self.samples_per_block / sample_rate
        self.discard_blocks = max(
            self.SDK_DISCARD_BLOCKS, math.ceil(self.discard_start_ms * 1e-3 / block_s)
        )
        self.retained_blocks = max(1, math.ceil(self.acquisition_duration_s / block_s))
        self.retained_samples = self.retained_blocks * self.samples_per_block
        self.actual_discard_s = self.discard_blocks * block_s
        self.actual_acquisition_s = self.retained_samples / sample_rate
        self.total_acquisition_s = self.actual_discard_s + self.actual_acquisition_s
        if self.actual_acquisition_s < 0.2 and not getattr(self, "_warned_short", False):
            warnings.warn(
                f"Retained acquisition is {self.actual_acquisition_s:.6g} s per point "
                "(< 0.2 s); increase acquisition_duration_s to observe slower fluctuations.",
                UserWarning,
                stacklevel=3,
            )
            self._warned_short = True

    @staticmethod
    def _pool(means: npt.ArrayLike, stds: npt.ArrayLike, nsum: int, ddof: int = 0):
        """Equal-size block summaries -> population mean/std, including between-block spread."""
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
        if (
            not means.size
            or means.shape != stds.shape
            or not np.all(np.isfinite(means))
            or not np.all(np.isfinite(stds))
            or np.any(stds < 0)
        ):
            raise ValueError("Presto returned invalid mean/std summaries")
        mean = float(np.mean(means))
        variance = np.mean(stds**2 * ((nsum - ddof) / nsum) + (means - mean) ** 2)
        return mean, float(np.sqrt(variance))

    def run(
        self,
        bias: Optional[Agilent33220A] = None,
        *,
        presto_address: Optional[str] = None,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
        save_filename: Optional[str] = None,
    ) -> str:
        """Acquire summaries only; a supplied Agilent stays connected but is switched off."""
        if self.bias_source != "Agilent33220A" or self.voltage_divider is None:
            raise ValueError(
                "Legacy Presto-DC files are analysis-only. Construct a new SweepFreqAndDC "
                "with chip bias_arr and an explicit voltage_divider to run with the Agilent."
            )
        if getattr(self, "device", None) is None:
            raise ValueError("device is required for saving; set it before starting the sweep")
        self._reset_results()
        # Never let a re-run use a stale generator-voltage conversion.
        self.generator_bias_arr = self.bias_arr * self.voltage_divider
        start = self.freq_center - self.freq_span / 2
        count = int(math.floor(self.freq_span / self.freq_step + 1e-10)) + 1
        self.freq_arr = start + self.freq_step * np.arange(count)
        shape = (len(self.bias_arr), count)
        self.resp_arr = np.full(shape, np.nan + 1j * np.nan)
        self.std_i_arr = np.full(shape, np.nan)
        self.std_q_arr = np.full(shape, np.nan)
        self.std_iq_arr = np.full(shape, np.nan)
        self.generator_bias_readback_arr = np.full(len(self.bias_arr), np.nan)
        dac_mode, dac_fsample = recommended_dac_config(self.freq_center)
        params = dict(self.DC_PARAMS, dac_mode=dac_mode, dac_fsample=dac_fsample)

        with ExitStack() as stack:
            if bias is None:
                bias = stack.enter_context(Agilent33220A())
            # Registered before opening Presto so connection failures also switch bias off.
            stack.callback(setattr, bias, "output", False)
            lck = stack.enter_context(
                lockin.Lockin(
                    address=get_presto_address() if presto_address is None else presto_address,
                    port=get_presto_port() if presto_port is None else presto_port,
                    ext_ref_clk=ext_ref_clk,
                    **params,
                )
            )
            lck.hardware.set_adc_attenuation(self.input_port, self.ADC_ATTENUATION)
            lck.hardware.set_dac_current(self.output_port, self.DAC_CURRENT)
            lck.hardware.set_inv_sinc(self.output_port, 0)
            _, tuned_rate = lck.tune(0.0, self.sampling_frequency)
            self._set_timing(tuned_rate)
            lck.set_df(tuned_rate)
            lck.set_trigger_out([0])
            og = lck.add_output_group(self.output_port, 1)
            try:
                og.set_frequencies(0.0)
                og.set_amplitudes(self.amp)
                og.set_phases(0.0, -np.pi / 2)
                lck.set_dither(self.dither, self.output_port)
                ig = lck.add_input_group(self.input_port, 1)
                ig.set_frequencies(0.0)
                extra = self.discard_blocks - self.SDK_DISCARD_BLOCKS
                for jj, voltage in enumerate(self.generator_bias_arr):
                    bias.constant(float(voltage), settle_s=self.bias_settle_s)
                    state = bias.settings()
                    readback = float(state["offset_v"])
                    if not np.isfinite(readback) or not np.isclose(
                        readback, voltage, rtol=1e-4, atol=1e-6
                    ):
                        raise ValueError(
                            f"Agilent DC readback {readback} V differs from requested {voltage} V; "
                            "check chip voltages, divider and generator output range."
                        )
                    self.generator_bias_readback_arr[jj] = readback
                    self.attach(bias=state)
                    for ii, freq in enumerate(self.freq_arr):
                        lck.hardware.configure_mixer(
                            freq=float(freq), in_ports=self.input_port, out_ports=self.output_port
                        )
                        lck.apply_settings()
                        data = lck.get_pixels(
                            self.retained_blocks + extra,
                            summed=True,
                            nsum=self.samples_per_block,
                            fir_coeffs=None,
                        )
                        _, mean_i, std_i, mean_q, std_q = data[self.input_port]
                        arrays = [np.asarray(a) for a in (mean_i, std_i, mean_q, std_q)]
                        if any(a.shape != (self.retained_blocks + extra, 1) for a in arrays):
                            raise ValueError("Unexpected Presto summary shape")
                        mi, si, mq, sq = [a[extra:, 0].real for a in arrays]
                        i_mean, i_std = self._pool(
                            mi, si, self.samples_per_block, self.summary_ddof
                        )
                        q_mean, q_std = self._pool(
                            mq, sq, self.samples_per_block, self.summary_ddof
                        )
                        self.resp_arr[jj, ii] = i_mean + 1j * q_mean
                        self.std_i_arr[jj, ii] = i_std
                        self.std_q_arr[jj, ii] = q_std
                        self.std_iq_arr[jj, ii] = np.hypot(i_std, q_std)
                    print(
                        f"Bias {jj + 1}/{len(self.bias_arr)}: {self.bias_arr[jj]:.6g} chip V "
                        f"({voltage:.6g} Agilent V)"
                    )
            finally:
                try:
                    og.set_amplitudes(0.0)
                    lck.set_trigger_out([0])
                    lck.apply_settings()
                except Exception as err:
                    warnings.warn(f"Failed to mute Presto outputs: {err}", RuntimeWarning)
        return self.save(save_filename)

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    def _build_document(self, *args, **kwargs):
        """Keep the 2-D fluctuation grids in HDF5, just like the mean-response grid."""
        document = super()._build_document(*args, **kwargs)
        for key in ("std_i_arr", "std_q_arr", "std_iq_arr"):
            document.pop(key, None)
        return document

    @classmethod
    def load(cls, load_filename: str) -> "SweepFreqAndDC":
        """Load new summaries or historical response-only files without inventing a divider."""
        # Loading is deliberately not construction: legacy voltages are at the Presto output,
        # and neither their chip conversion nor an Agilent connection is known.
        self = cls.__new__(cls)
        self._reset_results()
        with h5py.File(load_filename, "r") as h5f:
            for key, value in h5f.attrs.items():
                setattr(self, key, value)
            for key, value in h5f.items():
                if key != "source_code":
                    setattr(self, key, value[()])
        if getattr(self, "schema_version", 1) < 2:
            self.schema_version = 1
            self.bias_source = "Presto DC"
            self.bias_units = "Presto output V"
            self.voltage_divider = None
            self.generator_bias_arr = None
            self.freq_step = float(self.df)
        return self

    def analyze(self, quantity: str = "amplitude", title: Optional[str] = None):
        """Plot mean response or ``std_i``, ``std_q``, ``std_iq`` (also ``sigma_iq``)."""
        import matplotlib.pyplot as plt

        if self.resp_arr is None or self.freq_arr is None:
            raise RuntimeError("Run or load the measurement before analyzing it")
        if quantity in ("std_i", "std_q", "std_iq", "sigma_iq"):
            key = "std_iq" if quantity == "sigma_iq" else quantity
            data = getattr(self, key + "_arr", None)
            if data is None:
                raise ValueError("This legacy file has no fluctuation statistics")
            label = {"std_i": "σI [FS]", "std_q": "σQ [FS]", "std_iq": "σIQ [FS]"}[key]
        elif quantity == "amplitude":
            data, label = np.abs(self.resp_arr), "Amplitude [FS]"
        elif quantity == "dB":
            with np.errstate(divide="ignore"):
                data = 20 * np.log10(np.abs(self.resp_arr))
            label = "Amplitude [dBFS]"
        elif quantity in ("phase", "group delay", "dpdb"):
            data = np.unwrap(np.unwrap(np.angle(self.resp_arr), axis=1), axis=0)
            label = "Phase [rad]"
            if quantity == "group delay":
                if len(self.freq_arr) < 2:
                    raise ValueError("Group delay needs at least two frequencies")
                data = -np.gradient(data, 2 * np.pi * self.freq_arr, axis=1) * 1e9
                label = "Group delay [ns]"
            elif quantity == "dpdb":
                if len(self.bias_arr) < 2 or len(np.unique(self.bias_arr)) != len(self.bias_arr):
                    raise ValueError("dpdb needs at least two distinct bias voltages")
                data = np.gradient(data, self.bias_arr, axis=0)
                label = f"dφ/dV [rad / {self.bias_units}]"
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
        # Sort only for plotting: data and saved acquisition order remain untouched. Explicit
        # cell edges support nonuniform biases and single-point scans, unlike old imshow.
        order = np.argsort(self.bias_arr)
        biases = self.bias_arr[order]
        if len(np.unique(biases)) != len(biases):
            raise ValueError("Heatmaps require distinct bias values")

        def edges(values, width):
            if len(values) == 1:
                return np.array([values[0] - width / 2, values[0] + width / 2])
            return np.r_[
                values[0] - (values[1] - values[0]) / 2,
                (values[:-1] + values[1:]) / 2,
                values[-1] + (values[-1] - values[-2]) / 2,
            ]

        finite = np.asarray(data)[np.isfinite(data)]
        limits = np.percentile(finite, [10, 90]) if finite.size else (None, None)
        fig, ax = plt.subplots(tight_layout=True)
        im = ax.pcolormesh(
            edges(self.freq_arr * 1e-9, self.freq_step * 1e-9),
            edges(biases, max(abs(biases[0]) * 0.1, 1e-6)),
            np.asarray(data)[order],
            shading="flat",
            vmin=limits[0],
            vmax=limits[1],
        )
        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel(f"Bias [{self.bias_units}]")
        if title is not None:
            ax.set_title(title)
        fig.colorbar(im, ax=ax, label=label)
        plt.show()
        return fig
