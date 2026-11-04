"""Fast constant-bias optimization from on-hardware I/Q mean/std summaries."""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt

from .._base import Base
from ..instruments import Agilent33220A
from .sweep_freq_and_dc import SweepFreqAndDC


class BiasHunt(SweepFreqAndDC):
    """Find the candidate chip bias with the largest sigma_IQ at a fixed readout frequency.

    Uses the same hardware-summary acquisition as :class:`SweepFreqAndDC`, keeping one Presto
    session open throughout the hunt. No raw pixels or per-bias TimeStream files are created.
    The score is ``sqrt(var(I) + var(Q))``, including movement between block means. This is
    total readout fluctuation, not an SNR, proof of parity switching, or ``std(abs(signal))``.

    All voltages (explicit ``bias_voltages`` or random ``v_min``/``v_max``) are **chip volts**.
    ``voltage_divider`` is required; pass 455 explicitly for the usual wiring. Candidates are
    drawn uniformly at construction with optional ``seed``; rerunning uses the same voltages.
    There is no implicit range. ``best_bias`` is in chip volts and ``best_generator_bias`` is
    the corresponding Agilent setting. The winner is reported, not left applied.

    Defaults match the 2-D sweep: sampling_frequency=10 kHz, acquisition_duration_s=0.5 s,
    discard_start_ms=25, bias_settle_s=0.5. Actual durations round up to whole hardware blocks
    at the tuned rate, counting the SDK's hidden initial block. Retained duration below 0.2 s
    warns. ``summary_ddof=0`` assumes population-normalized hardware variance, pending bench
    verification; 1 supports sample-normalized summaries. Pooled outputs use ddof=0.

    ``resp_arr``, ``std_i_arr``, ``std_q_arr``, ``std_iq_arr`` and ``parity_contrast`` are 1-D
    in candidate order after success; the latter is a compatibility name for the sigma_IQ
    score, identified by saved ``contrast_metric``. Save one ``bias_hunt`` record containing
    the candidate voltages, statistics, timing and winner. Acquire a separate TimeStream at
    the winner for noise spectra or switching rates; those cannot be recovered from moments.

    Historical files still load with ``contrast_metric='abs'`` and generator-voltage labels,
    without inventing a divider. Their ``bias_files`` paths remain available for explicitly
    loading old TimeStreams. Legacy files cannot be rerun through the chip-voltage API.
    """

    def __init__(
        self,
        readout_freq: float,
        amp: float,
        output_port: int,
        input_port: int,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        n_bias_try: int = 20,
        bias_voltages: Optional[npt.ArrayLike] = None,
        *,
        voltage_divider: float,
        sampling_frequency: float = 10_000.0,
        acquisition_duration_s: float = 0.5,
        discard_start_ms: float = 25.0,
        bias_settle_s: float = Agilent33220A.DEFAULT_SETTLE_S,
        summary_ddof: int = 0,
        dither: bool = True,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        if bias_voltages is None:
            if v_min is None or v_max is None:
                raise ValueError("Supply v_min and v_max in CHIP volts, or explicit bias_voltages")
            if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max < v_min:
                raise ValueError("Bias bounds must be finite with v_min <= v_max")
            if (
                not np.isfinite(n_bias_try)
                or isinstance(n_bias_try, bool)
                or int(n_bias_try) != n_bias_try
                or n_bias_try < 1
            ):
                raise ValueError("n_bias_try must be a positive integer")
            voltages = np.random.default_rng(seed).uniform(v_min, v_max, int(n_bias_try))
        else:
            voltages = np.atleast_1d(np.asarray(bias_voltages, dtype=float))
            if voltages.ndim != 1 or not voltages.size or not np.all(np.isfinite(voltages)):
                raise ValueError("bias_voltages must be a nonempty finite 1-D array of CHIP volts")
            v_min, v_max = float(voltages.min()), float(voltages.max())

        super().__init__(
            freq_center=readout_freq,
            freq_span=0.0,
            freq_step=1.0,
            amp=amp,
            bias_arr=voltages,
            voltage_divider=voltage_divider,
            output_port=output_port,
            input_port=input_port,
            sampling_frequency=sampling_frequency,
            acquisition_duration_s=acquisition_duration_s,
            discard_start_ms=discard_start_ms,
            bias_settle_s=bias_settle_s,
            summary_ddof=summary_ddof,
            dither=dither,
            device=device,
            filter=filter,
            notes=notes,
        )
        self.readout_freq = float(readout_freq)
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.n_bias_try = int(voltages.size)
        self.seed = seed
        self.contrast_metric = "sigma_iq"

    @property
    def bias_voltages(self):
        """Candidate chip voltages, in acquisition order (generator volts in legacy files)."""
        return self.bias_arr

    def _reset_results(self) -> None:
        super()._reset_results()
        self.parity_contrast = None
        self.best_bias = None
        self.best_contrast = None
        self.best_generator_bias = None

    def run(
        self,
        bias: Optional[Agilent33220A] = None,
        *,
        presto_address: Optional[str] = None,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
        save_filename: Optional[str] = None,
    ) -> str:
        """Measure each candidate, rank sigma_IQ, and save once after switching outputs off."""
        if self.contrast_metric != "sigma_iq" or self.voltage_divider is None:
            raise ValueError(
                "Legacy BiasHunt files are analysis-only. Construct a new hunt with explicit "
                "chip voltages and voltage_divider."
            )
        self._reset_results()
        # Keep the single-frequency invariant even if readout_freq was changed in a notebook.
        self.freq_center, self.freq_span = self.readout_freq, 0.0
        self.n_bias_try = len(self.bias_voltages)
        self._acquire(
            bias, presto_address=presto_address, presto_port=presto_port, ext_ref_clk=ext_ref_clk
        )
        for key in ("resp_arr", "std_i_arr", "std_q_arr", "std_iq_arr"):
            setattr(self, key, getattr(self, key)[:, 0].copy())
        self.parity_contrast = self.std_iq_arr.copy()
        best = int(np.argmax(self.parity_contrast))  # first candidate wins an exact tie
        self.best_bias = float(self.bias_voltages[best])
        self.best_generator_bias = float(self.generator_bias_arr[best])
        self.best_contrast = float(self.parity_contrast[best])
        print(
            f"Best bias: {self.best_bias:.6g} chip V "
            f"({self.best_generator_bias:.6g} Agilent V), sigma_IQ={self.best_contrast:.6g} FS"
        )
        return self.save(save_filename)

    def save(self, save_filename: Optional[str] = None) -> str:
        return Base._save(self, __file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "BiasHunt":
        """Restore summary records or legacy magnitude-ranked results with original units."""
        self = cls.__new__(cls)
        self._reset_results()
        with h5py.File(load_filename, "r") as f:
            for key, value in f.attrs.items():
                setattr(self, key, value)
            for key, value in f.items():
                if key == "source_code":
                    continue
                if key == "bias_voltages":
                    self.bias_arr = value[()]
                elif key == "bias_files":
                    self.bias_files = value.asstr()[()].tolist()
                else:
                    setattr(self, key, value[()])
        if getattr(self, "schema_version", 1) < 2:
            self.schema_version = 1
            self.contrast_metric = "abs"
            self.bias_units = "Agilent V"
            self.bias_source = "Agilent33220A"
            self.voltage_divider = None
            self.generator_bias_arr = self.bias_arr.copy()
            self.best_generator_bias = self.best_bias
        return self

    def analyze(self, title: Optional[str] = None):
        """Plot the scores versus voltage, sorted for display, marking the winning candidate."""
        import matplotlib.pyplot as plt

        if self.parity_contrast is None or self.best_bias is None:
            raise RuntimeError("Run or load a completed BiasHunt before analyzing it")
        order = np.argsort(self.bias_voltages, kind="stable")
        fig, ax = plt.subplots(tight_layout=True)
        ax.plot(self.bias_voltages[order], self.parity_contrast[order], ".-")
        ax.plot(self.best_bias, self.best_contrast, "*", ms=12, label="Best candidate")
        ax.set_xlabel(f"Bias [{self.bias_units}]")
        ax.set_ylabel("σIQ [FS]" if self.contrast_metric == "sigma_iq" else "std(|signal|) [FS]")
        ax.set_title(
            title if title is not None else f"Bias hunt at {self.readout_freq / 1e9:g} GHz"
        )
        ax.legend()
        plt.show()
        return fig
