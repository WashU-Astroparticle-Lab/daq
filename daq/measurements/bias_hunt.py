# -*- coding: utf-8 -*-
"""Constant-gate-bias hunt for the largest parity contrast.

One ungated :class:`~daq.measurements.timestream.TimeStream` per candidate gate voltage, ranked
by the spread of the readout magnitude.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from ..instruments import Agilent33220A
from ._gate_bias import GateBiasMeasurement
from .timestream import TimeStream

FloatAny = Union[float, List[float], npt.NDArray[np.floating]]


class BiasHunt(GateBiasMeasurement):
    """Find the gate bias with the largest charge-parity contrast.

    The gate is parked at each entry of :attr:`bias_voltages` in turn and a single-tone time
    stream of :attr:`ts_duration_s` is recorded at :attr:`readout_freq`. Charge-parity
    switching moves the resonator between two frequencies, so a two-level-telegraph response
    shows up as a spread in the readout magnitude: ``std(|signal|)`` is the scalar used to rank
    the candidates, and the winner is the operating point to take parity spectra at.

    Nothing here is gated. The bias is a DC level written over SCPI before each acquisition,
    so the Presto trigger stays deasserted throughout -- the opposite of
    :class:`~daq.measurements.qc_trace.QCTrace`, whose ramp only runs while the trigger is
    asserted.

    Like ``QCTrace``, this measurement does **not** locate the resonance; read out where a
    fitted :class:`~daq.measurements.sweep.Sweep` says to.

    The measurement is fully specified at construction: the bias voltages are drawn in
    :meth:`__init__`, so the object (and its saved record) pin down exactly what was measured
    regardless of the random seed. Pass *bias_voltages* explicitly -- e.g. a
    ``numpy.linspace`` -- to scan the range systematically instead of sampling it.

    Requires the Presto **and** the 33220A over VISA; it cannot run without hardware.

    :param readout_freq: Readout frequency in hertz, normally a fitted ``fr``.
    :param amp: Drive amplitude in DAC full scale. Convert from dBm with
        :func:`~daq.calibrations.power_dbm_to_amp`.
    :param output_port: Presto output port.
    :param input_port: Presto input port.
    :param v_min: Lower bound of the random bias draw in volts. Required unless
        *bias_voltages* is given.
    :param v_max: Upper bound of the random bias draw in volts. Required unless
        *bias_voltages* is given.
    :param n_bias_try: Number of constant-bias tries to draw. Ignored when *bias_voltages* is
        given.
    :param bias_voltages: Explicit bias voltages to try, instead of a random draw.
    :param ts_duration_s: Length in seconds of each try, excluding the discarded start-up
        window.
    :param sampling_frequency: Time-stream sample rate in hertz.
    :param discard_start_ms: Leading milliseconds of start-up junk each time stream drops from
        its in-memory arrays. Passed through to
        :class:`~daq.measurements.timestream.TimeStream`.
    :param dither: Whether to dither the Presto output.
    :param seed: Seed for the random bias draw. The drawn voltages are saved either way, so
        this only matters for reproducing the draw itself.
    :param device: Device name, required for database logging.
    :param filter: Filter / amplifier chain description, for database logging.
    :param notes: Free-text note. Also prefixed onto each time stream's own note.
    :raises ValueError: If any parameter is out of range, or the draw bounds are missing.

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
        bias_voltages: Optional[FloatAny] = None,
        ts_duration_s: float = 5.0,
        sampling_frequency: float = 5e4,
        discard_start_ms: float = 25.0,
        dither: bool = True,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._init_readout(
            readout_freq=readout_freq,
            amp=amp,
            output_port=output_port,
            input_port=input_port,
            sampling_frequency=sampling_frequency,
            discard_start_ms=discard_start_ms,
            dither=dither,
            device=device,
            filter=filter,
            notes=notes,
        )

        if ts_duration_s <= 0:
            raise ValueError(f"ts_duration_s must be positive, got {ts_duration_s}")
        self.ts_duration_s = ts_duration_s

        # Draw here rather than in run(), so the measurement is fully specified -- and its
        # saved record exactly reproducible -- before any hardware is touched.
        self._seed = seed
        if bias_voltages is None:
            # No default range: the usable gate span is a property of the device and of what
            # the generator is wired through, and guessing it would put an arbitrary voltage
            # on somebody's sample. Name the bounds, or name the voltages.
            if v_min is None or v_max is None:
                raise ValueError(
                    "Give the bounds of the random bias draw (v_min and v_max, in volts), or "
                    "pass bias_voltages explicitly (e.g. numpy.linspace(0, 2, 20)). There is "
                    "no default gate range -- it depends on the device and the wiring."
                )
            if v_max < v_min:
                raise ValueError(f"v_max={v_max} is below v_min={v_min}")
            if n_bias_try < 1:
                raise ValueError(f"n_bias_try must be at least 1, got {n_bias_try}")
            rng = np.random.default_rng(seed)
            self.bias_voltages = rng.uniform(v_min, v_max, n_bias_try)
            self.v_min = float(v_min)
            self.v_max = float(v_max)
        else:
            self.bias_voltages = np.atleast_1d(np.asarray(bias_voltages, dtype=np.float64))
            if self.bias_voltages.size < 1:
                raise ValueError("bias_voltages must hold at least one voltage")
            # Report the span actually tried, so the record describes the measurement whether
            # the voltages were drawn or handed in.
            self.v_min = float(np.min(self.bias_voltages))
            self.v_max = float(np.max(self.bias_voltages))
        self.n_bias_try = int(self.bias_voltages.size)

        # Results - replaced by run()
        self.parity_contrast = None
        """``std(|signal|)`` for each entry of :attr:`bias_voltages`."""
        self.best_bias = None
        """Bias voltage with the largest parity contrast."""
        self.best_contrast = None
        """The largest parity contrast found."""
        self.bias_files = None
        """HDF5 path of each try's time stream, in :attr:`bias_voltages` order."""
        self.best_bias_file = None
        """Path of the winning try's HDF5 file."""

        # Constituent acquisitions, kept off the saved record (Base skips underscore-prefixed
        # attributes) and exposed through read-only properties.
        self._bias_streams: List[TimeStream] = []

    # ------------------------------------------------------------------ constituent objects

    @property
    def bias_streams(self) -> List[TimeStream]:
        """The constant-bias time streams, in :attr:`bias_voltages` order.

        :returns: One time stream per try; empty before :meth:`run`.

        """
        return self._bias_streams

    @property
    def best_bias_stream(self) -> Optional[TimeStream]:
        """The time stream with the largest parity contrast.

        This is the stream to feed to the parity analysis --
        :func:`~daq.analysis.noise.compute_psd` and
        :func:`~daq.analysis.noise.fit_parity_psd`.

        :returns: The winning time stream, or ``None`` before :meth:`run`.

        """
        if not self._bias_streams or self.parity_contrast is None:
            return None
        return self._bias_streams[int(np.nanargmax(self.parity_contrast))]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def parity_contrast_of(stream: TimeStream) -> float:
        """Return the parity contrast of a time stream.

        :param stream: A run single-tone time stream.
        :returns: ``std(|signal|)`` over the trimmed record.

        """
        return float(np.std(np.abs(np.asarray(stream.signal)[:, 0])))

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
        """Acquire one time stream per candidate bias and save the derived record.

        Every try saves its own HDF5 + MongoDB record through the normal path with
        ``attach(bias=...)`` applied, so each raw acquisition stays individually loadable. This
        measurement saves one further record holding the contrast-versus-bias curve, the
        winning bias and the constituent file paths.

        The gate-bias generator's output is forced off when the hunt finishes -- including on
        exception -- so no bias is left on the device. When *bias* is omitted the generator is
        opened and closed here; when it is passed in the caller keeps ownership of the VISA
        session and only the output is de-energised.

        As with :class:`~daq.measurements.qc_trace.QCTrace`, this ``run`` takes the *bias
        generator* as its first (and only) positional argument, not ``presto_address``; the
        Presto connection parameters are keyword-only.

        :param bias: An open :class:`~daq.instruments.function_generator.Agilent33220A`. When
            ``None``, one is discovered and opened for the duration of the run.
        :param presto_address: Presto address. Defaults to ``DAQ_PRESTO_ADDRESS``.
        :param presto_port: Presto port. Defaults to the presto default.
        :param ext_ref_clk: Whether to use an external reference clock.
        :param save_filename: Explicit path for this measurement's own HDF5 file.
        :returns: Path of this measurement's HDF5 file.

        """
        run_kwargs: Dict[str, Any] = dict(
            presto_address=presto_address,
            presto_port=presto_port,
            ext_ref_clk=ext_ref_clk,
        )

        with ExitStack() as stack:
            if bias is None:
                bias = stack.enter_context(Agilent33220A())
            else:
                # The caller owns the session, but never leave a bias on the gate.
                stack.callback(setattr, bias, "output", False)

            print(f"Bias hunt: reading out at {self.readout_freq / 1e9:.6f} GHz")

            ts_samples = self._stream_samples(self.ts_duration_s)
            contrasts = np.full(self.n_bias_try, np.nan)
            self._bias_streams = []
            paths: List[str] = []
            for ii, voltage in enumerate(self.bias_voltages):
                bias.constant(float(voltage))
                stream = self._make_timestream(
                    ts_samples,
                    # Ungated: the bias is a DC level already written over SCPI, so there is
                    # nothing for a trigger to start.
                    external_trigger=False,
                    notes=f"Constant bias {voltage:.4f} V (try {ii + 1}/{self.n_bias_try})",
                )
                stream.attach(bias=bias)
                paths.append(stream.run(**run_kwargs))

                contrasts[ii] = self.parity_contrast_of(stream)
                self._bias_streams.append(stream)
                print(
                    f"Bias try {ii + 1}/{self.n_bias_try}: {voltage:.4f} V, "
                    f"std(|signal|) = {contrasts[ii]:.4e}"
                )

            self.parity_contrast = contrasts
            self.bias_files = paths
            best = int(np.nanargmax(contrasts))
            self.best_bias = float(self.bias_voltages[best])
            self.best_contrast = float(contrasts[best])
            self.best_bias_file = paths[best]
            print(
                f"Max parity contrast at bias {self.best_bias:.4f} V, "
                f"std(|signal|) = {self.best_contrast:.4e}"
            )

        # Saved after the bias is de-energised, so a failure here cannot leave it applied.
        return self.save(save_filename=save_filename)

    def save(self, save_filename: Optional[str] = None) -> str:
        """Write this measurement's HDF5 file and MongoDB record.

        :param save_filename: Explicit path. Generated under ``DAQ_DATA_FOLDER`` when ``None``.
        :returns: Path of the written file.

        """
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "BiasHunt":
        """Rebuild a measurement from its saved HDF5 file.

        The contrast-versus-bias curve, the winner and the constituent file paths are restored.
        The time streams themselves are not: load them from :attr:`bias_files` with
        :meth:`TimeStream.load <daq.measurements.timestream.TimeStream.load>` when you need the
        raw records.

        :param load_filename: Path of the HDF5 file to load.
        :returns: The reconstructed measurement.

        """
        with h5py.File(load_filename, "r") as h5f:
            attrs = h5f.attrs

            self = cls(
                readout_freq=float(attrs["readout_freq"]),  # type: ignore
                amp=float(attrs["amp"]),  # type: ignore
                output_port=int(attrs["output_port"]),  # type: ignore
                input_port=int(attrs["input_port"]),  # type: ignore
                bias_voltages=h5f["bias_voltages"][()],  # type: ignore
                ts_duration_s=float(attrs["ts_duration_s"]),  # type: ignore
                sampling_frequency=float(attrs["sampling_frequency"]),  # type: ignore
                discard_start_ms=float(attrs["discard_start_ms"]),  # type: ignore
                dither=bool(attrs["dither"]),  # type: ignore
                device=attrs.get("device", None),
                filter=attrs.get("filter", None),
                notes=attrs.get("notes", None),
            )
            # Restored rather than recomputed from bias_voltages: when the voltages were drawn,
            # these are the bounds they were drawn from, which the extremes only approximate.
            self.v_min = float(attrs["v_min"])  # type: ignore
            self.v_max = float(attrs["v_max"])  # type: ignore

            self.best_bias = float(attrs["best_bias"]) if "best_bias" in attrs else None
            self.best_contrast = float(attrs["best_contrast"]) if "best_contrast" in attrs else None
            self.best_bias_file = attrs.get("best_bias_file", None)
            if "parity_contrast" in h5f:
                self.parity_contrast = h5f["parity_contrast"][()]  # type: ignore
            if "bias_files" in h5f:
                # h5py hands back bytes for a variable-length string dataset.
                self.bias_files = [
                    path.decode() if isinstance(path, bytes) else str(path)
                    for path in h5f["bias_files"][()]  # type: ignore
                ]

        return self

    # ------------------------------------------------------------------ analysis

    def analyze(self, title: Optional[str] = None):
        """Plot the parity contrast against gate bias, with the winner marked.

        :param title: Axis title. Defaults to naming the device and the readout frequency.
        :raises RuntimeError: If the measurement has not been run or loaded.
        :returns: The created figure.

        """
        if self.parity_contrast is None:
            raise RuntimeError("No bias hunt available. Run or load the measurement first.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)

        # Sort by voltage so a random draw still reads as a curve rather than a zigzag.
        order = np.argsort(self.bias_voltages)
        ax.plot(
            np.asarray(self.bias_voltages)[order],
            np.asarray(self.parity_contrast)[order],
            ".-",
            color="tab:green",
        )
        if self.best_bias is not None:
            ax.axvline(
                self.best_bias,
                color="tab:red",
                ls="--",
                lw=1.0,
                label=f"best bias {self.best_bias:.4f} V",
            )
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("Gate bias [V]")
        ax.set_ylabel(r"Parity contrast std($|S|$) [FS]")
        ax.grid(True, alpha=0.3)

        if title is None:
            parts = ["Bias hunt"]
            if self.device is not None:
                parts.append(str(self.device))
            parts.append(f"{self.readout_freq / 1e9:.6f} GHz")
            title = " -- ".join(parts)
        ax.set_title(title)

        plt.show()
        return fig
