"""Energy-relaxation time from a variable delay after a sin² pi pulse.

See https://www.intermod.pro/manuals/presto/tutorial_qubits/t1.html.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ._pulsed_qubit import _PulsedQubit, _array, _fit_trace, _positive


class T1(_PulsedQubit):
    """Sweep control-to-readout delay and fit an exponential plus offset.

    Frequencies are in Hz, times in seconds and amplitudes in full-scale fractions.
    Ports are one-based Presto-8 RF ports. ``device`` is required when running/saving.

    :param control_amp: Calibrated pi-pulse amplitude in (0, 1].
    :param control_duration: Duration of the calibrated sin² pi pulse.
    :param delay_arr: Increasing nonnegative delays in seconds.
    :param auto_fit: Fit T1 before saving (default True).
    :param device: Device name for database logging.
    :param filter: Optional filter metadata.
    :param notes: Optional measurement notes.
    """

    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        readout_amp: float,
        control_amp: float,
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        delay_arr: ArrayLike,
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        auto_fit: bool = True,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.control_amp = _positive(control_amp, "control_amp")
        if self.control_amp > 1:
            raise ValueError("control_amp must be at most 1")
        self.control_duration = _positive(control_duration, "control_duration")
        self.delay_arr = _array(delay_arr, "delay_arr")
        self._init_common(
            readout_freq,
            control_freq,
            readout_amp,
            readout_duration,
            sample_duration,
            readout_port,
            control_port,
            sample_port,
            wait_delay,
            readout_sample_delay,
            num_averages,
            auto_fit,
            device,
            filter,
            notes,
        )

    def _sequence(self, pls: Any, readout: Any) -> Tuple[float, int]:
        control = self._shaped_control(pls)
        time = 0.0
        for delay in self.delay_arr:
            pls.reset_phase(time, self.control_port)
            pls.output_pulse(time, control)
            time = self._readout(pls, readout, time + self.control_duration + delay)
        return time, 1

    def _shape(self) -> Tuple[int, ...]:
        return (self.delay_arr.size,)

    def _fit(self, response: np.ndarray) -> Dict[str, Any]:
        return _fit_trace(self.delay_arr, response, decay=True)

    def _plot(self, response: np.ndarray, result: Optional[Dict[str, Any]]) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(self.delay_arr * 1e6, response, ".", label="Readout")
        if result is not None:
            ax.plot(
                self.delay_arr * 1e6, result["curve"], label=f"T1 = {result['t1'] * 1e6:.4g} µs"
            )
        ax.set(
            xlabel="Control-to-readout delay (µs)",
            ylabel="Readout quadrature (FS)",
            title="Energy relaxation",
        )
        ax.legend()
        return fig
