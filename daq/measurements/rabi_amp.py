"""Rabi amplitude calibration using sin² control pulses.

See https://www.intermod.pro/manuals/presto/tutorial_qubits/rabi_amp.html.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ._pulsed_qubit import _PulsedQubit, _amplitudes, _fit_trace, _integer, _positive


class RabiAmp(_PulsedQubit):
    """Sweep control amplitude at fixed pulse duration.

    Frequencies are in Hz, times in seconds and amplitudes in full-scale fractions.
    Ports are one-based Presto-8 RF ports. ``device`` is required when running/saving.

    :param control_amp_arr: Increasing amplitudes in [0, 1], at most 512 values.
    :param control_duration: Duration of one sin² control pulse.
    :param num_pulses: Number of identical back-to-back control pulses per readout.
    :param auto_fit: Fit before saving (default True).
    :param device: Device name for database logging.
    :param filter: Optional filter metadata.
    :param notes: Optional measurement notes.
    """

    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        readout_amp: float,
        control_amp_arr: ArrayLike,
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        num_pulses: int = 1,
        auto_fit: bool = True,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.control_amp_arr = _amplitudes(control_amp_arr)
        self.control_duration = _positive(control_duration, "control_duration")
        self.num_pulses = _integer(num_pulses, "num_pulses")
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
        pls.reset_phase(0.0, self.control_port)
        time = 0.0
        for _ in range(self.num_pulses):
            pls.output_pulse(time, control)
            time += self.control_duration
        time = self._readout(pls, readout, time)
        pls.next_scale(time - self.wait_delay, self.control_port, group=0)
        return time, self.control_amp_arr.size

    def _shape(self) -> Tuple[int, ...]:
        return (self.control_amp_arr.size,)

    def _fit(self, response: np.ndarray) -> Dict[str, Any]:
        result = _fit_trace(self.control_amp_arr, response)
        # The observed period contracts by num_pulses; report the amplitude of
        # ONE pi pulse, not the amplitude giving pi after the whole train.
        result["pi_amp"] = float(self.num_pulses / (2 * result["rate"]))
        result["pi_amp_err"] = result["pi_amp"] * result["rate_err"] / result["rate"]
        return result

    def _plot(self, response: np.ndarray, result: Optional[Dict[str, Any]]) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(self.control_amp_arr, response, ".", label="Readout")
        if result is not None:
            ax.plot(
                self.control_amp_arr,
                result["curve"],
                label=f"π amplitude = {result['pi_amp']:.5g} FS",
            )
        ax.set(
            xlabel="Control amplitude (FS)",
            ylabel="Readout quadrature (FS)",
            title="Rabi amplitude",
        )
        ax.legend()
        return fig
