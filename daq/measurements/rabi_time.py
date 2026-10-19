"""Two-dimensional Rabi duration/amplitude sweep with square control pulses.

See https://www.intermod.pro/manuals/presto/tutorial_qubits/rabi_time.html.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ._pulsed_qubit import _PulsedQubit, _amplitudes, _array, _fit_trace


class RabiTime(_PulsedQubit):
    """Sweep square-pulse duration for each control amplitude.

    Frequencies are in Hz, times in seconds and amplitudes in full-scale fractions.
    Ports are one-based Presto-8 RF ports. ``device`` is required when running/saving.

    :param control_amp_arr: Increasing amplitudes in [0, 1]; a scalar gives a single row.
    :param control_duration_arr: Increasing positive pulse durations, in seconds.
    :param auto_fit: Fit the Rabi rate of each amplitude before saving (default True).
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
        control_duration_arr: ArrayLike,
        sample_duration: float,
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
        self.control_amp_arr = _amplitudes(control_amp_arr)
        self.control_duration_arr = _array(
            control_duration_arr, "control_duration_arr", positive=True
        )
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
        control = pls.setup_flat_pulse(
            self.control_port,
            group=0,
            duration=self.control_duration_arr[0],
            amplitude=1 + 1j,
            envelope=False,
        )
        time = 0.0
        for duration in self.control_duration_arr:
            control.set_total_duration(duration)
            pls.reset_phase(time, self.control_port)
            pls.output_pulse(time, control)
            time = self._readout(pls, readout, time + duration)
        pls.next_scale(time - self.wait_delay, self.control_port, group=0)
        return time, self.control_amp_arr.size

    def _shape(self) -> Tuple[int, ...]:
        return self.control_amp_arr.size, self.control_duration_arr.size

    def _fit(self, response: np.ndarray) -> Dict[str, Any]:
        rates, errors, curves, failures = [], [], [], []
        for row in response:
            try:
                fit = _fit_trace(self.control_duration_arr, row, damped=True)
                rates.append(fit["rate"])
                errors.append(fit["rate_err"])
                curves.append(fit["curve"])
                failures.append(None)
            except (ValueError, RuntimeError) as err:
                # Zero drive is a useful ground-state reference, but has no Rabi rate.
                rates.append(None)
                errors.append(None)
                curves.append(None)
                failures.append(str(err))
        if all(rate is None for rate in rates):
            raise ValueError("No amplitude row supports a Rabi fit")
        result = {
            "rabi_rate": rates,
            "rabi_rate_err": errors,
            "pi_duration": [None if rate is None else 0.5 / rate for rate in rates],
            "curve": curves,
            "fit_errors": failures,
        }
        valid = np.array([rate is not None for rate in rates])
        if np.count_nonzero(valid) >= 2:
            slope, intercept = np.polyfit(
                self.control_amp_arr[valid], np.asarray(rates, dtype=float)[valid], 1
            )
            result.update(rate_slope=float(slope), rate_intercept=float(intercept))
        return result

    def _plot(self, response: np.ndarray, result: Optional[Dict[str, Any]]) -> Any:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        times = self.control_duration_arr * 1e6
        if min(response.shape) > 1:
            mesh = axes[0].pcolormesh(times, self.control_amp_arr, response, shading="nearest")
            fig.colorbar(mesh, ax=axes[0], label="Readout quadrature (FS)")
            axes[0].set_ylabel("Control amplitude (FS)")
        else:
            for amp, row in zip(self.control_amp_arr, response):
                axes[0].plot(times, row, ".-", label=f"{amp:g} FS")
            if result is not None and response.shape[0] == 1:
                axes[0].plot(times, result["curve"][0], label="Fit")
            axes[0].set_ylabel("Readout quadrature (FS)")
            axes[0].legend()
        axes[0].set(xlabel="Control duration (µs)", title="Rabi time")
        if result is not None:
            rates = np.array(result["rabi_rate"], dtype=float)
            errors = np.array(result["rabi_rate_err"], dtype=float)
            axes[1].errorbar(self.control_amp_arr, rates * 1e-6, yerr=errors * 1e-6, fmt="o-")
            if "rate_slope" in result:
                axes[1].plot(
                    self.control_amp_arr,
                    (result["rate_slope"] * self.control_amp_arr + result["rate_intercept"]) * 1e-6,
                    "--",
                    label="Linear fit",
                )
                axes[1].legend()
        axes[1].set(xlabel="Control amplitude (FS)", ylabel="Rabi rate (MHz)")
        return fig
