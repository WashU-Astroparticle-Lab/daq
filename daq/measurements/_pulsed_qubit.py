"""Shared acquisition, persistence and fitting for the Presto qubit tutorials.

Sequences follow https://www.intermod.pro/manuals/presto/tutorial_qubits/.
The fits operate on integrated, principal-axis-rotated complex readout, not on a
calibrated excited-state probability.
"""

import inspect
import json
import warnings
from typing import Any, Dict, Optional, Tuple, TypeVar

import h5py
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from .._base import Base
from ..config import get_presto_address, get_presto_port

QubitMeasurement = TypeVar("QubitMeasurement", bound="_PulsedQubit")


def _array(value: ArrayLike, name: str, *, positive: bool = False) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=float)).copy()
    if arr.ndim != 1 or not arr.size or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a nonempty, finite, one-dimensional array")
    if np.any(arr <= 0 if positive else arr < 0):
        raise ValueError(f"{name} must contain {'positive' if positive else 'nonnegative'} values")
    if arr.size > 1 and np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _integer(value: int, name: str, maximum: Optional[int] = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be between 1 and {maximum or 'the hardware limit'}")
    return value


def _amplitudes(value: ArrayLike) -> np.ndarray:
    arr = _array(value, "control_amp_arr")
    if arr.max() > 1 or arr.size > 512:
        raise ValueError("control_amp_arr must be within [0, 1] with at most 512 entries")
    return arr


def _fit_trace(
    x: np.ndarray, y: np.ndarray, *, decay: bool = False, damped: bool = False
) -> Dict[str, Any]:
    """Fit in dimensionless coordinates to keep nanosecond axes well conditioned."""
    minimum = 4 if decay else 6
    if x.size < minimum or np.any(np.diff(x) <= 0):
        raise ValueError(f"Fitting requires at least {minimum} distinct increasing sweep points")
    if not np.all(np.isfinite(y)) or np.ptp(y) <= 1e-12 * max(1.0, np.max(np.abs(y))):
        raise ValueError("Readout has no finite, resolvable contrast")
    span = float(np.ptp(x))
    u = (x - x[0]) / span
    center, scale = float(np.mean(y)), float(np.ptp(y))
    v = (y - center) / scale
    if decay:

        def model(t, offset, amplitude, tau):
            return offset + amplitude * np.exp(-t / tau)

        p, cov = curve_fit(
            model,
            u,
            v,
            p0=[v[-1], v[0] - v[-1], 0.3],
            bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 1e3]),
            maxfev=20000,
        )
        result = {"t1": float(p[2] * span), "t1_err": float(np.sqrt(cov[2, 2]) * span)}
    else:

        def model(t, offset, cosine, sine, rate, damping=0.0):
            return offset + np.exp(-damping * t) * (
                cosine * np.cos(2 * np.pi * rate * t) + sine * np.sin(2 * np.pi * rate * t)
            )

        # A frequency scan with a linear solve seeds the nonlinear fit, including
        # nonuniform sweeps. Limit the search by the largest sampling gap.
        upper = 0.5 / np.max(np.diff(u))
        rates = np.linspace(0.25, upper, max(128, 16 * x.size))
        candidates = []
        for rate in rates:
            matrix = np.column_stack(
                [np.ones_like(u), np.cos(2 * np.pi * rate * u), np.sin(2 * np.pi * rate * u)]
            )
            coeff, *_ = np.linalg.lstsq(matrix, v, rcond=None)
            candidates.append((np.sum((matrix @ coeff - v) ** 2), rate, coeff))
        _, rate, coeff = min(candidates, key=lambda item: item[0])
        initial = [*coeff, rate]
        lower = [-np.inf, -np.inf, -np.inf, 0.01]
        higher = [np.inf, np.inf, np.inf, upper]
        if damped:
            initial.append(1.0)
            lower.append(0.0)
            higher.append(100.0)
        p, cov = curve_fit(
            model,
            u,
            v,
            p0=initial,
            bounds=(lower, higher),
            maxfev=20000,
        )
        result = {"rate": float(p[3] / span), "rate_err": float(np.sqrt(cov[3, 3]) / span)}
        if damped:
            result["damping_rate"] = float(p[4] / span)
    if not np.all(np.isfinite(cov)):
        raise ValueError("Fit uncertainty could not be estimated")
    result["curve"] = (center + scale * model(u, *p)).tolist()
    return result


class _PulsedQubit(Base):
    """Internal common lifecycle; concrete classes define the sweep and sequence."""

    def _init_common(
        self,
        readout_freq: float,
        control_freq: float,
        readout_amp: float,
        readout_duration: float,
        sample_duration: float,
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        auto_fit: bool,
        device: Optional[str],
        filter: Optional[str],
        notes: Optional[str],
    ) -> None:
        self.readout_freq = _positive(readout_freq, "readout_freq")
        self.control_freq = _positive(control_freq, "control_freq")
        self.readout_amp = _positive(readout_amp, "readout_amp")
        if self.readout_amp > 1:
            raise ValueError("readout_amp must be at most 1")
        self.readout_duration = _positive(readout_duration, "readout_duration")
        self.sample_duration = _positive(sample_duration, "sample_duration")
        self.readout_port = _integer(readout_port, "readout_port", 8)
        self.control_port = _integer(control_port, "control_port", 8)
        self.sample_port = _integer(sample_port, "sample_port", 8)
        if self.readout_port == self.control_port:
            raise ValueError("readout_port and control_port must be different")
        self.wait_delay = _positive(wait_delay, "wait_delay")
        self.readout_sample_delay = _positive(
            readout_sample_delay, "readout_sample_delay", zero=True
        )
        self.num_averages = _integer(num_averages, "num_averages", 65536)
        self.auto_fit = bool(auto_fit)
        self.device, self.filter, self.notes = device, filter, notes
        self.t_arr = self.store_arr = self.resp_arr = self.fit_results = None

    def _quantize(self, pls: Any) -> None:
        self.clock_period = float(pls.get_clk_T())
        for name in (
            "readout_duration",
            "sample_duration",
            "control_duration",
            "control_duration_arr",
            "delay_arr",
            "wait_delay",
            "readout_sample_delay",
        ):
            if not hasattr(self, name):
                continue
            value = np.asarray(getattr(self, name))
            actual = np.rint(value / self.clock_period) * self.clock_period
            allow_zero = name in ("delay_arr", "readout_sample_delay")
            if np.any(actual < 0 if allow_zero else actual <= 0):
                raise ValueError(f"{name} rounds to zero on the Presto timing grid")
            if actual.ndim and actual.size > 1 and np.any(np.diff(actual) <= 0):
                raise ValueError(f"{name} contains duplicate points on the Presto timing grid")
            setattr(self, name, actual if actual.ndim else float(actual))

    def _shaped_control(self, pls: Any) -> Any:
        from presto.utils import sin2

        count = int(round(self.control_duration * pls.get_fs("dac")))
        if count < 3:
            raise ValueError("control_duration is too short for a sin² pulse")
        envelope = sin2(count)
        return pls.setup_template(
            self.control_port, group=0, template=envelope + 1j * envelope, envelope=False
        )

    def _readout(self, pls: Any, pulse: Any, time: float) -> float:
        pls.reset_phase(time, self.readout_port)
        pls.output_pulse(time, pulse)
        pls.store(time + self.readout_sample_delay)
        # Sampling can outlast the readout pulse. Finish both before waiting for
        # relaxation, so the next shot cannot contaminate the current record.
        return (
            time
            + max(self.readout_duration, self.readout_sample_delay + self.sample_duration)
            + self.wait_delay
        )

    def run(
        self,
        presto_address: Optional[str] = None,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
        save_filename: Optional[str] = None,
    ) -> str:
        """Acquire, optionally fit, and save with the package's HDF5/DB lifecycle.

        :param presto_address: Override ``DAQ_PRESTO_ADDRESS``.
        :param presto_port: Override ``DAQ_PRESTO_PORT``.
        :param ext_ref_clk: Use the external 10 MHz reference clock.
        :param save_filename: Optional explicit HDF5 output path.
        :returns: Saved HDF5 filename. Fit failures warn but preserve raw data.
        """
        from presto import pulsed

        if self.device is None:
            raise ValueError("device parameter is required for database logging")
        self.t_arr = self.store_arr = self.resp_arr = self.fit_results = None
        with pulsed.Pulsed(
            address=get_presto_address() if presto_address is None else presto_address,
            port=get_presto_port() if presto_port is None else presto_port,
            ext_ref_clk=ext_ref_clk,
            **self.DC_PARAMS,
        ) as pls:
            pls.hardware.set_adc_attenuation(self.sample_port, self.ADC_ATTENUATION)
            for port in (self.readout_port, self.control_port):
                pls.hardware.set_dac_current(port, self.DAC_CURRENT)
                pls.hardware.set_inv_sinc(port, 0)
            pls.hardware.configure_mixer(
                freq=self.readout_freq, in_ports=self.sample_port, out_ports=self.readout_port
            )
            pls.hardware.configure_mixer(freq=self.control_freq, out_ports=self.control_port)
            self._quantize(pls)
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            scales = self.control_amp_arr if hasattr(self, "control_amp_arr") else self.control_amp
            pls.setup_scale_lut(self.control_port, group=0, scales=scales)
            readout = pls.setup_flat_pulse(
                self.readout_port,
                group=0,
                duration=self.readout_duration,
                amplitude=1 + 1j,
                envelope=False,
            )
            pls.setup_store(self.sample_port, self.sample_duration)
            period, repeats = self._sequence(pls, readout)
            pls.run(period=period, repeat_count=repeats, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()
        if self.auto_fit:
            try:
                self.fit()
            except (ValueError, RuntimeError) as err:
                warnings.warn(
                    f"{type(self).__name__} fit failed; saving raw data: {err}", RuntimeWarning
                )
        return self.save(save_filename)

    def _response(self, sample_start: Optional[float], sample_end: Optional[float]) -> np.ndarray:
        if self.t_arr is None or self.store_arr is None:
            raise RuntimeError("No acquisition available; run or load a measurement first")
        times, traces = np.asarray(self.t_arr), np.asarray(self.store_arr)
        expected = int(np.prod(self._shape()))
        if times.ndim != 1 or traces.shape != (expected, 1, times.size):
            raise ValueError("store_arr must have shape (sweep points, 1, len(t_arr))")
        start = 0.0 if sample_start is None else _positive(sample_start, "sample_start", zero=True)
        end = self.sample_duration if sample_end is None else _positive(sample_end, "sample_end")
        if end <= start or end > self.sample_duration:
            raise ValueError("Integration window must lie within sample_duration, with start < end")
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            raise ValueError("Integration window contains no samples")
        self.resp_arr = traces[:, 0, mask].mean(axis=-1).reshape(self._shape())
        if not np.all(np.isfinite(self.resp_arr)):
            raise ValueError("Readout contains nonfinite samples")
        centered = self.resp_arr.ravel() - self.resp_arr.mean()
        angle = 0.5 * np.angle(np.sum(centered**2))
        return (self.resp_arr * np.exp(-1j * angle)).real

    def fit(
        self, sample_start: Optional[float] = None, sample_end: Optional[float] = None
    ) -> Dict[str, Any]:
        """Integrate and fit readout; store and return ``fit_results``.

        :param sample_start: Start of integration in seconds relative to store start (default 0).
        :param sample_end: Exclusive integration end (default the whole acquisition window).
        :returns: Fit parameters with one-sigma errors, curves and integration bounds.
        :raises ValueError: If the window, sweep or contrast cannot support a fit.
        """
        self.fit_results = None
        response = self._response(sample_start, sample_end)
        result = self._fit(response)
        result.update(
            sample_start=0.0 if sample_start is None else float(sample_start),
            sample_end=self.sample_duration if sample_end is None else float(sample_end),
        )
        self.fit_results = result
        return result

    def analyze(
        self,
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        fit: bool = True,
    ) -> Any:
        """Return a Matplotlib figure, optionally fitting the chosen integration window.

        :param sample_start: Integration start in seconds relative to store start.
        :param sample_end: Exclusive integration end in seconds.
        :param fit: Fit and overlay the result; unsuccessful fits leave a raw-data plot.
        :returns: Matplotlib figure (does not call ``show``).
        """
        response = self._response(sample_start, sample_end)
        result = None
        if fit:
            try:
                result = self.fit(sample_start, sample_end)
            except (ValueError, RuntimeError) as err:
                warnings.warn(f"Fit unavailable: {err}", RuntimeWarning)
        return self._plot(response, result)

    def save(self, save_filename: Optional[str] = None) -> str:
        """Save raw data, fit results and metadata; return the HDF5 filename."""
        # A dataset avoids HDF5's small-attribute limit for 2D fitted curves.
        self.fit_results_json = np.asarray(
            json.dumps(self.fit_results, allow_nan=False), dtype=h5py.string_dtype("utf-8")
        )
        try:
            return super()._save(inspect.getfile(type(self)), save_filename=save_filename)
        finally:
            del self.fit_results_json

    def _hdf5_skip(self) -> set:
        return {name for name, value in self.__dict__.items() if value is None}

    def _build_document(self, **kwargs: Any) -> Dict[str, Any]:
        document = super()._build_document(**kwargs)
        document.pop("fit_results_json", None)
        if self.fit_results is not None:
            document["fit_results"] = {
                key: value for key, value in self.fit_results.items() if key != "curve"
            }
        return document

    @classmethod
    def load(cls: type[QubitMeasurement], load_filename: str) -> QubitMeasurement:
        """Restore parameters, raw data, fit results and attached instrument metadata."""
        with h5py.File(load_filename, "r") as h5f:
            params = {}
            for name, parameter in inspect.signature(cls).parameters.items():
                if name in h5f.attrs:
                    params[name] = h5f.attrs[name]
                elif name in h5f:
                    params[name] = h5f[name][()]
                elif parameter.default is inspect.Parameter.empty:
                    raise ValueError(f"Missing required parameter {name} in {load_filename}")
            obj = cls(**params)
            for name, value in h5f.attrs.items():
                if not name.startswith("_") and name != "fit_results_json":
                    setattr(obj, name, value)
            for name in ("t_arr", "store_arr", "resp_arr"):
                if name in h5f:
                    setattr(obj, name, h5f[name][()])
            obj.fit_results = json.loads(
                h5f["fit_results_json"][()]
                if "fit_results_json" in h5f
                else h5f.attrs.get("fit_results_json", "null")
            )
        return obj

    def _sequence(self, pls: Any, readout: Any) -> Tuple[float, int]:
        raise NotImplementedError

    def _shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def _fit(self, response: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError

    def _plot(self, response: np.ndarray, result: Optional[Dict[str, Any]]) -> Any:
        raise NotImplementedError
