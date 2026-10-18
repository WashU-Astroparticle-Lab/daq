"""Offline sequence, fit and persistence checks. Run in the presto environment:

    python tests/test_pulsed_qubit.py

No hardware or MongoDB connection is made. The real Presto API is additionally
exercised in its built-in dry-run mode to validate sequence compilation.
"""

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="daq-mpl-"))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bson import BSON
from presto import pulsed

import daq._base as base
from daq import RabiAmp, RabiTime, T1

COMMON = dict(
    readout_freq=6.2e9,
    control_freq=4.2e9,
    readout_amp=0.1,
    readout_duration=2e-6,
    sample_duration=3e-6,
    readout_port=1,
    control_port=4,
    sample_port=1,
    wait_delay=10e-6,
    readout_sample_delay=0.5e-6,
    num_averages=10,
    device="test-qubit",
)


def make(cls, **kwargs):
    params = dict(COMMON)
    if cls is RabiAmp:
        params.update(control_duration=100e-9, control_amp_arr=np.linspace(0, 0.8, 101))
    elif cls is RabiTime:
        params.update(
            control_amp_arr=[0, 0.1, 0.2], control_duration_arr=np.linspace(20e-9, 4e-6, 101)
        )
    else:
        params.update(
            control_duration=100e-9, control_amp=0.4, delay_arr=np.linspace(0, 50e-6, 101)
        )
    params.update(kwargs)
    return cls(**params)


def synthetic(obj, noise=0):
    if isinstance(obj, RabiAmp):
        signal = 0.3 + 0.1 * np.cos(2 * np.pi * obj.num_pulses * obj.control_amp_arr / 0.4 + 0.3)
    elif isinstance(obj, RabiTime):
        signal = np.array(
            [
                (
                    0.3
                    + 0.1
                    * np.exp(-obj.control_duration_arr / 2e-6)
                    * np.cos(2 * np.pi * (amp * 1e7) * obj.control_duration_arr + 0.3)
                    if amp
                    else np.full(obj.control_duration_arr.size, 0.4)
                )
                for amp in obj.control_amp_arr
            ]
        )
    else:
        signal = 0.3 + 0.1 * np.exp(-obj.delay_arr / 17e-6)
    signal = signal + np.random.default_rng(42).normal(0, noise, signal.shape)
    obj.t_arr = np.arange(30) * obj.sample_duration / 30
    iq = signal.ravel() * np.exp(1.2j) + (0.1 - 0.07j)
    obj.store_arr = np.repeat(iq[:, None, None], obj.t_arr.size, axis=2)
    return obj


class FakePulse:
    def __init__(self, port, duration):
        self.port, self.duration = port, duration

    def set_total_duration(self, duration):
        self.duration = duration


class FakePulsed:
    instances = []
    fail = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.events, self.scales = [], {}
        self.hardware_calls = []
        self.closed = False
        self.hardware = SimpleNamespace(
            **{
                name: self.hardware_call
                for name in (
                    "set_adc_attenuation",
                    "set_dac_current",
                    "set_inv_sinc",
                    "configure_mixer",
                )
            }
        )
        self.instances.append(self)

    def hardware_call(self, *args, **kwargs):
        self.hardware_calls.append((args, kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def get_clk_T(self):
        return 2e-9

    def get_fs(self, which):
        return 1e9

    def setup_scale_lut(self, port, *, group, scales):
        self.scales[port] = np.atleast_1d(scales)

    def setup_template(self, port, *, group, template, envelope):
        self.template = template
        return FakePulse(port, len(template) / self.get_fs("dac"))

    def setup_flat_pulse(self, port, *, group, duration, amplitude, envelope):
        return FakePulse(port, duration)

    def setup_store(self, port, duration):
        self.store_port = port
        self.store_duration = duration

    def reset_phase(self, time, port):
        self.events.append(("reset", time, port))

    def output_pulse(self, time, pulse):
        self.events.append(("pulse", time, pulse.port, pulse.duration))

    def store(self, time):
        self.events.append(("store", time))

    def next_scale(self, time, port, *, group):
        self.events.append(("scale", time, port))

    def run(self, **kwargs):
        self.run_args = kwargs
        if self.fail:
            raise RuntimeError("injected acquisition error")

    def get_store_data(self):
        n = sum(e[0] == "store" for e in self.events) * self.run_args["repeat_count"]
        return np.arange(10) * self.store_duration / 10, np.ones((n, 1, 10), dtype=complex)


class PulsedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.documents = []
        for name, value in (
            ("get_next_number", lambda: "00000001"),
            ("get_data_folder", lambda: self.temp.name),
            ("insert_measurement", lambda doc: self.documents.append(doc) or "test"),
        ):
            p = patch.object(base, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(plt.close, "all")

    def test_sequences_and_roundtrip(self):
        for cls in (RabiAmp, RabiTime, T1):
            with self.subTest(cls=cls.__name__), patch.object(pulsed, "Pulsed", FakePulsed):
                obj = make(cls, auto_fit=False)
                obj.attach(bias={"offset": 0.2})
                path = obj.run(presto_address="offline", presto_port=1234, ext_ref_clk=True)
                fake = FakePulsed.instances[-1]
                self.assertTrue(fake.closed)
                self.assertEqual(fake.kwargs["address"], "offline")
                self.assertEqual(fake.kwargs["port"], 1234)
                self.assertTrue(fake.kwargs["ext_ref_clk"])
                controls = [e for e in fake.events if e[0] == "pulse" and e[2] == 4]
                readouts = [e for e in fake.events if e[0] == "pulse" and e[2] == 1]
                stores = [e for e in fake.events if e[0] == "store"]
                for readout, store in zip(readouts, stores):
                    self.assertAlmostEqual(store[1] - readout[1], obj.readout_sample_delay)
                for i in range(len(readouts) - 1):
                    self.assertGreaterEqual(
                        controls[i + 1][1] + 1e-15,
                        stores[i][1] + obj.sample_duration + obj.wait_delay,
                    )
                self.assertGreaterEqual(
                    fake.run_args["period"], stores[-1][1] + obj.sample_duration
                )
                if cls is RabiAmp:
                    self.assertEqual(len(controls), 1)
                    self.assertEqual(fake.run_args["repeat_count"], 101)
                    self.assertEqual(sum(e[0] == "scale" for e in fake.events), 1)
                    np.testing.assert_allclose(fake.scales[4], obj.control_amp_arr)
                    self.assertAlmostEqual(abs(fake.template[0]), 0)
                    self.assertGreater(abs(fake.template[len(fake.template) // 2]), 1)
                elif cls is RabiTime:
                    np.testing.assert_allclose([e[3] for e in controls], obj.control_duration_arr)
                    self.assertEqual(len(stores), 101)
                    self.assertEqual(fake.run_args["repeat_count"], 3)
                    self.assertGreater(
                        [e[1] for e in fake.events if e[0] == "scale"][0], stores[-1][1]
                    )
                else:
                    np.testing.assert_allclose(
                        [r[1] - c[1] - c[3] for r, c in zip(readouts, controls)],
                        obj.delay_arr,
                        atol=1e-15,
                    )
                    self.assertEqual(fake.run_args["repeat_count"], 1)
                    self.assertFalse(any(e[0] == "scale" for e in fake.events))
                loaded = cls.load(path)
                np.testing.assert_array_equal(loaded.store_arr, obj.store_arr)
                self.assertEqual(loaded.bias_offset, 0.2)
                self.assertEqual(loaded.device, "test-qubit")
                self.assertIsNone(loaded.fit_results)
                doc = self.documents[-1]
                self.assertNotIn("store_arr", doc)
                self.assertNotIn("t_arr", doc)
                BSON.encode(doc)
                self.assertEqual(
                    doc["type"], {RabiAmp: "rabi_amp", RabiTime: "rabi_time", T1: "t1"}[cls]
                )

    def test_calibration_fits_and_plots(self):
        for cls in (RabiAmp, RabiTime, T1):
            obj = synthetic(make(cls), noise=1e-4)
            fit = obj.fit()
            if cls is RabiAmp:
                self.assertAlmostEqual(fit["pi_amp"], 0.2, delta=0.001)
            elif cls is RabiTime:
                # Set the reference exactly flat to test partial fit failure.
                obj.store_arr[:101] = 0.4 + 0.2j
                fit = obj.fit()
                self.assertIsNone(fit["rabi_rate"][0])
                np.testing.assert_allclose(fit["rabi_rate"][1:], [1e6, 2e6], rtol=0.002)
                self.assertAlmostEqual(fit["rate_slope"], 1e7, delta=1e4)
                self.assertIsNotNone(fit["fit_errors"][0])
            else:
                self.assertAlmostEqual(fit["t1"], 17e-6, delta=0.1e-6)
                self.assertGreater(fit["t1_err"], 0)
            fig = obj.analyze()
            fig.canvas.draw()
            path = obj.save(os.path.join(self.temp.name, cls.__name__ + ".h5"))
            loaded = cls.load(path)
            self.assertEqual(loaded.fit_results, obj.fit_results)
            np.testing.assert_array_equal(loaded.resp_arr, obj.resp_arr)
            doc = self.documents[-1]
            self.assertIn("fit_results", doc)
            self.assertNotIn("curve", doc["fit_results"])
            BSON.encode(doc)
            with h5py.File(path) as h5f:
                self.assertIn("source_code", h5f)

    def test_num_pulses_calibration_and_sequence(self):
        obj = synthetic(make(RabiAmp, num_pulses=3))
        self.assertAlmostEqual(obj.fit()["pi_amp"], 0.2, places=6)
        with patch.object(pulsed, "Pulsed", FakePulsed):
            obj.auto_fit = False
            obj.run()
        controls = [e for e in FakePulsed.instances[-1].events if e[0] == "pulse" and e[2] == 4]
        np.testing.assert_allclose([e[1] for e in controls], np.arange(3) * obj.control_duration)

    def test_window_and_flat_readout(self):
        obj = synthetic(make(T1))
        obj.store_arr[:, :, :5] = 100  # startup transient excluded by the chosen window
        self.assertAlmostEqual(
            obj.fit(sample_start=0.5e-6, sample_end=2e-6)["t1"], 17e-6, places=10
        )
        for start, end in ((-1, 2e-6), (1e-6, 1e-6), (0, 4e-6), (1e-12, 2e-12)):
            with self.assertRaises(ValueError):
                obj.fit(start, end)
        obj.store_arr[:] = 1
        with self.assertRaises(ValueError):
            obj.fit()
        self.assertIsNone(obj.fit_results)
        with warnings.catch_warnings(record=True) as caught:
            obj.analyze().canvas.draw()
        self.assertTrue(caught)
        with self.assertRaises(RuntimeError):
            make(T1).fit()

    def test_validation(self):
        for params in (
            dict(readout_port=4),
            dict(control_port=9),
            dict(num_averages=1.5),
            dict(readout_amp=1.1),
            dict(control_amp_arr=[]),
            dict(control_amp_arr=[0.2, 0.1]),
            dict(control_amp_arr=[0.1, np.nan]),
            dict(control_amp_arr=np.linspace(0, 1, 513)),
            dict(wait_delay=-1),
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                make(RabiAmp, **params)
        with patch.object(pulsed, "Pulsed", FakePulsed):
            before = len(FakePulsed.instances)
            with self.assertRaises(ValueError):
                make(T1, device=None).run()
            self.assertEqual(len(FakePulsed.instances), before)
            with self.assertRaises(ValueError):
                make(T1, delay_arr=[0, 0.1e-9]).run()
            with self.assertRaises(ValueError):
                make(RabiAmp, control_duration=0.1e-9).run()

    def test_failed_rerun_clears_data_and_fit(self):
        obj = synthetic(make(T1))
        obj.fit()
        with patch.object(pulsed, "Pulsed", FakePulsed), patch.object(FakePulsed, "fail", True):
            with self.assertRaises(RuntimeError):
                obj.run()
        self.assertTrue(FakePulsed.instances[-1].closed)
        self.assertIsNone(obj.fit_results)
        self.assertIsNone(obj.store_arr)
        self.assertEqual(self.documents, [])

    def test_auto_fit_failure_preserves_data(self):
        with patch.object(pulsed, "Pulsed", FakePulsed), warnings.catch_warnings(
            record=True
        ) as caught:
            obj = make(T1)
            path = obj.run()
        self.assertTrue(caught)
        self.assertIsNone(T1.load(path).fit_results)
        self.assertIsNotNone(T1.load(path).store_arr)

    def test_auto_fit_success_and_large_fit_storage(self):
        obj = synthetic(make(T1))
        expected_time, expected_data = obj.t_arr.copy(), obj.store_arr.copy()
        with patch.object(pulsed, "Pulsed", FakePulsed), patch.object(
            FakePulsed, "get_store_data", return_value=(expected_time, expected_data)
        ):
            path = obj.run()
        self.assertAlmostEqual(T1.load(path).fit_results["t1"], 17e-6, places=10)
        # Curves for a large 2D sweep can exceed HDF5's attribute size limit.
        obj.fit_results["curve"] = np.linspace(0, 1, 10000).tolist()
        path = obj.save(os.path.join(self.temp.name, "large-fit.h5"))
        with h5py.File(path) as h5f:
            self.assertIn("fit_results_json", h5f)
        self.assertEqual(T1.load(path).fit_results, obj.fit_results)

    def test_timing_quantization(self):
        obj = make(T1, auto_fit=False, control_duration=101.2e-9, delay_arr=[0, 7.2e-9, 15.2e-9])
        with patch.object(pulsed, "Pulsed", FakePulsed):
            path = obj.run()
        loaded = T1.load(path)
        np.testing.assert_allclose(loaded.delay_arr, [0, 8e-9, 16e-9])
        self.assertAlmostEqual(loaded.control_duration / 2e-9, 51)
        for event in FakePulsed.instances[-1].events:
            self.assertAlmostEqual(event[1] / 2e-9, round(event[1] / 2e-9), places=7)

    def test_single_amplitude_time_and_nonuniform_sweep(self):
        obj = synthetic(make(RabiTime, control_amp_arr=0.1))
        self.assertAlmostEqual(obj.fit()["rabi_rate"][0], 1e6, delta=1)
        obj.analyze().canvas.draw()
        obj = synthetic(make(RabiAmp, control_amp_arr=np.linspace(0, 1, 101) ** 1.2))
        self.assertAlmostEqual(obj.fit()["pi_amp"], 0.2, places=6)

    def test_real_api_dry_run(self):
        # Use the actual Presto sequencer with no network/hardware. Smaller sweeps
        # keep dry-run sample allocations small. Only the connection is replaced.
        real_pulsed = pulsed.Pulsed

        def dry_run(**kwargs):
            return real_pulsed(dry_run=True, **kwargs)

        with patch.object(pulsed, "Pulsed", dry_run):
            for cls in (RabiAmp, RabiTime, T1):
                params = dict(auto_fit=False)
                if cls is RabiAmp:
                    params["control_amp_arr"] = [0.1, 0.2]
                elif cls is RabiTime:
                    params.update(control_amp_arr=[0.1, 0.2], control_duration_arr=[100e-9, 200e-9])
                else:
                    params["delay_arr"] = [0, 1e-6]
                obj = make(cls, **params)
                with self.assertRaisesRegex(
                    RuntimeError, "No store data available. This is a dry run"
                ):
                    obj.run()


if __name__ == "__main__":
    unittest.main(verbosity=2)
