"""Offline StdDevSweep integration checks; run with python tests/test_sweep_std_dev.py.

Real QCTrace folding, summary saving/loading and plotting; simulated TimeStream and bias
generator. Database calls are mocked, so this suite never connects to hardware or MongoDB.
"""

import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from daq import StdDevSweep
from daq.measurements import QCTrace


class FakeBias:
    def __init__(self, trigger_port=2):
        self.trigger_port = trigger_port
        self.output = False
        self.closed = False
        self.ramps = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.output = False
        self.closed = True

    def sawtooth(self, **kwargs):
        self.ramps.append(kwargs)
        self.output = True

    def samples_for_periods(self, n_periods, sample_rate, *, freq_hz, discard_ms):
        return round(n_periods * sample_rate / freq_hz) + round(discard_ms * 1e-3 * sample_rate)


class FakeTimeStream:
    instances = []
    fail_at = None
    directory = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.df = 6000.0  # Deliberately different from the requested 5000 Hz.
        self.signal = None
        self.instances.append(self)

    def attach(self, **kwargs):
        self.attached = kwargs

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        if self.kwargs["lo_freq"] == self.fail_at:
            raise RuntimeError("simulated acquisition failure")
        n = self.kwargs["pixel_counts"] - round(self.kwargs["discard_start_ms"] * 1e-3 * self.df)
        k = np.arange(n)
        # A repeating response, with block-to-block offsets that cancel under folding.
        wave = (k % 12) / 12
        offsets = np.where((k // 12) % 2, -0.2, 0.2)
        scale_i, scale_q = {2.7e9: (1, 2), 2.8e9: (3, 1), 2.9e9: (2, 1)}[self.kwargs["lo_freq"]]
        self.signal = (0.01 * (scale_i + 1j * scale_q) * wave + offsets).reshape(-1, 1)
        path = str(Path(self.directory) / f"raw-{len(self.instances)}.h5")
        with h5py.File(path, "w") as h5f:
            h5f["signal"] = self.signal
        return path


class StdDevSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(plt.close, "all")
        FakeTimeStream.directory = self.tmp.name
        FakeTimeStream.instances = []
        FakeTimeStream.fail_at = None
        self.documents = []
        counter = itertools.count(1)
        for target, value in (
            ("daq.measurements._gate_bias.TimeStream", FakeTimeStream),
            ("daq._base.get_data_folder", lambda: self.tmp.name),
            ("daq._base.get_next_number", lambda: f"{next(counter):08d}"),
            ("daq._base.insert_measurement", self.documents.append),
        ):
            p = patch(target, value)
            p.start()
            self.addCleanup(p.stop)

    def make(self, **kwargs):
        params = dict(
            freq_arr=[2.7e9, 2.8e9, 2.9e9],
            amp=0.01,
            output_port=4,
            input_port=1,
            ramp_vpp=1.2,
            ramp_freq_hz=500,
            sampling_frequency=5000,
            num_periods=10,
            discard_start_ms=2,
            device="offline-QPD",
            filter="test-chain",
            notes="frequency optimization",
        )
        params.update(kwargs)
        return StdDevSweep(**params)

    def test_folded_iq_selection_and_acquisition(self):
        sweep = self.make()
        bias = FakeBias()
        sweep.run(bias, presto_address="test-presto", presto_port=123, ext_ref_clk=True)
        sigma = np.std(np.arange(12) / 12) * 0.01
        np.testing.assert_allclose(sweep.std_i_arr, sigma * np.array([1, 3, 2]))
        np.testing.assert_allclose(sweep.std_q_arr, sigma * np.array([2, 1, 1]))
        np.testing.assert_allclose(sweep.std_arr, sigma * np.sqrt([5, 10, 5]))
        self.assertEqual(sweep.best_freq, 2.8e9)
        self.assertEqual(sweep.best_qc_file, sweep.qc_files[1])
        np.testing.assert_array_equal(sweep.sample_counts, [12] * 3)
        np.testing.assert_array_equal(sweep.sampling_frequencies, [6000] * 3)
        self.assertFalse(bias.output)
        self.assertFalse(bias.closed)
        self.assertEqual(len(bias.ramps), 3)
        for freq, stream in zip(sweep.freq_arr, FakeTimeStream.instances):
            self.assertEqual(stream.kwargs["lo_freq"], freq)
            self.assertEqual(stream.kwargs["amp"], 0.01)
            self.assertEqual(stream.kwargs["output_port"], 4)
            self.assertEqual(stream.kwargs["pixel_counts"], 110)
            self.assertEqual(stream.kwargs["discard_start_ms"], 2)
            np.testing.assert_array_equal(stream.kwargs["external_trigger"], [0, 1])
            self.assertEqual(
                stream.run_kwargs,
                dict(
                    presto_address="test-presto",
                    presto_port=123,
                    ext_ref_clk=True,
                ),
            )
        self.assertTrue(all(Path(p).exists() for p in sweep.qc_files + sweep.raw_files))

    def test_roundtrip_plot_and_database(self):
        sweep = self.make(quantity="imag", ddof=1)
        path = sweep.run(FakeBias())
        self.assertEqual(sweep.best_freq, 2.7e9)
        doc = self.documents[-1]
        self.assertEqual(doc["type"], "sweep_std_dev")
        self.assertEqual(doc["best_freq"], 2.7e9)
        self.assertEqual(doc["trace_source"], "folded")
        self.assertEqual(doc["qc_files"], sweep.qc_files)
        restored = StdDevSweep.load(path)
        for name in (
            "std_arr",
            "std_i_arr",
            "std_q_arr",
            "std_principal_arr",
            "principal_axes",
            "freq_arr",
            "sample_counts",
            "sampling_frequencies",
            "trigger_states",
        ):
            np.testing.assert_array_equal(getattr(restored, name), getattr(sweep, name))
        for name in (
            "best_freq",
            "best_std",
            "best_qc_file",
            "qc_files",
            "raw_files",
            "device",
            "filter",
            "notes",
            "quantity",
            "trace_source",
            "ddof",
        ):
            self.assertEqual(getattr(restored, name), getattr(sweep, name))
        with patch("matplotlib.pyplot.show"):
            fig = restored.analyze()
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), restored.std_i_arr)
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), restored.std_q_arr)
        self.assertIn("GHz", ax.get_xlabel())
        self.assertIn("FS", ax.get_ylabel())
        np.testing.assert_array_equal(ax.lines[2].get_ydata(), restored.std_principal_arr)
        self.assertIn("Maximum", ax.lines[3].get_label())
        restored.run(FakeBias(trigger_port=1))
        np.testing.assert_array_equal(restored.trigger_states, [1])

    def test_raw_statistic_and_projections(self):
        raw = self.make(trace_source="raw", quantity="complex", ddof=1)
        raw.run(FakeBias())
        expected = [np.std(s.signal[:, 0], ddof=1) for s in FakeTimeStream.instances]
        np.testing.assert_allclose(raw.std_arr, expected)
        np.testing.assert_array_equal(raw.sample_counts, [98] * 3)
        z = np.array([1, 1j, -1, -1j], dtype=complex)
        trace = SimpleNamespace(avg_iq=np.vstack([z.real, z.imag]))
        for quantity, expected in (
            ("principal", np.sqrt(0.5)),
            ("complex", 1),
            ("abs", 0),
            ("real", np.sqrt(0.5)),
            ("imag", np.sqrt(0.5)),
        ):
            with self.subTest(quantity=quantity):
                self.assertAlmostEqual(
                    self.make(quantity=quantity).standard_deviation(trace), expected
                )
        rotated = z * np.exp(0.73j) + 2 + 3j
        trace.avg_iq = np.vstack([rotated.real, rotated.imag])
        self.assertAlmostEqual(self.make(quantity="complex").standard_deviation(trace), 1)

    def test_principal_axis_rotation_invariance_and_covariance(self):
        # Elliptical QC response: I and Q maxima move under independent rotations,
        # while PCA retains the same curve and operating frequency.
        phase = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        sweep = self.make(ddof=1)
        original, rotated, i_values = [], [], []
        for major, minor, angle in ((1, 0.3, 0.1), (3, 0.2, 1.4), (2, 0.5, -0.6)):
            z = major * np.cos(phase) + 1j * minor * np.sin(phase)
            trace = SimpleNamespace(avg_iq=np.vstack([z.real, z.imag]))
            original.append(sweep.standard_deviation(trace))
            z = z * np.exp(1j * angle) + 7 - 13j
            trace.avg_iq = np.vstack([z.real, z.imag])
            rotated.append(sweep.standard_deviation(trace))
            i_values.append(self.make(quantity="real").standard_deviation(trace))
            axis = sweep._principal_axis(z)
            self.assertAlmostEqual(np.linalg.norm(axis), 1)
            self.assertAlmostEqual(abs(axis @ [np.cos(angle), np.sin(angle)]), 1)
        np.testing.assert_allclose(original, rotated, rtol=1e-12)
        np.testing.assert_allclose(rotated, np.array([1, 3, 2]) * np.sqrt(60 / 119))
        self.assertEqual(np.argmax(rotated), 1)
        self.assertNotEqual(np.argmax(i_values), 1)
        # A flat trace has zero spread, including with a large DC offset.
        flat = SimpleNamespace(avg_iq=np.full((2, 12), 5.0))
        self.assertEqual(sweep.standard_deviation(flat), 0)

    def test_failure_clears_old_optimum_and_disables_bias(self):
        sweep = self.make()
        bias = FakeBias()
        sweep.run(bias)
        old_files = list(sweep.qc_files)
        FakeTimeStream.fail_at = 2.8e9
        with self.assertRaisesRegex(RuntimeError, "simulated acquisition"):
            sweep.run(bias)
        self.assertIsNone(sweep.best_freq)
        self.assertIsNone(sweep.best_std)
        self.assertIsNone(sweep.best_qc_file)
        self.assertFalse(bias.output)
        self.assertEqual(len(sweep.qc_files), 1)
        self.assertNotIn(sweep.qc_files[0], old_files)
        self.assertTrue(np.isnan(sweep.std_arr[1:]).all())
        with self.assertRaises(RuntimeError):
            sweep.analyze()
        with self.assertRaises(RuntimeError):
            sweep.save()

    def test_owned_session_cleanup_and_explicit_routing(self):
        bias = FakeBias()
        with patch("daq.measurements.sweep_std_dev.Agilent33220A", return_value=bias) as factory:
            self.make(trigger_states=[0, 1, 1]).run()
        factory.assert_called_once_with()
        self.assertTrue(bias.closed)
        self.assertFalse(bias.output)
        for stream in FakeTimeStream.instances:
            np.testing.assert_array_equal(stream.kwargs["external_trigger"], [0, 1, 1])
        failed_bias = FakeBias()
        FakeTimeStream.fail_at = 2.7e9
        with patch("daq.measurements.sweep_std_dev.Agilent33220A", return_value=failed_bias):
            with self.assertRaises(RuntimeError):
                self.make().run()
        self.assertTrue(failed_bias.closed)
        self.assertFalse(failed_bias.output)

    def test_ties_zero_curve_and_single_frequency(self):
        sweep = self.make(freq_arr=[2.9e9, 2.7e9], quantity="complex")
        sweep.run(FakeBias())
        # Set an exact tie to avoid roundoff deciding which equal synthetic trace wins.
        sweep.std_arr[:] = 1
        sweep._select_best()
        self.assertEqual(sweep.best_freq, 2.9e9)
        sweep.std_arr[:] = 0
        sweep._select_best()
        self.assertIsNone(sweep.best_freq)
        zero_path = sweep.save(str(Path(self.tmp.name) / "zero.h5"))
        self.assertIsNone(StdDevSweep.load(zero_path).best_freq)
        with patch("matplotlib.pyplot.show"):
            self.assertIn("no optimum", sweep.analyze().axes[0].texts[0].get_text())
        single = self.make(freq_arr=[2.8e9])
        single.run(FakeBias())
        self.assertEqual(single.best_freq, 2.8e9)

    def test_invalid_configuration_and_samples(self):
        for kwargs in (
            dict(freq_arr=[]),
            dict(freq_arr=2.8e9),
            dict(freq_arr=[[2.8e9]]),
            dict(freq_arr=[np.nan]),
            dict(freq_arr=[np.inf]),
            dict(freq_arr=[-1]),
            dict(amp=np.nan),
            dict(ramp_freq_hz=np.inf),
            dict(ramp_offset_v=np.nan),
            dict(sampling_frequency=0),
            dict(discard_start_ms=-1),
            dict(num_periods=1.5),
            dict(num_periods=True),
            dict(ddof=-1),
            dict(ddof=0.5),
            dict(ddof=10),
            dict(trigger_states=False),
            dict(trigger_states=[0, 0]),
            dict(quantity="invalid"),
            dict(trace_source="invalid"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)
        with self.assertRaises(ValueError):
            self.make(device=None).run(FakeBias())
        for iq in (
            np.zeros((2, 0)),
            np.zeros((2, 1)),
            np.zeros((3, 4)),
            np.array([[0, np.nan], [1, 2]]),
            np.array([[0, 1], [1, np.inf]]),
        ):
            with self.subTest(iq=iq), self.assertRaises(ValueError):
                self.make().standard_deviation(SimpleNamespace(avg_iq=iq))
        with self.assertRaises(RuntimeError):
            self.make().standard_deviation(SimpleNamespace(avg_iq=None))
        with self.assertRaises(RuntimeError):
            self.make(trace_source="raw").standard_deviation(SimpleNamespace(qc_stream=None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
