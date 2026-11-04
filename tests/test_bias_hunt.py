"""Offline hardware-summary BiasHunt checks. Run in the presto environment.

The mock reduces explicit I/Q samples, never offers a raw acquisition path, and chooses a
known optimum whose magnitude is constant: ranking std(abs(signal)) would fail this test.
Shared timing, pooling, SDK and cleanup cases also live in test_sweep_freq_and_dc.py.
"""

import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from test_sweep_freq_and_dc import FakeBias, FakeLockin, plt

import daq._base as base_mod
import daq.measurements._gate_bias as gate_mod
import daq.measurements.sweep_freq_and_dc as sweep_mod
from daq import BiasHunt


def make(**kwargs):
    args = dict(
        readout_freq=6e9,
        amp=0.1,
        output_port=1,
        input_port=1,
        bias_voltages=[0.003, 0.001, 0.002],
        voltage_divider=455,
        device="test",
        notes="offline",
        filter="none",
    )
    args.update(kwargs)
    return BiasHunt(**args)


class HuntLockin(FakeLockin):
    def __init__(self, hunt, fail_at=None, factors=(0.1, 0.4, 0.2)):
        super().__init__(hunt)
        self.fail_at = fail_at
        self.factors = factors

    def get_pixels(self, n, *, summed, nsum, fir_coeffs):
        index = len(self.calls)
        if index == self.fail_at:
            raise RuntimeError("failed mid-hunt")
        self.calls.append((n, nsum))
        assert summed is True and fir_coeffs is None
        extra = self.sweep.discard_blocks - 1
        # Each point switches between +a and -a; abs(signal) has no contrast at all.
        samples = np.tile(np.where(np.arange(nsum) % 2, 1.0, -1.0), (n - extra, 1))
        samples *= self.factors[index % len(self.factors)]
        means = np.r_[np.full(extra, 1000.0), samples.mean(1)][:, None].astype(complex)
        stds = np.r_[np.full(extra, 500.0), samples.std(1, ddof=self.sweep.summary_ddof)][:, None]
        zeros = np.zeros_like(means)
        self.oracles.append((samples.mean(), samples.std()))
        return {1: (np.array([0.0]), means, stds.astype(complex), zeros, zeros)}


class HuntTests(unittest.TestCase):
    def run_mock(self, hunt, **kwargs):
        bias, lck = FakeBias(), HuntLockin(hunt, **kwargs)
        with patch.object(sweep_mod.lockin, "Lockin", return_value=lck) as opened, patch.object(
            gate_mod, "TimeStream", side_effect=AssertionError("raw acquisition")
        ), patch.object(BiasHunt, "save", return_value="hunt.h5") as saved:
            hunt.run(bias, presto_address="offline")
        opened.assert_called_once()
        saved.assert_called_once()
        return bias, lck

    def test_rank_sigma_iq_and_convert_winner(self):
        h = make()
        b, lck = self.run_mock(h)
        np.testing.assert_allclose(h.parity_contrast, [0.1, 0.4, 0.2])
        np.testing.assert_allclose(h.std_iq_arr, h.parity_contrast)
        np.testing.assert_allclose(h.std_q_arr, 0)
        self.assertEqual(h.resp_arr.shape, (3,))
        self.assertEqual(h.best_bias, 0.001)
        self.assertAlmostEqual(h.best_generator_bias, 0.455)
        self.assertAlmostEqual(h.best_contrast, 0.4)
        np.testing.assert_allclose([v for v, _ in b.calls], [1.365, 0.455, 0.91])
        self.assertEqual(lck.mixers, [6e9])
        self.assertEqual(lck.calls, [(20, 250)] * 3)
        self.assertTrue(all(t == [0] for t in lck.triggers))
        self.assertFalse(b.output)
        self.assertEqual(lck.og.amps[-1], 0)
        self.assertFalse(hasattr(h, "bias_files"))

    def test_random_draw_is_fixed_and_reproducible(self):
        a = make(bias_voltages=None, v_min=-0.001, v_max=0.002, seed=19)
        b = make(bias_voltages=None, v_min=-0.001, v_max=0.002, seed=19)
        np.testing.assert_array_equal(a.bias_voltages, b.bias_voltages)
        original = a.bias_voltages.copy()
        self.run_mock(a)
        self.run_mock(a)
        np.testing.assert_array_equal(a.bias_voltages, original)
        self.assertEqual(a.n_bias_try, 20)

    def test_explicit_candidates_override_bounds_and_count(self):
        h = make(v_min=100, v_max=-100, n_bias_try=0)
        self.assertEqual(h.v_min, 0.001)
        self.assertEqual(h.v_max, 0.003)
        self.assertEqual(h.n_bias_try, 3)

    def test_validation_and_required_divider(self):
        for kwargs in (
            dict(bias_voltages=None),
            dict(bias_voltages=[]),
            dict(bias_voltages=[[0, 1]]),
            dict(bias_voltages=[np.nan]),
            dict(voltage_divider=0),
            dict(bias_voltages=None, v_min=0, v_max=np.inf),
            dict(bias_voltages=None, v_min=0, v_max=1, n_bias_try=1.5),
            dict(bias_voltages=None, v_min=0, v_max=1, n_bias_try=np.nan),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make(**kwargs)
        with self.assertRaises(TypeError):
            BiasHunt(readout_freq=6e9, amp=0.1, output_port=1, input_port=1, bias_voltages=[0])

    def test_short_warning_and_high_rate_discard(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            h = make(sampling_frequency=1e6)
            make(acquisition_duration_s=0.1)
        self.assertEqual(len(seen), 1)
        _, lck = self.run_mock(h)
        self.assertEqual(h.discard_blocks, 25)
        self.assertGreaterEqual(h.actual_discard_s, 0.025)
        self.assertEqual(lck.calls[0][0], h.retained_blocks + 24)
        np.testing.assert_allclose(h.std_iq_arr, [o[1] for o in lck.oracles])

    def test_failure_clears_previous_winner_and_never_saves(self):
        h = make()
        self.run_mock(h)
        b, lck = FakeBias(), HuntLockin(h, fail_at=1)
        with patch.object(sweep_mod.lockin, "Lockin", return_value=lck), patch.object(
            BiasHunt, "save"
        ) as saved:
            with self.assertRaisesRegex(RuntimeError, "mid-hunt"):
                h.run(b, presto_address="offline")
        saved.assert_not_called()
        self.assertIsNone(h.best_bias)
        self.assertIsNone(h.best_generator_bias)
        self.assertIsNone(h.parity_contrast)
        self.assertFalse(b.output)
        self.assertEqual(lck.og.amps[-1], 0)

    def test_ties_and_single_candidate(self):
        h = make()
        self.run_mock(h, factors=(0.2, 0.2, 0.2))
        self.assertEqual(h.best_bias, 0.003)
        h = make(bias_voltages=[0.001])
        self.run_mock(h)
        self.assertEqual(h.std_iq_arr.shape, (1,))
        self.assertEqual(h.best_bias, 0.001)

    def test_save_once_after_outputs_off_and_roundtrip(self):
        h = make()
        bias, lck = FakeBias(), HuntLockin(h)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hunt.h5")

            def insert(document):
                self.assertFalse(bias.output)
                self.assertEqual(lck.og.amps[-1], 0)
                self.assertEqual(document["type"], "bias_hunt")
                self.assertEqual(document["contrast_metric"], "sigma_iq")
                return "offline"

            with patch.object(sweep_mod.lockin, "Lockin", return_value=lck), patch.object(
                base_mod, "get_next_number", return_value="1"
            ), patch.object(base_mod, "insert_measurement", side_effect=insert) as saved:
                h.run(bias, presto_address="offline", save_filename=path)
            saved.assert_called_once()
            self.assertEqual(os.listdir(tmp), ["hunt.h5"])
            with h5py.File(path) as f:
                self.assertEqual(f["std_iq_arr"].shape, (3,))
                self.assertFalse(any(k in f for k in ("signal", "pixel_i", "bias_files")))
            loaded = BiasHunt.load(path)
            np.testing.assert_array_equal(loaded.bias_voltages, h.bias_voltages)
            np.testing.assert_array_equal(loaded.parity_contrast, h.parity_contrast)
            self.assertEqual(loaded.best_generator_bias, h.best_generator_bias)
            self.assertEqual(loaded.actual_acquisition_s, 0.5)
            with patch.object(plt, "show"):
                fig = loaded.analyze()
                self.assertIn("chip V", fig.axes[0].get_xlabel())
                self.assertEqual(fig.axes[0].get_ylabel(), "σIQ [FS]")
                np.testing.assert_array_equal(
                    fig.axes[0].lines[0].get_xdata(), [0.001, 0.002, 0.003]
                )
                plt.close(fig)
            self.run_mock(loaded)

    def test_legacy_record_retains_magnitude_and_generator_volts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.h5")
            with h5py.File(path, "w") as f:
                f.attrs.update(
                    readout_freq=6e9,
                    best_bias=1.0,
                    best_contrast=0.5,
                    best_bias_file="old_stream.h5",
                )
                f["bias_voltages"] = [0.5, 1.0]
                f["parity_contrast"] = [0.2, 0.5]
                f.create_dataset(
                    "bias_files", data=["a.h5", "old_stream.h5"], dtype=h5py.string_dtype()
                )
            h = BiasHunt.load(path)
            self.assertEqual(h.bias_files, ["a.h5", "old_stream.h5"])
            self.assertEqual(h.best_bias_file, "old_stream.h5")
            self.assertEqual(h.contrast_metric, "abs")
            self.assertIsNone(h.voltage_divider)
            self.assertIsNone(h.std_iq_arr)
            with patch.object(plt, "show"):
                fig = h.analyze()
                self.assertIn("Agilent V", fig.axes[0].get_xlabel())
                self.assertEqual(fig.axes[0].get_ylabel(), "std(|signal|) [FS]")
                plt.close(fig)
            with self.assertRaisesRegex(ValueError, "analysis-only"):
                h.run(FakeBias())


if __name__ == "__main__":
    unittest.main(verbosity=2)
