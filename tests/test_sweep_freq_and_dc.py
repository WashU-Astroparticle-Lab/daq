"""Offline checks for the Agilent/hardware-summary sweep. No hardware or database.

Run with ``conda activate presto; python tests/test_sweep_freq_and_dc.py``.
Block moments are generated from explicit samples, providing an independent pooling oracle.
Actual firmware normalization/timing still needs a bench comparison; mocks cannot verify it.
"""

import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import daq._base as base_mod
import daq.measurements.sweep_freq_and_dc as mod
from daq import SweepFreqAndDC


def make(**kwargs):
    params = dict(
        freq_center=6e9,
        freq_span=14e3,
        freq_step=7e3,
        amp=0.1,
        bias_arr=[0.001, -0.002],
        voltage_divider=455,
        output_port=1,
        input_port=1,
        device="offline",
    )
    params.update(kwargs)
    return SweepFreqAndDC(**params)


class FakeBias:
    def __init__(self):
        self.output = False
        self.calls = []
        self.offset = 0.0
        self.readback_error = 0.0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.output = False
        self.closed = True

    def constant(self, voltage, *, settle_s):
        self.calls.append((voltage, settle_s))
        self.offset = voltage
        self.output = True

    def settings(self):
        return dict(
            function="DC",
            offset_v=self.offset + self.readback_error,
            output=self.output,
            load="INF",
        )


class FakeGroup:
    def __init__(self):
        self.amps = []

    def set_amplitudes(self, value):
        self.amps.append(value)

    def set_frequencies(self, *args):
        pass

    def set_phases(self, *args):
        pass


class FakeLockin:
    def __init__(self, sweep, fail=False, tuned_rate=None):
        self.sweep = sweep
        self.fail = fail
        self.tuned_rate = tuned_rate or sweep.sampling_frequency
        self.hardware = self
        self.og = FakeGroup()
        self.calls = []
        self.mixers = []
        self.triggers = []
        self.oracles = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def set_adc_attenuation(self, *args):
        pass

    def set_dac_current(self, *args):
        pass

    def set_inv_sinc(self, *args):
        pass

    def configure_mixer(self, **kwargs):
        self.mixers.append(kwargs["freq"])

    def tune(self, *args):
        return 0.0, self.tuned_rate

    def set_df(self, value):
        self.df = value

    def set_trigger_out(self, value):
        self.triggers.append(value)

    def add_output_group(self, *args):
        return self.og

    def add_input_group(self, *args):
        return FakeGroup()

    def set_dither(self, *args):
        pass

    def apply_settings(self):
        pass

    def get_pixels(self, n, *, summed, nsum, fir_coeffs):
        assert summed is True and fir_coeffs is None  # No raw acquisition is allowed.
        assert 2 <= nsum <= 1023
        self.calls.append((n, nsum))
        if self.fail:
            raise RuntimeError("acquisition failed")
        extra = self.sweep.discard_blocks - 1  # SDK has already hidden its initial block.
        # A slow ramp in block means plus alternating intra-block fluctuations: ignoring
        # the between-block term or averaging stds produces a very different answer.
        k = np.arange(n - extra)[:, None]
        x = np.arange(nsum)[None, :] % 2
        i = 0.1 + 0.002 * k + 0.004 * x
        q = -0.03 - 0.001 * k + 0.003 * x
        z = i.ravel() + 1j * q.ravel()
        self.oracles.append((z.mean(), np.std(z.real), np.std(z.imag), np.std(z)))

        def summaries(a):
            # Large startup contamination must be excluded, including any additional blocks.
            means = np.r_[np.full(extra, 1000.0), a.mean(axis=1)]
            stds = np.r_[np.full(extra, 500.0), a.std(axis=1, ddof=self.sweep.summary_ddof)]
            # Imaginary demodulator components must not enter the zero-IF response.
            return (means + 9j)[:, None], (stds + 7j)[:, None]

        mi, si = summaries(i)
        mq, sq = summaries(q)
        return {1: (np.array([0.0]), mi, si, mq, sq)}


class SweepTests(unittest.TestCase):
    def test_installed_sdk_discards_exactly_one_summary_block(self):
        # Exercise the installed SDK's command builder with NO hardware connection. This
        # independently guards the hidden-block assumption used by our timing planner.
        sdk = object.__new__(mod.lockin.Lockin)
        sdk.dry_run = False
        sdk.adc_direct = False
        sdk.apply_required = lambda: False
        sdk._input_used = np.array([True, True])
        command, skipped, _ = sdk._build_get_command(
            n=20,
            summed=True,
            fir_coeffs=None,
            nsum=250,
            quiet=False,
            auto_sync=True,
            rpu_params=None,
        )
        self.assertEqual(skipped, 1)
        self.assertEqual(command.value[1], 21)

    def run_mock(self, sweep, bias=None, **kwargs):
        bias = bias or FakeBias()
        lck = FakeLockin(sweep, **kwargs)
        with patch.object(mod.lockin, "Lockin", return_value=lck), patch.object(
            SweepFreqAndDC, "save", return_value="summary.h5"
        ):
            path = sweep.run(bias, presto_address="offline", presto_port=7878)
        return bias, lck, path

    def test_defaults_and_explicit_divider(self):
        s = make()
        self.assertEqual(s.samples_per_block, 250)
        self.assertEqual(s.discard_blocks, 1)
        self.assertEqual(s.retained_blocks, 20)
        self.assertEqual(s.retained_samples, 5000)
        self.assertAlmostEqual(s.actual_acquisition_s, 0.5)
        self.assertAlmostEqual(s.actual_discard_s, 0.025)
        np.testing.assert_allclose(s.generator_bias_arr, [0.455, -0.91])
        with self.assertRaises(TypeError):
            SweepFreqAndDC(
                6e9, 1e6, amp=0.1, bias_arr=[0], input_port=1, output_port=1, freq_step=1e3
            )

    def test_block_rounding_and_high_rate(self):
        for fs in (1, 1000, 10000, 9999.9, 50000, 1000000):
            s = make(sampling_frequency=fs)
            self.assertGreaterEqual(s.actual_discard_s, 0.025)
            self.assertGreaterEqual(s.actual_acquisition_s, 0.5)
            self.assertLessEqual(s.samples_per_block, 1023)
            self.assertLess(s.actual_acquisition_s - 0.5, s.samples_per_block / fs + 1e-12)
        s = make(sampling_frequency=1e6)
        self.assertEqual(s.discard_blocks, 25)
        s = make(discard_start_ms=0)
        self.assertEqual(s.discard_blocks, 1)  # mandatory SDK startup block

    def test_warning_depends_on_retained_duration(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            make(sampling_frequency=1e6)
            make(acquisition_duration_s=0.2)
            make(acquisition_duration_s=0.1)
        self.assertEqual(len(seen), 1)
        self.assertIn("Retained acquisition", str(seen[0].message))

    def test_validation_and_df_alias(self):
        for bad in (
            dict(voltage_divider=0),
            dict(voltage_divider=np.nan),
            dict(bias_arr=[]),
            dict(bias_arr=[[1, 2]]),
            dict(bias_arr=[np.inf]),
            dict(freq_step=0),
            dict(acquisition_duration_s=0),
            dict(discard_start_ms=-1),
            dict(sampling_frequency=1e7),
            dict(summary_ddof=2),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                make(**bad)
        with self.assertWarns(DeprecationWarning):
            s = make(freq_step=None, df=1234)
        self.assertEqual(s.freq_step, 1234)
        self.assertEqual(s.sampling_frequency, 10000)

    def test_summary_acquisition_and_independent_frequency_grid(self):
        s = make()
        bias, lck, path = self.run_mock(s, tuned_rate=9999.9)
        self.assertEqual(path, "summary.h5")
        np.testing.assert_allclose(s.freq_arr, [6e9 - 7000, 6e9, 6e9 + 7000])
        self.assertEqual(lck.df, 9999.9)
        self.assertEqual(len(lck.calls), 6)
        self.assertTrue(all(c == (s.retained_blocks, s.samples_per_block) for c in lck.calls))
        for arr, index in ((s.resp_arr, 0), (s.std_i_arr, 1), (s.std_q_arr, 2), (s.std_iq_arr, 3)):
            np.testing.assert_allclose(arr.ravel(), [o[index] for o in lck.oracles], atol=1e-14)
        self.assertEqual(bias.calls, [(0.455, 0.5), (-0.91, 0.5)])
        self.assertFalse(bias.output)
        self.assertFalse(bias.closed)  # caller keeps the VISA session
        self.assertEqual(lck.og.amps[-1], 0)
        self.assertTrue(all(t == [0] for t in lck.triggers))
        self.assertTrue(lck.closed)

    def test_multiple_discard_blocks_and_sample_variance(self):
        s = make(sampling_frequency=1e6, freq_span=0, bias_arr=[0], summary_ddof=1)
        _, lck, _ = self.run_mock(s)
        self.assertEqual(lck.calls[0][0], s.retained_blocks + 24)
        np.testing.assert_allclose(s.std_iq_arr[0, 0], lck.oracles[0][3])
        self.assertLess(abs(s.resp_arr[0, 0]), 10)  # contaminated blocks excluded

    def test_pool_matches_explicit_samples_and_includes_slow_switching(self):
        for ddof in (0, 1):
            a = np.random.default_rng(34).normal(size=(7, 250)) + np.arange(7)[:, None] * 10
            mean, std = SweepFreqAndDC._pool(a.mean(1), a.std(1, ddof=ddof), 250, ddof)
            self.assertAlmostEqual(mean, a.mean())
            self.assertAlmostEqual(std, a.std())
        mean, std = SweepFreqAndDC._pool([-1, 1], [0, 0], 250)
        self.assertEqual(mean, 0)
        self.assertEqual(std, 1)  # per-block stds alone contain no signal

    def test_outputs_off_on_failure_and_old_results_cleared(self):
        s = make()
        self.run_mock(s)
        bias, lck = FakeBias(), FakeLockin(s, fail=True)
        with patch.object(mod.lockin, "Lockin", return_value=lck), patch.object(
            SweepFreqAndDC, "save"
        ) as save:
            with self.assertRaisesRegex(RuntimeError, "acquisition failed"):
                s.run(bias, presto_address="offline")
        save.assert_not_called()
        self.assertFalse(bias.output)
        self.assertEqual(lck.og.amps[-1], 0)
        self.assertTrue(np.all(np.isnan(s.std_iq_arr)))

    def test_connection_failure_also_turns_off_bias(self):
        bias = FakeBias()
        bias.output = True
        with patch.object(mod.lockin, "Lockin", side_effect=RuntimeError("connect")):
            with self.assertRaises(RuntimeError):
                make().run(bias, presto_address="offline")
        self.assertFalse(bias.output)

    def test_generator_clipping_is_detected(self):
        bias = FakeBias()
        bias.readback_error = 0.5
        s = make()
        lck = FakeLockin(s)
        with patch.object(mod.lockin, "Lockin", return_value=lck):
            with self.assertRaisesRegex(ValueError, "readback"):
                s.run(bias, presto_address="offline")
        self.assertFalse(bias.output)
        self.assertEqual(lck.calls, [])
        self.assertEqual(lck.og.amps[-1], 0)

    def test_owned_generator_closed(self):
        s, bias = make(), FakeBias()
        with patch.object(mod, "Agilent33220A", return_value=bias), patch.object(
            mod.lockin, "Lockin", return_value=FakeLockin(s)
        ), patch.object(SweepFreqAndDC, "save", return_value="saved"):
            s.run(presto_address="offline")
        self.assertTrue(bias.closed)
        self.assertFalse(bias.output)

    def test_hdf5_roundtrip_and_plots(self):
        s = make(bias_arr=[-0.002, 0.001, 0.004], freq_span=0)
        self.run_mock(s)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sweep.h5")
            with patch.object(base_mod, "get_next_number", return_value="1"), patch.object(
                base_mod, "insert_measurement", return_value="offline"
            ) as insert:
                s.save(path)
            document = insert.call_args.args[0]
            self.assertNotIn("std_iq_arr", document)
            self.assertEqual(document["voltage_divider"], 455)
            loaded = SweepFreqAndDC.load(path)
            for key in (
                "resp_arr",
                "std_i_arr",
                "std_q_arr",
                "std_iq_arr",
                "bias_arr",
                "generator_bias_arr",
                "generator_bias_readback_arr",
            ):
                np.testing.assert_array_equal(getattr(loaded, key), getattr(s, key))
            self.assertEqual(loaded.voltage_divider, 455)
            self.assertEqual(loaded.actual_acquisition_s, 0.5)
            self.assertEqual(loaded.bias_function, "DC")
            with h5py.File(path) as f:
                self.assertFalse(any(k in f for k in ("signal", "pixel_i", "pixel_q", "summaries")))
                self.assertEqual(f["std_iq_arr"].shape, (3, 1))
            with patch.object(plt, "show"):
                for quantity in ("amplitude", "phase", "dB", "std_i", "std_q", "std_iq", "dpdb"):
                    fig = loaded.analyze(quantity)
                    self.assertIn("chip V", fig.axes[0].get_ylabel())
                    plt.close(fig)
            self.run_mock(loaded)  # new files remain runnable

    def test_legacy_file_preserves_voltage_meaning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.h5")
            with h5py.File(path, "w") as f:
                f.attrs.update(
                    dict(
                        freq_center=6e9,
                        freq_span=1e6,
                        df=1e6,
                        num_averages=100,
                        bias_port=1,
                        bias_ramp_rate=0.01,
                    )
                )
                f["bias_arr"] = [0.0, 1.0]
                f["freq_arr"] = [6e9, 6e9 + 1e6]
                f["resp_arr"] = np.ones((2, 2), dtype=complex)
            s = SweepFreqAndDC.load(path)
            self.assertIsNone(s.voltage_divider)
            self.assertIsNone(s.std_iq_arr)
            with patch.object(plt, "show"):
                for quantity in ("amplitude", "phase", "group delay", "dpdb"):
                    fig = s.analyze(quantity)
                    self.assertIn("Presto output V", fig.axes[0].get_ylabel())
                    plt.close(fig)
            with self.assertRaisesRegex(ValueError, "no fluctuation"):
                s.analyze("std_iq")
            with self.assertRaisesRegex(ValueError, "analysis-only"):
                s.run(FakeBias())


if __name__ == "__main__":
    unittest.main(verbosity=2)
