"""Offline lifecycle, timing and persistence checks for software LED attachments.

Run with ``python tests/test_software_led.py``. Uses the installed Presto utilities,
but mocks all hardware connections and database calls.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import daq._base as base
import daq.measurements.timestream as tsm
from daq.measurements.qc_trace import QCTrace


class FakeLED:
    def __init__(self, events):
        self.events = events
        self._output = False
        self.mode = "PULS"
        self.on_time = 30e-6
        self.count = 0
        self.fail_start = False
        self.fail_stop = False

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self.events.append("led_on" if value else "led_off")
        self._output = value
        if value and self.fail_start:
            raise RuntimeError("LED start failed")
        if not value and self.fail_stop:
            raise OSError("LED stop failed")

    def settings(self):
        self.events.append("led_settings")
        return dict(
            mode=self.mode,
            output=self.output,
            pulse_on_time_s=self.on_time,
            pulse_off_time_s=0.060,
            pulse_count=self.count,
            pulse_current_a=0.099,
        )


class SoftwareLEDTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.led = FakeLED(self.events)
        self.lockin = MagicMock()
        self.lockin.__enter__.return_value = self.lockin
        self.lockin.tune.side_effect = lambda f, df: (f, df)
        self.lockin.apply_settings.side_effect = lambda: self.events.append("apply")
        self.lockin.get_pixels.side_effect = self.pixels
        self.original_ts_save = tsm.TimeStream.save
        self.original_qc_save = QCTrace.save
        for p in (
            patch.object(tsm.lockin, "Lockin", side_effect=self.connect),
            patch.object(tsm.TimeStream, "save", side_effect=self.save),
            patch.object(QCTrace, "save", side_effect=self.save),
            patch.object(base, "get_next_number", return_value="00000001"),
            patch.object(base, "insert_measurement", return_value="offline"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def connect(self, **kwargs):
        self.events.append("connect")
        return self.lockin

    def pixels(self, n):
        self.events.append("pixels")
        return {1: (np.zeros(1), np.ones((n, 1), complex), np.ones((n, 1), complex))}

    def save(self, **kwargs):
        self.assertFalse(self.led.output)
        self.events.append("save")
        return "offline.h5"

    def stream(self):
        return tsm.TimeStream(
            lo_freq=2.8e9,
            if_freqs=[0.0],
            df=5e4,
            pixel_counts=1000,
            amp=0.01,
            input_port=1,
            output_port=1,
            device="offline",
            external_trigger=False,
            discard_start_ms=0,
        )

    def qc(self):
        return QCTrace(
            readout_freq=2.8e9,
            amp=0.01,
            input_port=1,
            output_port=1,
            num_periods=10,
            discard_start_ms=0,
            device="offline",
        )

    def bias(self):
        bias = MagicMock()
        bias.trigger_port = 2
        bias.samples_for_periods.return_value = 1000
        bias.settings.return_value = dict(function="RAMP", freq_hz=500.0)
        return bias

    def test_snapshot_attachment_does_not_start_led(self):
        ts = self.stream()
        self.led.mode = "TTL"
        ts.attach(led=self.led, bias={"function": "DC", "offset_v": 0.1})
        ts.run(presto_address="offline")
        self.assertNotIn("led_on", self.events)
        self.assertNotIn("led_off", self.events)
        self.assertEqual(ts.bias_mode, "constant")
        self.lockin.set_trigger_out.assert_not_called()

    def test_start_order_metadata_and_cleanup(self):
        ts = self.stream()
        ts.attach_led(self.led)
        ts.attach(bias={"function": "DC", "offset_v": 0.1})
        self.assertNotIn("led_on", self.events)
        ts.run(presto_address="offline", on_acquire=lambda: self.events.append("hook"))
        start = self.events.index("led_on")
        self.assertEqual(self.events[start - 2 : start + 2], ["apply", "hook", "led_on", "pixels"])
        self.assertLess(
            self.events.index("pixels"), len(self.events) - 1 - self.events[::-1].index("led_off")
        )
        self.assertEqual(self.events[-1], "save")
        self.assertEqual(ts.led_synchronization, "software")
        self.assertFalse(ts.led_output)  # configuration snapshot, before the start command
        self.assertGreaterEqual(ts.led_start_completed_unix, ts.led_start_command_unix)
        self.assertGreaterEqual(ts.led_acquire_requested_unix, ts.led_start_completed_unix)
        self.assertGreaterEqual(ts.led_start_command_duration_s, 0)
        self.assertEqual(ts.bias_mode, "constant")
        self.lockin.set_trigger_out.assert_not_called()

    def test_repeat_refreshes_settings_and_restarts(self):
        ts = self.stream()
        ts.attach_led(self.led)
        ts.run(presto_address="offline")
        self.led.on_time = 60e-6
        self.led.count = 1
        ts.run(presto_address="offline")
        self.assertEqual(self.events.count("led_on"), 2)
        self.assertEqual(ts.led_pulse_on_time_s, 60e-6)
        self.assertEqual(ts.led_pulse_count, 1)

    def test_invalid_attachment_preserves_previous_one(self):
        ts = self.stream()
        ts.attach_led(self.led)
        other = FakeLED([])
        for mode, output in (("TTL", False), ("PWM", False), ("PULS", True)):
            other.mode, other._output = mode, output
            with self.assertRaises(ValueError):
                ts.attach_led(other)
        self.assertIs(ts._software_led, self.led)

    def test_mode_change_aborts_and_disables_before_presto(self):
        ts = self.stream()
        ts.attach_led(self.led)
        self.led.mode = "TTL"
        with self.assertRaises(ValueError):
            ts.run(presto_address="offline")
        self.assertFalse(self.led.output)
        self.assertNotIn("connect", self.events)

    def test_setup_failure_disables_led(self):
        ts = self.stream()
        ts.attach_led(self.led)
        self.lockin.tune.side_effect = RuntimeError("setup")
        with self.assertRaisesRegex(RuntimeError, "setup"):
            ts.run(presto_address="offline")
        self.assertFalse(self.led.output)
        self.assertNotIn("led_on", self.events)

    def test_snapshot_failure_disables_before_connecting(self):
        ts = self.stream()
        ts.attach_led(self.led)
        with patch.object(self.led, "settings", side_effect=RuntimeError("snapshot")):
            with self.assertRaisesRegex(RuntimeError, "snapshot"):
                ts.run(presto_address="offline")
        self.assertFalse(self.led.output)
        self.assertNotIn("connect", self.events)

    def test_connection_failure_disables_led(self):
        ts = self.stream()
        ts.attach_led(self.led)
        with patch.object(tsm.lockin, "Lockin", side_effect=RuntimeError("connection")):
            with self.assertRaisesRegex(RuntimeError, "connection"):
                ts.run(presto_address="offline")
        self.assertFalse(self.led.output)

    def test_hook_failure_disables_without_starting(self):
        ts = self.stream()
        ts.attach_led(self.led)
        with self.assertRaisesRegex(RuntimeError, "hook"):
            ts.run(presto_address="offline", on_acquire=MagicMock(side_effect=RuntimeError("hook")))
        self.assertFalse(self.led.output)
        self.assertNotIn("led_on", self.events)
        self.assertNotIn("pixels", self.events)

    def test_start_failure_disables_and_aborts(self):
        ts = self.stream()
        ts.attach_led(self.led)
        self.led.fail_start = True
        with self.assertRaisesRegex(RuntimeError, "LED start"):
            ts.run(presto_address="offline")
        self.assertFalse(self.led.output)
        self.assertNotIn("pixels", self.events)

    def test_acquisition_failure_disables(self):
        ts = self.stream()
        ts.attach_led(self.led)
        self.lockin.get_pixels.side_effect = TimeoutError("pixels")
        with self.assertRaisesRegex(TimeoutError, "pixels"):
            ts.run(presto_address="offline")
        self.assertFalse(self.led.output)

    def test_cleanup_failure_preserves_acquisition_error(self):
        ts = self.stream()
        ts.attach_led(self.led)

        def fail(n):
            self.led.fail_stop = True
            raise TimeoutError("pixels")

        self.lockin.get_pixels.side_effect = fail
        with self.assertLogs("daq.measurements._software_led", level="ERROR"):
            with self.assertRaisesRegex(TimeoutError, "pixels"):
                ts.run(presto_address="offline")

    def test_cleanup_failure_after_success_prevents_save(self):
        ts = self.stream()
        ts.attach_led(self.led)

        def pixels(n):
            self.led.fail_stop = True
            return self.pixels(n)

        self.lockin.get_pixels.side_effect = pixels
        with self.assertRaisesRegex(OSError, "LED stop"):
            ts.run(presto_address="offline")
        self.assertNotIn("save", self.events)

    def test_save_failure_leaves_led_off(self):
        ts = self.stream()
        ts.attach_led(self.led)
        with patch.object(tsm.TimeStream, "save", side_effect=RuntimeError("save")):
            with self.assertRaisesRegex(RuntimeError, "save"):
                ts.run(presto_address="offline")
        self.assertFalse(self.led.output)

    def test_detach_clears_metadata_and_live_start(self):
        ts = self.stream()
        ts.attach_led(self.led)
        ts.run(presto_address="offline")
        ts.attach_led(None)
        self.assertFalse(any(key.startswith("led_") for key in vars(ts)))
        self.events.clear()
        ts.run(presto_address="offline")
        self.assertNotIn("led_on", self.events)

    def test_qc_forwards_led_and_preserves_ramp_routing(self):
        qc, bias = self.qc(), self.bias()
        qc.attach_led(self.led)
        qc.run(bias=bias, presto_address="offline")
        self.assertEqual(self.events.count("led_on"), 1)
        np.testing.assert_array_equal(qc.trigger_states, [0, 1])
        self.assertEqual(qc.qc_stream.led_start_command_unix, qc.led_start_command_unix)
        self.assertEqual(qc.qc_stream.bias_function, "RAMP")
        self.assertFalse(bias.output)
        self.assertFalse(self.led.output)
        self.assertIsNotNone(qc.avg_iq)

    def test_qc_setup_failure_disables_both_instruments(self):
        qc, bias = self.qc(), self.bias()
        qc.attach_led(self.led)
        bias.sawtooth.side_effect = RuntimeError("bias")
        with self.assertRaisesRegex(RuntimeError, "bias"):
            qc.run(bias=bias, presto_address="offline")
        self.assertFalse(bias.output)
        self.assertFalse(self.led.output)

    def test_qc_restarts_and_refreshes_settings(self):
        qc, bias = self.qc(), self.bias()
        qc.attach_led(self.led)
        qc.run(bias=bias, presto_address="offline")
        self.led.on_time = 60e-6
        qc.run(bias=bias, presto_address="offline")
        self.assertEqual(self.events.count("led_on"), 2)
        self.assertEqual(qc.led_pulse_on_time_s, 60e-6)
        self.assertEqual(qc.qc_stream.led_pulse_on_time_s, 60e-6)

    def test_qc_fold_failure_leaves_both_outputs_off(self):
        qc, bias = self.qc(), self.bias()
        qc.attach_led(self.led)
        with patch.object(qc, "fold", side_effect=RuntimeError("fold")):
            with self.assertRaisesRegex(RuntimeError, "fold"):
                qc.run(bias=bias, presto_address="offline")
        self.assertFalse(bias.output)
        self.assertFalse(self.led.output)

    def test_hdf5_and_database_metadata_for_raw_and_qc(self):
        qc = self.qc()
        qc.attach_led(self.led)
        qc.run(bias=self.bias(), presto_address="offline")
        with tempfile.TemporaryDirectory() as tmp:
            for obj, save in ((qc, self.original_qc_save), (qc.qc_stream, self.original_ts_save)):
                path = str(Path(tmp) / (type(obj).__name__ + ".h5"))
                save(obj, save_filename=path)
                loaded = type(obj).load(path)
                self.assertEqual(loaded.led_synchronization, "software")
                self.assertEqual(loaded.led_start_command_unix, obj.led_start_command_unix)
                self.assertEqual(loaded.led_pulse_current_a, 0.099)
                with h5py.File(path) as h5f:
                    self.assertNotIn("_software_led", h5f)
                    self.assertNotIn("_software_led", h5f.attrs)
                doc = base.insert_measurement.call_args.args[0]
                self.assertEqual(doc["led_synchronization"], "software")
                with self.assertRaisesRegex(ValueError, "Reattach"):
                    loaded.run(presto_address="offline")
                loaded.attach_led(None)
                self.assertFalse(any(key.startswith("led_") for key in vars(loaded)))


if __name__ == "__main__":
    unittest.main()
