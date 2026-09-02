"""Offline regression tests for the packaged DAC power calibration.

Run from the repository root with ``python tests/test_calibrations.py``.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

module_path = Path(__file__).parents[1] / "daq" / "calibrations" / "__init__.py"
spec = importlib.util.spec_from_file_location("daq_calibrations_test", module_path)
calibrations = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(calibrations)

amp_to_power_dbm = calibrations.amp_to_power_dbm
amp_to_power_dbm_hz = calibrations.amp_to_power_dbm_hz
power_dbm_to_amp = calibrations.power_dbm_to_amp

results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# A measured source-data point at 2.8 GHz. The previous grid-generation code
# discarded this point and returned about -11 dBm instead of -31.44 dBm.
measured_amp = 0.0390693993705461
measured_power = amp_to_power_dbm(2.8, measured_amp)
check(
    "2.8 GHz measured point is preserved",
    np.isclose(measured_power, -31.44, atol=0.05),
    f"{measured_power:.4f} dBm",
)

amp_minus_32 = power_dbm_to_amp(2.8, -32.0)
check(
    "-32 dBm maps to the expected voltage fraction",
    np.isclose(amp_minus_32, 0.03654, atol=5e-5),
    f"{amp_minus_32:.8f}",
)
check(
    "thirteen -32 dBm tones remain below the conservative amplitude limit",
    13.0 * amp_minus_32 < 1.0,
    f"sum={13.0 * amp_minus_32:.6f}",
)

powers = amp_to_power_dbm(2.8, np.asarray([0.01, 0.1, 0.5, 1.0]))
check("10x voltage is +20 dB", np.isclose(powers[1] - powers[0], 20.0))
check(
    "forward and inverse conversions round-trip",
    np.isclose(power_dbm_to_amp(2.8, powers[2]), 0.5),
)
check(
    "Hz convenience wrapper matches GHz conversion",
    np.isclose(amp_to_power_dbm_hz(2.8e9, 0.1), amp_to_power_dbm(2.8, 0.1)),
)
check(
    "calibration boundaries round-trip",
    np.isclose(power_dbm_to_amp(2.8, amp_to_power_dbm(2.8, 0.001)), 0.001)
    and np.isclose(power_dbm_to_amp(2.8, amp_to_power_dbm(2.8, 1.0)), 1.0),
)

for label, fn in (
    ("out-of-range frequency is rejected", lambda: amp_to_power_dbm(2.0, 0.1)),
    ("zero amplitude is rejected", lambda: amp_to_power_dbm(2.8, 0.0)),
    ("out-of-range power is rejected", lambda: power_dbm_to_amp(2.8, 10.0)),
):
    try:
        fn()
        check(label, False)
    except ValueError:
        check(label, True)

failed = [label for label, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label in failed:
        print("  FAILED:", label)
    sys.exit(1)
