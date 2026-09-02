"""Offline regression tests for the packaged DAC power calibration.

Run from the repository root with ``python tests/test_calibrations.py``.  Needs only numpy: the
module is loaded by path so ``presto`` is never imported, and the source measurements it is
checked against are committed under ``daq/calibrations/source_data/``.
"""

import csv
import importlib.util
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[1]
module_path = REPO_ROOT / "daq" / "calibrations" / "__init__.py"
spec = importlib.util.spec_from_file_location("daq_calibrations_test", module_path)
calibrations = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(calibrations)

amp_to_power_dbm = calibrations.amp_to_power_dbm
amp_to_power_dbm_hz = calibrations.amp_to_power_dbm_hz
power_dbm_to_amp = calibrations.power_dbm_to_amp
min_verified_amp = calibrations.min_verified_amp
CalibrationWarning = calibrations.CalibrationWarning

results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def warned(fn):
    """Run *fn* and return ``(result, n_calibration_warnings)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, sum(issubclass(w.category, CalibrationWarning) for w in caught)


# ---------------------------------------------------------------------------
# The model reproduces the measurements it was built from, wherever they are verifiable.
# ---------------------------------------------------------------------------
with np.load(REPO_ROOT / "daq" / "calibrations" / "power_calibration.npz") as asset:
    tolerance_db = float(asset["floor_tolerance_db"])
    cal_freqs = np.asarray(asset["frequency_ghz"])
    cal_floors = np.asarray(asset["amp_floor"])

measurements = defaultdict(list)
for path in sorted((REPO_ROOT / "daq" / "calibrations" / "source_data").glob("*combined*.csv")):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            measurements[float(row["freq_ghz"])].append(
                (float(row["amp"]), float(row["power_dbm"]))
            )
check("source measurements are committed", len(measurements) >= 2, f"{len(measurements)} freqs")
check(
    "every calibrated frequency has source data",
    all(f in measurements for f in cal_freqs),
)

worst = 0.0
n_verified = 0
n_warned_above_floor = 0
n_silent_below_floor = 0
for frequency, points in sorted(measurements.items()):
    floor = min_verified_amp(frequency)
    for amplitude, measured in points:
        modeled, n_warn = warned(lambda: amp_to_power_dbm(frequency, amplitude))
        if amplitude >= floor:
            n_verified += 1
            worst = max(worst, measured - modeled)
            n_warned_above_floor += n_warn
        else:
            n_silent_below_floor += n_warn == 0
check(
    "model reproduces every measurement above the verified floor",
    worst <= tolerance_db,
    f"{n_verified} points, worst measured-minus-model {worst:.3f} dB <= {tolerance_db:g} dB",
)
check("no warning is raised above the verified floor", n_warned_above_floor == 0)
check("every point below the verified floor warns", n_silent_below_floor == 0)
check(
    "verified floors lie inside the measured amplitude span",
    np.all(cal_floors > 0.0) and np.all(cal_floors < 1.0),
    f"{cal_floors.min():.4f} to {cal_floors.max():.4f}",
)

# ---------------------------------------------------------------------------
# Golden values at 2.8 GHz from the shipped asset (they move if the asset is rebuilt).
# ---------------------------------------------------------------------------
# A measured source-data point at 2.8 GHz. The four-corner grid this replaced returned
# -51.4 dBm for it, 20 dB below the -31.44 dBm the analyzer read.
measured_amp = 0.0390693993705461
measured_power, n_warn = warned(lambda: amp_to_power_dbm(2.8, measured_amp))
check(
    "2.8 GHz measured point is preserved",
    np.isclose(measured_power, -31.44, atol=0.05) and n_warn == 0,
    f"{measured_power:.4f} dBm",
)

amp_minus_32, n_warn = warned(lambda: power_dbm_to_amp(2.8, -32.0))
check(
    "-32 dBm maps to the expected voltage fraction",
    np.isclose(amp_minus_32, 0.03654, atol=5e-5) and n_warn == 0,
    f"{amp_minus_32:.8f}",
)
check(
    "thirteen -32 dBm tones remain below the conservative amplitude limit",
    13.0 * amp_minus_32 < 1.0,
    f"sum={13.0 * amp_minus_32:.6f}",
)

# ---------------------------------------------------------------------------
# The law itself.
# ---------------------------------------------------------------------------
powers = amp_to_power_dbm(2.8, np.asarray([0.1, 0.5, 1.0]))
check("10x voltage is +20 dB", np.isclose(powers[2] - powers[0], 20.0))
check(
    "forward and inverse conversions round-trip",
    np.isclose(power_dbm_to_amp(2.8, powers[1]), 0.5),
)
check(
    "Hz convenience wrapper matches GHz conversion",
    np.isclose(amp_to_power_dbm_hz(2.8e9, 0.1), amp_to_power_dbm(2.8, 0.1)),
)
check(
    "full scale round-trips",
    np.isclose(power_dbm_to_amp(2.8, amp_to_power_dbm(2.8, 1.0)), 1.0),
)
check(
    "negative amplitude is a phase flip, same power",
    np.isclose(amp_to_power_dbm(2.8, -0.1), amp_to_power_dbm(2.8, 0.1)),
)
check(
    "array input keeps its shape",
    amp_to_power_dbm(2.8, np.full((2, 3), 0.5)).shape == (2, 3),
)

# ---------------------------------------------------------------------------
# The saturation (analyzer-floor) regime warns but still answers.
# ---------------------------------------------------------------------------
floor_28 = min_verified_amp(2.8)
low_power, n_warn = warned(lambda: amp_to_power_dbm(2.8, floor_28 / 2.0))
check(
    "amplitude below the verified floor warns and extrapolates the law",
    n_warn == 1 and np.isclose(low_power, amp_to_power_dbm(2.8, floor_28) - 20.0 * np.log10(2.0)),
    f"floor {floor_28:.4g}",
)
_, n_warn = warned(lambda: amp_to_power_dbm(2.8, np.asarray([floor_28, floor_28 / 3.0])))
check("one warning per call for a mixed array", n_warn == 1)
low_amp, n_warn = warned(lambda: power_dbm_to_amp(2.8, -90.0))
check(
    "very low power request warns and returns the law's amplitude",
    n_warn == 1 and 0.0 < low_amp < floor_28,
    f"{low_amp:.3g}",
)
_, n_warn = warned(lambda: amp_to_power_dbm(2.8, floor_28))
check("amplitude exactly at the floor does not warn", n_warn == 0)
check(
    "floor between calibrated frequencies is the larger neighbour",
    min_verified_amp(2.9) == max(min_verified_amp(2.8), min_verified_amp(3.05)),
)

# ---------------------------------------------------------------------------
# Hardware limits and bad input are rejected outright.
# ---------------------------------------------------------------------------
for label, fn in (
    ("out-of-range frequency is rejected", lambda: amp_to_power_dbm(2.0, 0.1)),
    ("non-finite frequency is rejected", lambda: amp_to_power_dbm(float("nan"), 0.1)),
    ("zero amplitude is rejected", lambda: amp_to_power_dbm(2.8, 0.0)),
    ("amplitude above full scale is rejected", lambda: amp_to_power_dbm(2.8, 1.5)),
    ("non-finite amplitude is rejected", lambda: amp_to_power_dbm(2.8, float("nan"))),
    ("power above full scale is rejected", lambda: power_dbm_to_amp(2.8, 10.0)),
    ("non-finite power is rejected", lambda: power_dbm_to_amp(2.8, float("inf"))),
):
    check(label, raises(fn))

failed = [label for label, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label in failed:
        print("  FAILED:", label)
    sys.exit(1)
