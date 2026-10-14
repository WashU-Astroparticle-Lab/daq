"""Offline regression tests for the packaged DAC power and fridge-loss calibrations.

Run from the repository root with ``python tests/test_calibrations.py``. Needs numpy and scipy: the
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
nominal_fridge_loss = calibrations.nominal_fridge_loss
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


# ---------------------------------------------------------------------------
# Fridge loss: measured S21, frequency units, interpolation and valid band.
# ---------------------------------------------------------------------------
def loopback_loss(freq_hz, **kwargs):
    return nominal_fridge_loss(freq_hz, include_output=True, **kwargs)


fridge_path = REPO_ROOT / "daq/calibrations/source_data/Empty_Line_port_2_-_2026-06-15.s2p"
fridge_data = np.loadtxt(fridge_path, comments=["#", "!"])
check(
    "raw fridge loss reproduces all measured S21 magnitudes with attenuation positive",
    np.allclose(
        loopback_loss(fridge_data[:, 0], smooth=False),
        -fridge_data[:, 3],
        rtol=0,
        atol=1e-10,
    ),
)
check(
    "fridge-loss endpoints accept Hz and return a scalar float",
    isinstance(loopback_loss(1e10), float)
    and np.isclose(loopback_loss(1e5, smooth=False), 72.0394968383)
    and np.isclose(loopback_loss(1e10, smooth=False), 78.4864410019)
    and np.all(np.isfinite(loopback_loss([1e5, 1e10]))),
)
midpoint_hz = (fridge_data[280, 0] + fridge_data[281, 0]) / 2
check(
    "raw fridge loss interpolates in dB between measured frequencies",
    np.isclose(
        loopback_loss(midpoint_hz, smooth=False),
        -(fridge_data[280, 3] + fridge_data[281, 3]) / 2,
        rtol=0,
        atol=1e-10,
    ),
)
fridge_grid = np.array([[1e5, midpoint_hz], [1e10, 2.8e9]])
check(
    "fridge-loss array input retains its shape and matches scalar calls",
    loopback_loss(fridge_grid).shape == fridge_grid.shape
    and np.allclose(
        loopback_loss(fridge_grid).ravel(),
        [loopback_loss(f) for f in fridge_grid.ravel()],
    ),
)
smoothed = loopback_loss(fridge_data[:, 0])
check(
    "smoothing is enabled by default and changes the measured ripple",
    np.array_equal(smoothed, loopback_loss(fridge_data[:, 0], smooth=True))
    and not np.allclose(smoothed, -fridge_data[:, 3]),
)
check(
    "smoothing reduces point-to-point curvature of the measured curve",
    np.std(np.diff(smoothed, n=2)) < 0.2 * np.std(np.diff(-fridge_data[:, 3], n=2)),
)
check(
    "smoothed result interpolates between adjacent smoothed calibration values",
    np.isclose(loopback_loss(midpoint_hz), (smoothed[280] + smoothed[281]) / 2, rtol=0, atol=1e-10),
)
# A cubic trend must survive the cubic filter, including its edge fits. An alternating
# perturbation should be suppressed. Swap only the source fixture and restore the cache.
original_loader = calibrations._load_fridge_loss
synthetic_freqs = np.linspace(1e5, 1e10, 101)
x = np.linspace(-1, 1, 101)
trend = 70 + 2 * x + x**2 + 0.5 * x**3
try:
    calibrations._load_fridge_loss = lambda include_output: (synthetic_freqs, trend)
    calibrations._smoothed_fridge_loss.cache_clear()
    check(
        "smoothing preserves a cubic trend including both band edges",
        np.allclose(loopback_loss(synthetic_freqs), trend, rtol=0, atol=1e-10),
    )
    noisy = trend + 0.5 * (-1.0) ** np.arange(101)
    calibrations._load_fridge_loss = lambda include_output: (synthetic_freqs, noisy)
    calibrations._smoothed_fridge_loss.cache_clear()
    check(
        "smoothing suppresses a synthetic alternating perturbation",
        np.sqrt(np.mean((loopback_loss(synthetic_freqs) - trend) ** 2)) < 0.1,
    )
finally:
    calibrations._load_fridge_loss = original_loader
    calibrations._smoothed_fridge_loss.cache_clear()

# Room-temperature input-to-MC loss is now the default. Check independently against
# the committed digitization, including the red (not yellow) trace near 4 GHz.
input_data = np.loadtxt(
    REPO_ROOT / "daq/calibrations/source_data/fridge_loss_to_mc.csv", delimiter=","
)
input_freqs, input_db = input_data.T
check(
    "default raw input loss reproduces all 201 digitized red-trace values",
    input_data.shape == (201, 2)
    and np.array_equal(nominal_fridge_loss(input_freqs, smooth=False), input_db),
)
check(
    "input calibration matches screenshot axes and red trace near 4 GHz",
    input_freqs[0] == 1e5
    and input_freqs[-1] == 1e10
    and abs(nominal_fridge_loss(1e5, smooth=False) - 59.4) < 0.1
    and abs(nominal_fridge_loss(1e10, smooth=False) - 75.4) < 0.1
    and abs(nominal_fridge_loss(4e9, smooth=False) - 68.4) < 0.15,
)
input_midpoints = (input_freqs[:-1] + input_freqs[1:]) / 2
check(
    "raw input loss interpolates all interval midpoints in dB",
    np.allclose(
        nominal_fridge_loss(input_midpoints, smooth=False),
        (input_db[:-1] + input_db[1:]) / 2,
        rtol=0,
        atol=1e-10,
    ),
)
input_smoothed = nominal_fridge_loss(input_freqs)
check(
    "input smoothing reduces point-to-point curvature",
    np.std(np.diff(input_smoothed, n=2)) < 0.5 * np.std(np.diff(input_db, n=2))
    and not np.array_equal(input_smoothed, input_db),
)
check(
    "smoothed input loss interpolates all interval midpoints",
    np.allclose(
        nominal_fridge_loss(input_midpoints),
        (input_smoothed[:-1] + input_smoothed[1:]) / 2,
        rtol=0,
        atol=1e-10,
    ),
)
check(
    "include_output=True retains the previously published 2.8 GHz results",
    np.isclose(loopback_loss(2.8e9), 68.0334289878, rtol=0, atol=1e-9)
    and np.isclose(loopback_loss(2.8e9, smooth=False), 67.8543393797, rtol=0, atol=1e-9),
)
for smooth in (False, True):
    expected_input = nominal_fridge_loss(fridge_grid, smooth=smooth)
    expected_loopback = loopback_loss(fridge_grid, smooth=smooth)
    for output in (True, False, True, False):
        expected = expected_loopback if output else expected_input
        actual = nominal_fridge_loss(fridge_grid, include_output=output, smooth=smooth)
        check(
            f"selection/cache, shape and scalar consistency: {output=}, {smooth=}",
            actual.shape == fridge_grid.shape
            and np.array_equal(actual, expected)
            and np.array_equal(
                actual.ravel(),
                [
                    nominal_fridge_loss(f, include_output=output, smooth=smooth)
                    for f in fridge_grid.ravel()
                ],
            ),
        )
    check(
        f"omitting include_output selects input loss, distinct from loopback: {smooth=}",
        np.array_equal(
            expected_input, nominal_fridge_loss(fridge_grid, include_output=False, smooth=smooth)
        )
        and not np.allclose(expected_input, expected_loopback),
    )

# The default must not depend on the previous loopback source being loaded.
original_path = calibrations._FRIDGE_LOSS_PATH
try:
    calibrations._FRIDGE_LOSS_PATH = original_path.with_name("missing-loopback.s2p")
    calibrations._load_fridge_loss.cache_clear()
    calibrations._smoothed_fridge_loss.cache_clear()
    check(
        "default source works independently of the loopback file",
        np.isfinite(nominal_fridge_loss(2.8e9))
        and np.isfinite(nominal_fridge_loss(2.8e9, smooth=False))
        and raises(lambda: loopback_loss(2.8e9), RuntimeError),
    )
finally:
    calibrations._FRIDGE_LOSS_PATH = original_path
    calibrations._load_fridge_loss.cache_clear()
    calibrations._smoothed_fridge_loss.cache_clear()

try:
    calibrations._load_fridge_loss = lambda include_output: (synthetic_freqs, trend)
    calibrations._smoothed_fridge_loss.cache_clear()
    check(
        "input smoothing preserves a cubic trend including both band edges",
        np.allclose(nominal_fridge_loss(synthetic_freqs), trend, rtol=0, atol=1e-10),
    )
finally:
    calibrations._load_fridge_loss = original_loader
    calibrations._smoothed_fridge_loss.cache_clear()

for bad_freq in (0.0, -1.0, 2.8, 9e4, 1.001e10, np.nan, np.inf, [2.8e9, np.nan], [2.8e9, 1.1e10]):
    check(
        f"fridge loss rejects invalid frequency {bad_freq}",
        all(
            raises(lambda: nominal_fridge_loss(bad_freq, smooth=smooth, include_output=output))
            for smooth in (False, True)
            for output in (False, True)
        ),
    )

failed = [label for label, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label in failed:
        print("  FAILED:", label)
    sys.exit(1)
