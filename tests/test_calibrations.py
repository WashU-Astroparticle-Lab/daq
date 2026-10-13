"""Offline regression tests for the packaged DAC power calibration.

Run from the repository root with ``python tests/test_calibrations.py``.  Needs only numpy: the
module is loaded by path so ``presto`` is never imported, and the source measurements it is
checked against are committed under ``daq/calibrations/source_data/``.
"""

import csv
import hashlib
import tempfile
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
with np.load(REPO_ROOT / "daq/calibrations/power_calibration.npz") as asset_file:
    asset = {key: asset_file[key].copy() for key in asset_file.files}
cal_freqs = asset["frequency_ghz"]
cal_floors = asset["amp_floor"]
tolerance_db = float(asset["floor_tolerance_db"])
source_dir = REPO_ROOT / "daq/calibrations/source_data/2026-10-12"
builder_spec = importlib.util.spec_from_file_location("calibration_builder", REPO_ROOT / "scripts/build_power_calibration.py")
builder = importlib.util.module_from_spec(builder_spec)
builder_spec.loader.exec_module(builder)
warnings.simplefilter("ignore", CalibrationWarning)
measurements = defaultdict(list)
for row in builder.read_measurements(source_dir):
    measurements[float(row["frequency_mhz"]) / 1000].append((
        float(row["amplitude"]), float(row["median_peak_power_dbm"]),
        row["review_signal_or_timing"] == "True"))
check(
    "251 unfiltered nodes; 14,200 selected observations",
    len(measurements) == 251 and sum(map(len, measurements.values())) == 14200,
)
check("all calibration nodes have source measurements", set(cal_freqs) == set(measurements))
check(
    "source checksums match packaged provenance",
    all(
        hashlib.sha256((source_dir / str(name)).read_bytes()).hexdigest() == sha
        for name, sha in zip(asset["source_files"], asset["source_sha256"])
    ),
)
worst = 0.0
n_verified = n_warned_above_floor = n_silent_below_floor = n_flagged_above_floor = 0
for f, points in sorted(measurements.items()):
    floor = min_verified_amp(f)
    for amplitude, measured, flagged in points:
        modeled, n_warn = warned(lambda: amp_to_power_dbm(f, amplitude))
        if amplitude >= floor:
            n_verified += 1
            worst = max(worst, abs(measured - modeled))
            n_warned_above_floor += n_warn
            n_flagged_above_floor += flagged
        else:
            n_silent_below_floor += n_warn == 0
check(
    "verified measurements agree with model on BOTH sides",
    worst <= tolerance_db,
    f"{n_verified} points, worst absolute residual {worst:.6f} dB",
)
check("no acquisition review flags above empirical floor", n_flagged_above_floor == 0)
check("no warning above empirical floor", n_warned_above_floor == 0)
check("every point below empirical floor warns", n_silent_below_floor == 0)
check(
    "floors stay within measured amplitude range",
    np.all(cal_floors >= 0.001) and np.all(cal_floors < 1),
)
check(
    "low-band full-scale reading is reproduced",
    abs(amp_to_power_dbm(0.1, 1) - 4.355325698852539) < 0.03,
)
check("9 GHz full-scale reading is reproduced", abs(amp_to_power_dbm(9, 1) - (-11.373)) < 0.1)

# Every measured node, including the transition endpoints, survives Hz roundoff.
check(
    "all exact measured nodes reproduce the fitted asset",
    all(amp_to_power_dbm(float(f), 1) == p for f, p in zip(cal_freqs, asset["power_dbm_at_amp_1"])),
)
check(
    "Hz conversion preserves every measured node",
    all(
        amp_to_power_dbm_hz(float(f) * 1e9, 1) == p
        for f, p in zip(cal_freqs, asset["power_dbm_at_amp_1"])
    ),
)
check(
    "adjacent-ULP roundoff preserves measured transition nodes",
    all(
        amp_to_power_dbm(float(np.nextafter(f, np.inf)), 1) == p
        for f, p in zip(cal_freqs, asset["power_dbm_at_amp_1"])
    ),
)
allowed = asset["interpolation_allowed"]
allowed_ok = blocked_ok = floors_ok = True
for i, ok in enumerate(allowed):
    f = float((cal_freqs[i] + cal_freqs[i + 1]) / 2)
    if ok:
        expected = (asset["power_dbm_at_amp_1"][i] + asset["power_dbm_at_amp_1"][i + 1]) / 2
        allowed_ok &= np.isclose(amp_to_power_dbm(f, 1), expected)
        floors_ok &= min_verified_amp(f) == max(cal_floors[i : i + 2])
    else:
        blocked_ok &= all(
            raises(fn)
            for fn in (
                lambda: amp_to_power_dbm(f, 0.1),
                lambda: power_dbm_to_amp(f, -20),
                lambda: amp_to_power_dbm_hz(f * 1e9, 0.1),
                lambda: min_verified_amp(f),
            )
        )
check("all allowed intervals interpolate between adjacent measured nodes", allowed_ok)
check("all rejected intervals fail in BOTH conversions and floor query", blocked_ok)
check("off-grid floors use the more conservative neighbor", floors_ok)
check("issue69: 8 GHz jump is not blended", raises(lambda: amp_to_power_dbm(7.975, 1)))
check(
    "issue69: narrow 7 GHz mode interval is not skipped",
    raises(lambda: amp_to_power_dbm(7.025, 1))
    and np.isfinite(amp_to_power_dbm(7.05, 1))
    and np.isfinite(amp_to_power_dbm(7.1, 1)),
)
check(
    "switch endpoint uncertainty also rejects the other side",
    raises(lambda: amp_to_power_dbm(8.025, 1)),
)
check("new gap is usable away from the 2.4 GHz switch", np.isfinite(amp_to_power_dbm(2.35, 0.1)))
check("gap switch remains excluded", raises(lambda: amp_to_power_dbm(2.39, 0.1)))
check("single-anchor nodes expose amplitude bounds", calibrations.verified_amp_range(.12) == (.1,.1))
check("sparse amplitude checks expose their range", calibrations.verified_amp_range(.1) == (.01,1.) and calibrations.verified_amp_range(1.) == (.01,1.))
for amplitude in [.01, .5, 1.]:
    value, count = warned(lambda: amp_to_power_dbm(.12, amplitude))
    inverse, inverse_count = warned(lambda: power_dbm_to_amp(.12, value))
    check(f"single-anchor estimate warns both directions at {amplitude}", count==1 and inverse_count==1 and np.isclose(inverse,amplitude))
value,count=warned(lambda:amp_to_power_dbm(.12,.1))
inverse,inverse_count=warned(lambda:power_dbm_to_amp(.12,value))
check("exact anchor forward/inverse does not spuriously warn", count==0 and inverse_count==0 and np.isclose(inverse,.1))
check("filtered low-band table excluded", "low_power.csv" not in asset["source_files"] and "unfiltered" in str(asset["reference_plane"]))
for name,freq_col,power_col in [("low_amplitude_checks.csv","frequency_mhz","median_peak_power_dbm"),("high_validation.csv","frequency_ghz","measured_power_dbm")]:
    with (source_dir/name).open(newline="") as stream: checks=list(csv.DictReader(stream))
    errors=[]
    for r in checks:
        f=float(r[freq_col])/(1000 if freq_col=="frequency_mhz" else 1)
        errors.append(abs(float(r[power_col])-amp_to_power_dbm(f,float(r["amplitude"]))))
    check("held-out hardware agreement: "+name,max(errors)<.2, f"max {max(errors):.6f} dB")

check(
    "all measured frequencies forward/inverse round-trip above floor",
    all(
        np.isclose(power_dbm_to_amp(float(f), amp_to_power_dbm(float(f), 0.1)), 0.1)
        for f in cal_freqs
    ),
)

# Rebuilding from committed source must reproduce all arrays, not just rounded examples.
builder_spec = importlib.util.spec_from_file_location(
    "calibration_builder", REPO_ROOT / "scripts/build_power_calibration.py"
)
builder = importlib.util.module_from_spec(builder_spec)
builder_spec.loader.exec_module(builder)
rebuilt = builder.build_calibration(source_dir)
check(
    "packaged asset rebuilds exactly from source",
    set(rebuilt) == set(asset) and all(np.array_equal(rebuilt[k], asset[k]) for k in asset),
)

# Corruption, missing interval flags and obsolete assets must never enable interpolation.
original_path = calibrations._DATA_PATH
invalid_cases = {
    "obsolete filtered schema": {"schema_version": np.array(2)},
    "wrong schema type/value": {"schema_version": np.array(3.5)},
    "nonfinite power": {"power_dbm_at_amp_1": np.full(len(cal_freqs), np.nan)},
    "unordered frequencies": {"frequency_ghz": cal_freqs[::-1]},
    "bad amplitude floor": {"amp_floor": np.zeros(len(cal_freqs))},
    "bad amplitude ceiling": {"amp_ceiling": np.zeros(len(cal_freqs))},
    "nonfinite amplitude ceiling": {"amp_ceiling": np.full(len(cal_freqs),np.nan)},
    "wrong interval count": {"interpolation_allowed": np.ones(len(cal_freqs), dtype=bool)},
    "nonboolean intervals": {"interpolation_allowed": allowed.astype(float)},
}
with tempfile.TemporaryDirectory() as temp:
    try:
        calibrations._DATA_PATH = Path(temp) / "calibration.npz"
        for label, changes in invalid_cases.items():
            np.savez_compressed(calibrations._DATA_PATH, **{**asset, **changes})
            calibrations._load_calibration.cache_clear()
            check(
                "reject asset: " + label, raises(lambda: amp_to_power_dbm(2.8, 0.1), RuntimeError)
            )
        for missing in ("schema_version", "interpolation_allowed", "amp_floor", "amp_ceiling"):
            np.savez_compressed(
                calibrations._DATA_PATH, **{k: v for k, v in asset.items() if k != missing}
            )
            calibrations._load_calibration.cache_clear()
            check(
                "reject missing " + missing,
                raises(lambda: amp_to_power_dbm(2.8, 0.1), RuntimeError),
            )
        calibrations._DATA_PATH.write_bytes(b"not a numpy archive")
        calibrations._load_calibration.cache_clear()
        check("reject corrupt asset", raises(lambda: amp_to_power_dbm(2.8, 0.1), RuntimeError))
    finally:
        calibrations._DATA_PATH = original_path
        calibrations._load_calibration.cache_clear()

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
    min_verified_amp(2.875) == max(min_verified_amp(2.85), min_verified_amp(2.9)),
)

# ---------------------------------------------------------------------------
# Hardware limits and bad input are rejected outright.
# ---------------------------------------------------------------------------
for label, fn in (
    ("out-of-range frequency is rejected", lambda: amp_to_power_dbm(9.01, 0.1)),
    ("below measured band is rejected", lambda: amp_to_power_dbm(0.09, 0.1)),
    ("inverse underflow is rejected", lambda: power_dbm_to_amp(2.8, -1e6)),
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
