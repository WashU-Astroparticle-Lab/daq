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
    return result, sum(w.category is CalibrationWarning for w in caught)


# ---------------------------------------------------------------------------
# The model reproduces the measurements it was built from, wherever they are verifiable.
# ---------------------------------------------------------------------------
with np.load(REPO_ROOT / "daq/calibrations/power_calibration.npz") as asset_file:
    asset = {key: asset_file[key].copy() for key in asset_file.files}
cal_freqs = asset["frequency_ghz"]
cal_floors = asset["amp_floor"]
tolerance_db = float(asset["floor_tolerance_db"])
source_dir = REPO_ROOT / "daq/calibrations/source_data/2026-10-16"
builder_spec = importlib.util.spec_from_file_location(
    "calibration_builder", REPO_ROOT / "scripts/build_power_calibration.py"
)
builder = importlib.util.module_from_spec(builder_spec)
builder_spec.loader.exec_module(builder)
warnings.simplefilter("ignore", CalibrationWarning)
measurements = defaultdict(list)
for row in builder.read_measurements(source_dir):
    measurements[float(row["frequency_mhz"]) / 1000].append(
        (
            float(row["amplitude"]),
            float(row["median_peak_power_dbm"]),
            row["review_signal_or_timing"] == "True",
        )
    )
check(
    "251 nodes; 5,050 corrected low points and 14,050 direct gap/high points",
    len(measurements) == 251 and sum(map(len, measurements.values())) == 19100,
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
    "supported source points agree; corrected points are not independent validation",
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
check(
    "corrected low curves expose an amplitude range",
    calibrations.verified_amp_range(0.12)[0] < 0.01
    and calibrations.verified_amp_range(0.12)[1] == 1.0,
)
check(
    "filtered low input and correction provenance retained",
    "low_filtered_power.csv" in asset["source_files"]
    and "filter offset" in str(asset["reference_plane"]),
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    value = amp_to_power_dbm(0.12, 0.1)
    inverse = power_dbm_to_amp(0.12, value)
check(
    "both low-band conversions identify filter-offset estimates",
    len(caught) == 2
    and all(w.category is calibrations.FilterOffsetWarning for w in caught)
    and np.isclose(inverse, 0.1),
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    amp_to_power_dbm(2.1, 0.1)
check(
    "2.1 GHz endpoint is explicitly corrected",
    len(caught) == 1 and caught[0].category is calibrations.FilterOffsetWarning,
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    amp_to_power_dbm(2.12, 0.1)
check("direct gap does not emit correction warning", not caught)
for name, freq_col, power_col in [
    ("low_amplitude_checks.csv", "frequency_mhz", "median_peak_power_dbm"),
    ("high_validation.csv", "frequency_ghz", "measured_power_dbm"),
]:
    with (source_dir / name).open(newline="") as stream:
        checks = list(csv.DictReader(stream))
    errors = []
    for r in checks:
        f = float(r[freq_col]) / (1000 if freq_col == "frequency_mhz" else 1)
        errors.append(abs(float(r[power_col]) - amp_to_power_dbm(f, float(r["amplitude"]))))
    check("held-out hardware agreement: " + name, max(errors) < 0.2, f"max {max(errors):.6f} dB")

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

# Independent checks of the correction formula and curve-based inversion.
original = builder.read_csv(source_dir / "low_filtered_power.csv")
corrected = builder.read_csv(source_dir / "low_corrected_power.csv")
offsets = builder.read_csv(source_dir / "low_filter_offsets.csv")
check(
    "all 5,050 original observations preserved in corrected table",
    len(original) == len(corrected) == 5050
    and all(
        str(a["point_id"]) == str(b["point_id"])
        and a["review_signal_or_timing"] == b["review_signal_or_timing"]
        for a, b in zip(original, corrected)
    ),
)
check(
    "corrected table is exactly original plus recorded offset",
    all(
        abs(
            float(b["median_peak_power_dbm"])
            - float(a["median_peak_power_dbm"])
            - float(b["filter_offset_db"])
        )
        < 1e-12
        for a, b in zip(original, corrected)
    ),
)
check(
    "all offsets follow recorded anchor interpolation",
    all(
        abs(
            float(r["filter_offset_db"])
            - (
                float(r["direct_unfiltered_anchor_dbm"])
                - (
                    (1 - float(r["upper_bracket_weight_log_amplitude"]))
                    * float(r["filtered_bracket_low_power_dbm"])
                    + float(r["upper_bracket_weight_log_amplitude"])
                    * float(r["filtered_bracket_high_power_dbm"])
                )
            )
        )
        < 1e-12
        for r in offsets
    ),
)
check(
    "three negative offsets retained without clamping",
    sum(float(r["filter_offset_db"]) < 0 for r in offsets) == 3
    and all(
        (r["negative_offset"] == "True") == (float(r["filter_offset_db"]) < 0) for r in offsets
    ),
)
anchors = builder.read_csv(source_dir / "low_anchor.csv")
check(
    "all 101 corrected curves pass through unfiltered anchors",
    all(
        abs(
            amp_to_power_dbm(float(r["frequency_mhz"]) / 1000, 0.1)
            - float(r["median_peak_power_dbm"])
        )
        < 1e-12
        for r in anchors
    ),
)
shape_ok = True
inverse_ok = True
for f in cal_freqs[cal_freqs <= 2.1]:
    group = [
        r
        for r in corrected
        if float(r["frequency_mhz"]) / 1000 == f and float(r["amplitude"]) >= min_verified_amp(f)
    ]
    aa = np.array([float(r["amplitude"]) for r in group])
    pp = np.array([float(r["median_peak_power_dbm"]) for r in group])
    shape_ok &= np.allclose(amp_to_power_dbm(float(f), aa), pp, rtol=0, atol=1e-12)
    for j in range(len(aa) - 1):
        amplitude = float(np.sqrt(aa[j] * aa[j + 1]))
        expected = float((pp[j] + pp[j + 1]) / 2)
        shape_ok &= abs(amp_to_power_dbm(float(f), amplitude) - expected) < 1e-12
        inverse_ok &= abs(power_dbm_to_amp(float(f), expected) / amplitude - 1) < 1e-11
check("low functions preserve supported corrected samples and between-sample shape", shape_ok)
check("inverse solves the corrected curves between every supported pair", inverse_ok)
check(
    "low-band curve is not silently replaced by a 20 dB/decade line",
    abs(amp_to_power_dbm(0.1, 0.01) - (amp_to_power_dbm(0.1, 0.1) - 20)) > 0.05,
)
roundtrip = True
for i, ok in enumerate(allowed):
    if ok:
        f = float((cal_freqs[i] + cal_freqs[i + 1]) / 2)
        for a in np.geomspace(min_verified_amp(f) / 2, 1, 9):
            roundtrip &= abs(power_dbm_to_amp(f, amp_to_power_dbm(f, float(a))) / a - 1) < 1e-11
check("all allowed off-grid curves invert consistently, including below-floor estimates", roundtrip)
for f in [0.1, 0.12, 1.0, 2.1]:
    floor = min_verified_amp(f)
    value, nwarn = warned(lambda: amp_to_power_dbm(f, floor / 2))
    inverse, nwarn_inverse = warned(lambda: power_dbm_to_amp(f, value))
    check(
        f"weak corrected tail extrapolates from quality floor at {f:g} GHz",
        nwarn == nwarn_inverse == 1
        and np.isclose(value, amp_to_power_dbm(f, floor) - 20 * np.log10(2))
        and np.isclose(inverse, floor / 2),
    )
# Verify direct high/gap behavior against independently recomputed original analytic levels.
direct_ok = True
for f, points in measurements.items():
    if f > 2.1:
        fit = [p - 20 * np.log10(a) for a, p, flagged in points if a >= 0.1 and not flagged]
        expected = float(np.median(fit))
        direct_ok &= np.allclose(
            amp_to_power_dbm(f, np.array([0.001, 0.01, 0.1, 0.5, 1.0])),
            expected + 20 * np.log10([0.001, 0.01, 0.1, 0.5, 1.0]),
            rtol=0,
            atol=1e-12,
        )
check("direct gap/high model behavior preserved", direct_ok)
overlap = [
    r
    for r in builder.read_csv(source_dir / "gap_power.csv")
    if float(r["frequency_mhz"]) == 2100
    and float(r["amplitude"]) >= min_verified_amp(2.1)
    and r["review_signal_or_timing"] == "False"
]
overlap_error = max(
    abs(float(r["median_peak_power_dbm"]) - amp_to_power_dbm(2.1, float(r["amplitude"])))
    for r in overlap
)
check(
    "independent direct 2.1-GHz sweep validates corrected endpoint",
    len(overlap) == 42 and overlap_error < 0.2,
    f"max {overlap_error:.6f} dB",
)
check(
    "original filtered source bytes remain unchanged",
    (source_dir / "low_filtered_power.csv").read_bytes()
    == (source_dir.parent / "2026-10-02/low_power.csv").read_bytes(),
)
check(
    "off-grid amplitude values blend the two actual curves",
    all(
        abs(
            amp_to_power_dbm(float((cal_freqs[i] + cal_freqs[i + 1]) / 2), 0.1)
            - (
                amp_to_power_dbm(float(cal_freqs[i]), 0.1)
                + amp_to_power_dbm(float(cal_freqs[i + 1]), 0.1)
            )
            / 2
        )
        < 1e-12
        for i, ok in enumerate(allowed)
        if ok
    ),
)

# Corruption, missing interval flags and obsolete assets must never enable interpolation.
original_path = calibrations._DATA_PATH
invalid_cases = {
    "obsolete filtered schema": {"schema_version": np.array(2)},
    "obsolete anchor-only schema": {"schema_version": np.array(3)},
    "wrong schema type/value": {"schema_version": np.array(3.5)},
    "nonfinite power": {"power_dbm_at_amp_1": np.full(len(cal_freqs), np.nan)},
    "unordered frequencies": {"frequency_ghz": cal_freqs[::-1]},
    "bad amplitude floor": {"amp_floor": np.zeros(len(cal_freqs))},
    "bad amplitude ceiling": {"amp_ceiling": np.zeros(len(cal_freqs))},
    "nonfinite amplitude ceiling": {"amp_ceiling": np.full(len(cal_freqs), np.nan)},
    "missing curve row": {"curve_power_dbm": asset["curve_power_dbm"][:-1]},
    "nonfinite curve": {"curve_power_dbm": np.full(asset["curve_power_dbm"].shape, np.nan)},
    "unordered curve grid": {"curve_log_amplitude": asset["curve_log_amplitude"][::-1]},
    "inconsistent curve endpoint": {"power_dbm_at_amp_1": asset["power_dbm_at_amp_1"] + 1},
    "nonmonotonic supported curve": {
        "curve_power_dbm": np.concatenate(
            [asset["curve_power_dbm"][:, :-1][:, ::-1], asset["curve_power_dbm"][:, -1:]], axis=1
        )
    },
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
        for missing in (
            "schema_version",
            "interpolation_allowed",
            "amp_floor",
            "amp_ceiling",
            "curve_log_amplitude",
            "curve_power_dbm",
        ):
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
