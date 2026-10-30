"""Offline regression tests for the packaged DAC power and fridge-loss calibrations.

Run from the repository root with ``python tests/test_calibrations.py``. Needs numpy and scipy: the
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
check("all 250 adjacent intervals permit interpolation", len(allowed) == 250 and np.all(allowed))
interpolation_ok = floors_ok = inverse_ok = True
for i in range(len(allowed)):
    # Asymmetric weights exercise interpolation beyond just interval midpoints.
    for weight in (0.1, 0.5, 0.9):
        f = float((1 - weight) * cal_freqs[i] + weight * cal_freqs[i + 1])
        floor = max(cal_floors[i : i + 2])
        floors_ok &= calibrations.verified_amp_range(f) == (floor, 1.0)
        for amplitude in (float(floor), 0.1, 1.0):
            expected = (1 - weight) * amp_to_power_dbm(
                float(cal_freqs[i]), amplitude
            ) + weight * amp_to_power_dbm(float(cal_freqs[i + 1]), amplitude)
            actual = amp_to_power_dbm(f, amplitude)
            interpolation_ok &= np.isclose(actual, expected, atol=1e-12)
            interpolation_ok &= np.isclose(
                amp_to_power_dbm_hz(f * 1e9, amplitude), expected, atol=1e-12
            )
            # Use the API's exact full-scale value to avoid a rounding-only excess.
            inverse_ok &= np.isclose(power_dbm_to_amp(f, actual), amplitude, atol=1e-12)
check("all intervals interpolate neighboring curves at three frequency weights", interpolation_ok)
check("all intervals invert the interpolated curve", inverse_ok)
check("off-grid support uses the more conservative neighbor", floors_ok)

# The reported failure and representative former switch/jump exclusions.
for f in (
    0.95,
    1.03,
    2.01,
    2.39,
    2.41,
    2.975,
    3.025,
    4.275,
    4.875,
    4.925,
    5.575,
    5.625,
    6.025,
    6.575,
    6.9968,
    7.025,
    7.075,
    7.125,
    7.975,
    8.025,
):
    check(
        f"formerly excluded frequency {f:g} GHz converts and round-trips",
        np.isclose(amp_to_power_dbm(f, power_dbm_to_amp(f, -32.0)), -32.0),
    )
check(
    "6.9968 GHz uses the measured 6.95 and 7.00 GHz levels",
    np.isclose(
        power_dbm_to_amp(6.9968, -32.0),
        10 ** ((-32 - (0.064 * -6.949591174270165 + 0.936 * -7.096530798709757)) / 20),
        rtol=1e-12,
    ),
)
check(
    "disjoint configured bands do not bridge an uncovered gap",
    np.array_equal(
        builder.interpolation_flags(
            np.array([1.0, 2.0, 3.0, 4.0]),
            {"bands_ghz": [[1.0, 2.0], [3.0, 4.0]]},
        ),
        [True, False, True],
    ),
)
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
