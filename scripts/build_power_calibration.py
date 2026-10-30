"""Build schema 4: filter-offset-corrected low curves and direct gap/high models.

Requires NumPy only. Raw inputs are preserved; corrected tables are derived outputs.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "daq/calibrations/source_data/2026-10-16"
DEFAULT_OUTPUT = REPO_ROOT / "daq/calibrations/power_calibration.npz"
TOLERANCE_DB = 0.5
SOURCE_FILES = [
    "low_filtered_power.csv",
    "low_anchor.csv",
    "gap_power.csv",
    "high_power.csv",
    "low_amplitude_checks.csv",
    "high_validation.csv",
    "filter_pair_comparison.csv",
    "interpolation_policy.json",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def corrected_low_band(input_dir):
    """Shift all 50 original powers per frequency; validation points never fit offsets."""
    input_dir = Path(input_dir)
    original = read_csv(input_dir / "low_filtered_power.csv")
    anchors = read_csv(input_dir / "low_anchor.csv")
    frequencies = sorted({float(r["frequency_mhz"]) for r in original})
    if len(original) != 5050 or frequencies != list(range(100, 2101, 20)):
        raise ValueError("Expected the complete original 101-frequency low-band sweep")
    by_frequency = {float(r["frequency_mhz"]): r for r in anchors}
    if len(anchors) != 101 or len(by_frequency) != 101 or set(by_frequency) != set(frequencies):
        raise ValueError("Expected exactly one unfiltered anchor per low-band frequency")
    corrected, offsets = [], []
    for f in frequencies:
        group = sorted(
            [r for r in original if float(r["frequency_mhz"]) == f],
            key=lambda r: float(r["amplitude"]),
        )
        a = np.array([float(r["amplitude"]) for r in group])
        p = np.array([float(r["median_peak_power_dbm"]) for r in group])
        if (
            len(a) != 50
            or not np.allclose(a, np.geomspace(0.001, 1, 50), rtol=1e-12, atol=1e-15)
            or not np.isfinite(p).all()
        ):
            raise ValueError("Invalid original low-band amplitude grid or power")
        anchor = by_frequency[f]
        values = [
            float(anchor[k])
            for k in (
                "amplitude",
                "median_peak_power_dbm",
                "power_range_db",
                "offset_100ms_sensitivity_db",
                "reading_count",
            )
        ]
        if (
            not np.isfinite(values).all()
            or values[0] != 0.1
            or values[4] < 1
            or not 0 <= values[2] <= TOLERANCE_DB
            or not 0 <= values[3] <= TOLERANCE_DB
        ):
            raise ValueError("Invalid or unstable unfiltered anchor")
        hi = int(np.searchsorted(a, 0.1))
        lo = hi - 1
        if any(group[i]["review_signal_or_timing"] != "False" for i in (lo, hi)):
            raise ValueError("Offset uses a flagged filtered anchor bracket")
        x = np.log10(a)
        weight = float((-1 - x[lo]) / (x[hi] - x[lo]))
        filtered = float(np.interp(-1, x, p))
        delta = values[1] - filtered
        offsets.append(
            dict(
                frequency_mhz=f,
                anchor_amplitude=0.1,
                direct_unfiltered_anchor_dbm=values[1],
                filtered_interpolated_anchor_dbm=filtered,
                filter_offset_db=delta,
                negative_offset=delta < 0,
                filtered_bracket_low_amplitude=a[lo],
                filtered_bracket_high_amplitude=a[hi],
                filtered_bracket_low_power_dbm=p[lo],
                filtered_bracket_high_power_dbm=p[hi],
                upper_bracket_weight_log_amplitude=weight,
                method="unfiltered_A0.1_minus_filtered_log_amplitude_interpolation",
                interpretation="filter/path/session offset; not isolated filter insertion loss",
            )
        )
        for row, amplitude, power in zip(group, a, p):
            corrected.append(
                dict(
                    point_id=row["point_id"],
                    frequency_mhz=f,
                    amplitude=amplitude,
                    original_filtered_power_dbm=power,
                    filter_offset_db=delta,
                    median_peak_power_dbm=power + delta,
                    reading_count=row["reading_count"],
                    power_range_db=row["power_range_db"],
                    review_signal_or_timing=row["review_signal_or_timing"],
                    source_sweep_ids=row["sweep_ids"],
                    measurement_origin="filter_offset_corrected_estimate",
                )
            )
    return corrected, offsets


def write_correction_tables(input_dir=DEFAULT_INPUT_DIR):
    rows, offsets = corrected_low_band(input_dir)
    for name, records in [("low_corrected_power.csv", rows), ("low_filter_offsets.csv", offsets)]:
        with (Path(input_dir) / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)


def read_measurements(input_dir):
    """Corrected low <=2.1; direct gap >2.1 and <2.5; direct high >=2.5 GHz."""
    input_dir = Path(input_dir)
    low, _ = corrected_low_band(input_dir)
    rows = [
        dict(r, support_kind="filter_offset_corrected_curve", source_table="low_filtered_power.csv")
        for r in low
    ]
    for name in ["gap_power.csv", "high_power.csv"]:
        for row in read_csv(input_dir / name):
            f = float(row["frequency_mhz"])
            if (name == "gap_power.csv" and 2100 < f < 2500) or (
                name == "high_power.csv" and f >= 2500
            ):
                rows.append(dict(row, support_kind="direct_unfiltered_sweep", source_table=name))
    seen = set()
    for row in rows:
        f, a, p = [float(row[k]) for k in ("frequency_mhz", "amplitude", "median_peak_power_dbm")]
        if (
            not np.isfinite([f, a, p]).all()
            or f <= 0
            or not 0 < a <= 1
            or int(row["reading_count"]) < 1
            or (f, a) in seen
            or row["review_signal_or_timing"] not in ("True", "False")
        ):
            raise ValueError("Invalid or duplicate source observation")
        seen.add((f, a))
    return rows


def interpolation_flags(frequencies, policy):
    """Allow adjacent-node interpolation throughout each configured frequency band."""
    left, right = frequencies[:-1], frequencies[1:]
    allowed = np.zeros(left.shape, dtype=bool)
    for start, stop in policy["bands_ghz"]:
        allowed |= (left >= start) & (right <= stop)
    return allowed


def build_calibration(input_dir=DEFAULT_INPUT_DIR):
    input_dir = Path(input_dir)
    rows = read_measurements(input_dir)
    policy = json.loads((input_dir / "interpolation_policy.json").read_text())
    if policy["schema_version"] != 4:
        raise ValueError("Expected schema-4 correction policy")
    frequencies = np.unique([float(r["frequency_mhz"]) / 1000 for r in rows])
    if len(frequencies) != 251 or not all(
        any(lo <= f <= hi for lo, hi in policy["bands_ghz"]) for f in frequencies
    ):
        raise ValueError("Invalid frequency coverage")
    low_rows, offsets = corrected_low_band(input_dir)
    grid = np.log10([float(r["amplitude"]) for r in low_rows[:50]])
    powers, floors, kinds, curves, checked_counts, worst, fit_counts = [], [], [], [], [], [], []
    for f in frequencies:
        points = sorted(
            [r for r in rows if float(r["frequency_mhz"]) / 1000 == f],
            key=lambda r: float(r["amplitude"]),
        )
        a = np.array([float(r["amplitude"]) for r in points])
        measured = np.array([float(r["median_peak_power_dbm"]) for r in points])
        flagged = np.array([r["review_signal_or_timing"] == "True" for r in points])
        fit = (a >= 0.1) & ~flagged
        if fit.sum() < 5 or a[-1] != 1:
            raise ValueError("Insufficient sweep support")
        level = float(np.median(measured[fit] - 20 * np.log10(a[fit])))
        residual = measured - (level + 20 * np.log10(a))
        failed = flagged | (abs(residual) > TOLERANCE_DB)
        if failed[-1]:
            raise ValueError(f"Full-scale support failed at {f:g} GHz")
        first = int(np.flatnonzero(failed)[-1]) + 1 if np.any(failed) else 0
        if points[0]["support_kind"] == "filter_offset_corrected_curve":
            if not np.array_equal(np.log10(a), grid) or not np.all(np.diff(measured[first:]) > 0):
                raise ValueError(
                    "Corrected curve must have shared grid and monotonic supported range"
                )
            curve = measured.copy()
            level = float(curve[-1])
        else:
            curve = level + 20 * grid
        powers.append(level)
        floors.append(a[first])
        curves.append(curve)
        kinds.append(points[0]["support_kind"])
        fit_counts.append(int(fit.sum()))
        checked_counts.append(len(a) - first)
        worst.append(float(max(abs(residual[first:]))))
    powers = np.array(powers)
    return dict(
        schema_version=4,
        calibration_id=policy["calibration_id"],
        frequency_ghz=frequencies,
        power_dbm_at_amp_1=powers,
        amp_floor=np.array(floors),
        amp_ceiling=np.ones(len(frequencies)),
        amplitude_support=np.array(kinds),
        curve_log_amplitude=grid,
        curve_power_dbm=np.array(curves),
        low_filter_offset_db=np.array([r["filter_offset_db"] for r in offsets]),
        interpolation_allowed=interpolation_flags(frequencies, policy),
        fit_point_count=np.array(fit_counts),
        verified_point_count=np.array(checked_counts),
        max_verified_abs_residual_db=np.array(worst),
        floor_tolerance_db=TOLERANCE_DB,
        source_files=np.array(SOURCE_FILES),
        source_sha256=np.array(
            [hashlib.sha256((input_dir / n).read_bytes()).hexdigest() for n in SOURCE_FILES]
        ),
        model="Low: corrected measured curves, linear dBm versus log-amplitude. Above 2.1 GHz: P1 + 20*log10(|amp|). Below support: 20 dB/decade from supported floor.",
        reference_plane=policy["reference_plane"],
        measurement=policy["measurement"],
        interpolation_policy_json=json.dumps(policy, sort_keys=True),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    asset = build_calibration(args.input_dir)
    write_correction_tables(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **asset)
    print("Wrote 251 nodes: 101 filter-offset-corrected curves and 150 direct sweep models.")
    print("Excluded interpolation intervals:", int(sum(~asset["interpolation_allowed"])))


if __name__ == "__main__":
    main()
