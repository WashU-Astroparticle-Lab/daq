"""Build the unfiltered schema-3 calibration from preserved measured source tables.

Requires numpy only. Historical filtered low-band powers are never inputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / 'daq/calibrations/source_data/2026-10-12'
DEFAULT_OUTPUT = REPO_ROOT / 'daq/calibrations/power_calibration.npz'
TOLERANCE_DB = .5
SOURCE_FILES = ['low_anchor.csv', 'gap_power.csv', 'high_power.csv',
                'low_amplitude_checks.csv', 'high_validation.csv', 'interpolation_policy.json']


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def read_measurements(input_dir):
    """Select one source per node: low anchors <2.1, gap <2.5, high >=2.5 GHz."""
    rows, seen = [], set()
    for name, kind in [('low_anchor.csv', 'single_anchor'), ('gap_power.csv', 'dense_sweep'),
                       ('high_power.csv', 'dense_sweep')]:
        for row in read_csv(input_dir / name):
            f, a, p = [float(row[k]) for k in ('frequency_mhz', 'amplitude', 'median_peak_power_dbm')]
            if not np.isfinite([f, a, p]).all() or f <= 0 or not 0 < a <= 1:
                raise ValueError('Invalid source frequency, amplitude or power')
            if int(row['reading_count']) < 1:
                raise ValueError('Source point has no reading')
            selected = (name == 'low_anchor.csv' and f < 2100 or
                        name == 'gap_power.csv' and 2100 <= f < 2500 or
                        name == 'high_power.csv' and f >= 2500)
            if not selected:
                continue
            if (f, a) in seen:
                raise ValueError('Duplicate selected frequency/amplitude pair')
            seen.add((f, a))
            if kind == 'single_anchor':
                if a != .1 or float(row['power_range_db']) > TOLERANCE_DB or float(row['offset_100ms_sensitivity_db']) > TOLERANCE_DB:
                    raise ValueError('Invalid or unstable low-band anchor')
                row['review_signal_or_timing'] = 'False'
            elif row['review_signal_or_timing'] not in ('True', 'False'):
                raise ValueError('Missing explicit quality flag')
            rows.append(dict(row, support_kind=kind, source_table=name))
    return rows


def interpolation_flags(frequencies, policy, powers):
    left, right = frequencies[:-1], frequencies[1:]
    allowed = np.zeros(left.shape, dtype=bool)
    for start, stop in policy['bands_ghz']:
        allowed |= (left >= start) & (right <= stop)
    for upper in policy['switch_scan_upper_ghz']:
        lower = upper - policy['switch_scan_step_ghz']
        allowed &= ~((left <= upper + 1e-12) & (right >= lower - 1e-12))
    allowed &= np.abs(np.diff(powers)) <= policy['maximum_adjacent_jump_db']
    return allowed


def build_calibration(input_dir=DEFAULT_INPUT_DIR):
    input_dir = Path(input_dir)
    rows = read_measurements(input_dir)
    policy = json.loads((input_dir / 'interpolation_policy.json').read_text())
    checks = read_csv(input_dir / 'low_amplitude_checks.csv')
    frequencies = np.unique([float(r['frequency_mhz']) / 1000 for r in rows])
    if len(frequencies) < 2 or not all(any(lo <= f <= hi for lo, hi in policy['bands_ghz']) for f in frequencies):
        raise ValueError('Invalid frequency coverage')
    powers, floors, ceilings, kinds, fit_counts, checked_counts, worst = [], [], [], [], [], [], []
    for f in frequencies:
        points = sorted([r for r in rows if float(r['frequency_mhz']) / 1000 == f], key=lambda r: float(r['amplitude']))
        amps = np.array([float(r['amplitude']) for r in points])
        measured = np.array([float(r['median_peak_power_dbm']) for r in points])
        flagged = np.array([r['review_signal_or_timing'] == 'True' for r in points])
        if points[0]['support_kind'] == 'single_anchor':
            if len(points) != 1 or amps[0] != .1:
                raise ValueError('Expected exactly one amplitude-0.1 anchor')
            p1 = float(measured[0] + 20)
            floor = ceiling = .1
            kind, nfit, ncheck, error = 'single_anchor', 1, 1, 0.
            validation = [r for r in checks if float(r['frequency_mhz']) / 1000 == f]
            if validation:
                va = np.array([float(r['amplitude']) for r in validation])
                vp = np.array([float(r['median_peak_power_dbm']) for r in validation])
                residual = vp - (p1 + 20 * np.log10(va))
                if (not np.allclose(va, np.geomspace(.01, 1, 5)) or not np.isfinite(vp).all()
                        or max(abs(residual)) > TOLERANCE_DB
                        or any(r['review_signal_or_timing'] != 'False' for r in validation)):
                    raise ValueError(f'Amplitude validation failed at {f:g} GHz')
                floor, ceiling, kind, ncheck, error = .01, 1., 'sparse_sweep_check', 5, float(max(abs(residual)))
        else:
            fit = (amps >= .1) & ~flagged
            if fit.sum() < 5 or amps[-1] != 1.:
                raise ValueError(f'Insufficient sweep support at {f:g} GHz')
            p1 = float(np.median(measured[fit] - 20 * np.log10(amps[fit])))
            residual = measured - (p1 + 20 * np.log10(amps))
            failed = flagged | (abs(residual) > TOLERANCE_DB)
            if failed[-1]:
                raise ValueError(f'Full-scale validation failed at {f:g} GHz')
            first = int(np.flatnonzero(failed)[-1]) + 1 if np.any(failed) else 0
            floor, ceiling, kind = float(amps[first]), 1., 'dense_sweep'
            nfit, ncheck, error = int(fit.sum()), len(amps)-first, float(max(abs(residual[first:])))
        powers.append(p1); floors.append(floor); ceilings.append(ceiling); kinds.append(kind)
        fit_counts.append(nfit); checked_counts.append(ncheck); worst.append(error)
    powers = np.array(powers)
    return dict(schema_version=3, calibration_id=policy['calibration_id'], frequency_ghz=frequencies,
        power_dbm_at_amp_1=powers, amp_floor=np.array(floors), amp_ceiling=np.array(ceilings),
        amplitude_support=np.array(kinds), interpolation_allowed=interpolation_flags(frequencies, policy, powers),
        fit_point_count=np.array(fit_counts), verified_point_count=np.array(checked_counts),
        max_verified_abs_residual_db=np.array(worst), floor_tolerance_db=TOLERANCE_DB,
        source_files=np.array(SOURCE_FILES), source_sha256=np.array([hashlib.sha256((input_dir/n).read_bytes()).hexdigest() for n in SOURCE_FILES]),
        model='power_dbm_at_amp_1 + 20*log10(|amp|)', reference_plane=policy['reference_plane'],
        measurement=policy['measurement'], interpolation_policy_json=json.dumps(policy, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', nargs='?', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    asset = build_calibration(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **asset)
    print(f"Wrote {len(asset['frequency_ghz'])} unfiltered frequency nodes to {args.output}")
    print('Amplitude support:', dict(zip(*np.unique(asset['amplitude_support'], return_counts=True))))
    print('Excluded interpolation intervals:', int(sum(~asset['interpolation_allowed'])))


if __name__ == '__main__':
    main()
