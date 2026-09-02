"""Build the packaged DAC power calibration from spectrum-analyzer CSV files.

The source measurements contain ``freq_ghz``, ``amp`` and ``power_dbm``
columns.  High-amplitude points are used to estimate the
full-scale power at each frequency; amplitude dependence is kept analytic as
``20 * log10(amp)`` because ``amp`` is a voltage fraction.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_measurements(input_dir: Path) -> tuple[list[tuple[float, float, float]], list[str]]:
    """Read and de-duplicate all combined calibration CSVs in *input_dir*."""
    files = sorted(input_dir.glob("*combined*.csv"))
    if not files:
        raise FileNotFoundError(f"No '*combined*.csv' files found in {input_dir}")

    # Overlapping files can repeat a point. Collapse those repeats without
    # giving an overlapping file extra weight in the frequency-level fit.
    measurements: dict[tuple[float, float], list[float]] = defaultdict(list)
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                frequency = float(row["freq_ghz"])
                amplitude = float(row["amp"])
                power = float(row["power_dbm"])
                key = (frequency, amplitude)
                measurements[key].append(power)

    rows = [
        (frequency, amplitude, float(np.median(powers)))
        for (frequency, amplitude), powers in measurements.items()
    ]
    return rows, [path.name for path in files]


def build_calibration(
    rows: list[tuple[float, float, float]], fit_min_amp: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return frequency, full-scale power, and measured amplitude bounds."""
    if not 0.0 < fit_min_amp <= 1.0:
        raise ValueError("fit_min_amp must satisfy 0 < fit_min_amp <= 1")

    estimates: dict[float, list[float]] = defaultdict(list)
    amplitudes = []
    for frequency, amplitude, power in rows:
        if not (math.isfinite(frequency) and math.isfinite(amplitude) and math.isfinite(power)):
            raise ValueError("Calibration measurements must be finite")
        if not 0.0 < amplitude <= 1.0:
            raise ValueError(f"Invalid full-scale amplitude {amplitude}")
        amplitudes.append(amplitude)
        if amplitude >= fit_min_amp:
            estimates[frequency].append(power - 20.0 * math.log10(amplitude))

    frequencies = np.asarray(sorted(estimates), dtype=np.float64)
    if frequencies.size < 2:
        raise ValueError("Need high-amplitude measurements at at least two frequencies")
    full_scale_powers = np.asarray(
        [np.median(estimates[frequency]) for frequency in frequencies], dtype=np.float64
    )
    return frequencies, full_scale_powers, min(amplitudes), max(amplitudes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="directory containing combined CSV files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "daq" / "calibrations" / "power_calibration.npz",
    )
    parser.add_argument(
        "--fit-min-amp",
        type=float,
        default=0.4,
        help="minimum amplitude used to estimate full-scale power (default: 0.4)",
    )
    args = parser.parse_args()

    rows, source_files = read_measurements(args.input_dir)
    frequencies, full_scale_powers, amp_min, amp_max = build_calibration(rows, args.fit_min_amp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frequency_ghz=frequencies,
        power_dbm_at_amp_1=full_scale_powers,
        amp_min=amp_min,
        amp_max=amp_max,
        fit_min_amp=args.fit_min_amp,
        source_files=np.asarray(source_files),
        model="power_dbm_at_amp_1 + 20*log10(amp)",
    )
    print(f"Wrote {len(frequencies)} frequencies from {len(rows)} points to {args.output}")


if __name__ == "__main__":
    main()
