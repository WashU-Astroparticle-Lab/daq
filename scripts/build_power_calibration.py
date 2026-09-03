"""Build the packaged DAC power calibration from spectrum-analyzer CSV files.

The source measurements (``daq/calibrations/source_data/*combined*.csv``, one row per
``freq_ghz``, ``amp``, ``power_dbm``) were taken on 2026-03-04/05 with a ``presto.lockin``
tone on output port 1 (``DacMode.Mixed`` with the automatic DAC configuration, DAC current
40.5 mA, ``df`` 2 kHz) read by a Signal Hound Spike channel-power measurement.

At each frequency the high-amplitude points estimate the full-scale power ``P(1)`` and the
amplitude dependence is kept analytic as ``20 * log10(amp)``, because ``amp`` is a voltage
fraction.  The builder also records, per frequency, the smallest amplitude at which the
measurements still follow that law: below it the analyzer's channel power flattens onto its
own noise while the DAC keeps scaling, so the library warns there instead of trusting the
data.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "daq" / "calibrations" / "source_data"
DEFAULT_OUTPUT = REPO_ROOT / "daq" / "calibrations" / "power_calibration.npz"

Row = Tuple[float, float, float]

MEASUREMENT_NOTES = (
    "Presto-8 output port 1, presto.lockin tone, DacMode.Mixed with automatic DAC "
    "configuration, DAC current 40.5 mA, df 2 kHz; Signal Hound Spike channel power; "
    "2026-03-04/05."
)


def read_measurements(input_dir: Path) -> Tuple[List[Row], List[str]]:
    """Read and de-duplicate all combined calibration CSVs in *input_dir*.

    :param input_dir: Directory holding ``*combined*.csv`` files with ``freq_ghz``, ``amp`` and
        ``power_dbm`` columns.
    :returns: ``(rows, file_names)`` with one ``(freq_ghz, amp, power_dbm)`` row per distinct
        ``(freq_ghz, amp)``; repeated points across overlapping files collapse to their median.
    :raises FileNotFoundError: If the directory holds no matching file.
    """
    files = sorted(input_dir.glob("*combined*.csv"))
    if not files:
        raise FileNotFoundError(f"No '*combined*.csv' files found in {input_dir}")

    # Overlapping files can repeat a point. Collapse those repeats without
    # giving an overlapping file extra weight in the frequency-level fit.
    measurements: Dict[Tuple[float, float], List[float]] = defaultdict(list)
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                frequency = float(row["freq_ghz"])
                amplitude = float(row["amp"])
                power = float(row["power_dbm"])
                measurements[(frequency, amplitude)].append(power)

    rows = [
        (frequency, amplitude, float(np.median(powers)))
        for (frequency, amplitude), powers in measurements.items()
    ]
    return rows, [path.name for path in files]


def verified_floor(
    amplitudes: np.ndarray, powers: np.ndarray, full_scale_power: float, tolerance_db: float
) -> float:
    """Smallest amplitude at which the measurements still follow ``P(1) + 20 log10(amp)``.

    The analyzer floor pushes measured power *above* the law, so only positive residuals count;
    an isolated reading below the law (a marker that missed the tone) does not move the floor.

    :param amplitudes: Measured amplitudes at one frequency, any order.
    :param powers: Measured powers in dBm, matching *amplitudes*.
    :param full_scale_power: The ``P(1)`` estimate for that frequency, in dBm.
    :param tolerance_db: Largest positive residual still counted as following the law.
    :returns: The amplitude just above the last point that exceeds the tolerance, or the
        smallest measured amplitude if none does.
    :raises ValueError: If even the largest amplitude exceeds the tolerance.
    """
    order = np.argsort(amplitudes)
    amps = amplitudes[order]
    residual = powers[order] - (full_scale_power + 20.0 * np.log10(amps))
    above = np.flatnonzero(residual > tolerance_db)
    if above.size == 0:
        return float(amps[0])
    last = int(above.max())
    if last + 1 >= amps.size:
        raise ValueError("No amplitude follows the voltage law within the floor tolerance")
    return float(amps[last + 1])


def build_calibration(
    rows: List[Row], fit_min_amp: float, floor_tolerance_db: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the per-frequency full-scale power and verified amplitude floor.

    :param rows: ``(freq_ghz, amp, power_dbm)`` measurements from :func:`read_measurements`.
    :param fit_min_amp: Only points at or above this amplitude enter the ``P(1)`` estimate
        (the median of ``power - 20 log10(amp)`` over them).
    :param floor_tolerance_db: Passed to :func:`verified_floor`.
    :returns: ``(frequencies, full_scale_powers, amp_floors, noise_floors)``, one entry per
        frequency with high-amplitude data; ``noise_floors`` is the median measured power well
        below the amplitude floor (``nan`` where no such point exists), recorded for the mute
        test that would attribute the floor to the analyzer or to the Presto.
    :raises ValueError: On non-finite or out-of-range input, or fewer than two usable
        frequencies.
    """
    if not 0.0 < fit_min_amp <= 1.0:
        raise ValueError("fit_min_amp must satisfy 0 < fit_min_amp <= 1")
    if not floor_tolerance_db > 0.0:
        raise ValueError("floor_tolerance_db must be positive")

    by_frequency: Dict[float, List[Tuple[float, float]]] = defaultdict(list)
    for frequency, amplitude, power in rows:
        if not (math.isfinite(frequency) and math.isfinite(amplitude) and math.isfinite(power)):
            raise ValueError("Calibration measurements must be finite")
        if not 0.0 < amplitude <= 1.0:
            raise ValueError(f"Invalid full-scale amplitude {amplitude}")
        by_frequency[frequency].append((amplitude, power))

    frequencies: List[float] = []
    full_scale_powers: List[float] = []
    amp_floors: List[float] = []
    noise_floors: List[float] = []
    for frequency in sorted(by_frequency):
        data = np.asarray(by_frequency[frequency], dtype=np.float64)
        amplitudes, powers = data[:, 0], data[:, 1]
        fit = amplitudes >= fit_min_amp
        if not np.any(fit):
            continue
        full_scale = float(np.median(powers[fit] - 20.0 * np.log10(amplitudes[fit])))
        floor = verified_floor(amplitudes, powers, full_scale, floor_tolerance_db)
        deep = amplitudes < floor / 3.0
        noise = float(np.median(powers[deep])) if np.any(deep) else math.nan
        frequencies.append(frequency)
        full_scale_powers.append(full_scale)
        amp_floors.append(floor)
        noise_floors.append(noise)

    if len(frequencies) < 2:
        raise ValueError("Need high-amplitude measurements at at least two frequencies")
    return (
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(full_scale_powers, dtype=np.float64),
        np.asarray(amp_floors, dtype=np.float64),
        np.asarray(noise_floors, dtype=np.float64),
    )


def save_diagnostic_plot(
    rows: List[Row],
    frequencies: np.ndarray,
    full_scale_powers: np.ndarray,
    amp_floors: np.ndarray,
    fit_min_amp: float,
    floor_tolerance_db: float,
    output: Path,
) -> None:
    """Plot measured powers, the voltage-law model, the verified floors and the residuals.

    :param rows: The measurements, as returned by :func:`read_measurements`.
    :param frequencies: Calibrated frequencies in GHz.
    :param full_scale_powers: ``P(1)`` per frequency, in dBm.
    :param amp_floors: Verified amplitude floor per frequency.
    :param fit_min_amp: The amplitude cutoff used for the ``P(1)`` estimate.
    :param floor_tolerance_db: The residual tolerance that defined the floors.
    :param output: Where to write the figure.
    :raises RuntimeError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:
        raise RuntimeError("Diagnostic plotting requires matplotlib") from exc

    data = np.asarray(rows, dtype=np.float64)
    measured_frequency = data[:, 0]
    measured_amp = data[:, 1]
    measured_power = data[:, 2]
    modeled_power = np.interp(measured_frequency, frequencies, full_scale_powers) + 20.0 * np.log10(
        measured_amp
    )
    residual = measured_power - modeled_power
    floor_at_point = np.interp(measured_frequency, frequencies, amp_floors)

    cmap = plt.colormaps["viridis"]
    norm = Normalize(vmin=float(frequencies[0]), vmax=float(frequencies[-1]))
    figure, (ax_power, ax_residual) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    selected_indices = np.unique(np.rint(np.linspace(0, len(frequencies) - 1, 6)).astype(int))
    near_28 = np.flatnonzero(np.isclose(frequencies, 2.8))
    if near_28.size and near_28[0] not in selected_indices:
        selected_indices[1] = near_28[0]
        selected_indices = np.unique(np.sort(selected_indices))

    for index in selected_indices:
        frequency = frequencies[index]
        mask = np.isclose(measured_frequency, frequency)
        order = np.argsort(measured_amp[mask])
        amps = measured_amp[mask][order]
        powers = measured_power[mask][order]
        color = cmap(norm(frequency))
        ax_power.scatter(amps, powers, s=16, color=color, alpha=0.8)
        ax_power.plot(
            amps,
            full_scale_powers[index] + 20.0 * np.log10(amps),
            color=color,
            linewidth=1.7,
            label=f"{frequency:g} GHz",
        )
        ax_power.plot(
            [amp_floors[index]],
            [full_scale_powers[index] + 20.0 * np.log10(amp_floors[index])],
            marker="v",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor=color,
            linestyle="none",
            zorder=5,
        )

    ax_power.plot(
        [],
        [],
        marker="v",
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="none",
        label="verified floor",
    )
    ax_power.axvspan(
        measured_amp.min(),
        fit_min_amp,
        color="0.85",
        alpha=0.35,
        label="below fit region",
        zorder=0,
    )
    ax_power.set_xscale("log")
    ax_power.set_xlabel("DAC amplitude (fraction of full scale)")
    ax_power.set_ylabel("Measured output power (dBm)")
    ax_power.set_title("Measurements and fitted $20\\log_{10}(amp)$ law")
    ax_power.grid(alpha=0.25)
    ax_power.legend(fontsize=8, ncol=2)

    points = ax_residual.scatter(
        measured_amp,
        residual,
        c=measured_frequency,
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=0.65,
        edgecolors="none",
    )
    ax_residual.axhline(0.0, color="black", linewidth=1.0)
    ax_residual.axhline(
        floor_tolerance_db, color="tab:gray", linestyle=":", linewidth=1.0, label="floor tolerance"
    )
    ax_residual.axvline(
        fit_min_amp, color="tab:red", linestyle="--", linewidth=1.2, label="fit cutoff"
    )
    ax_residual.set_xscale("log")
    ax_residual.set_xlabel("DAC amplitude (fraction of full scale)")
    ax_residual.set_ylabel("Measured − modeled power (dB)")
    ax_residual.set_title("Residuals for all frequencies")
    ax_residual.grid(alpha=0.25)
    ax_residual.legend(fontsize=8)
    figure.colorbar(points, ax=ax_residual, label="Frequency (GHz)")

    fit_residual = np.abs(residual[measured_amp >= fit_min_amp])
    verified_residual = np.abs(residual[measured_amp >= floor_at_point])
    residual_summary = (
        f"amp ≥ {fit_min_amp:g}: median |residual| = {np.median(fit_residual):.3f} dB, "
        f"95th percentile = {np.percentile(fit_residual, 95):.3f} dB\n"
        f"above verified floor: median |residual| = {np.median(verified_residual):.3f} dB, "
        f"95th percentile = {np.percentile(verified_residual, 95):.3f} dB"
    )
    ax_residual.text(
        0.03,
        0.97,
        residual_summary,
        transform=ax_residual.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    figure.suptitle("Presto-8 output-power calibration diagnostic", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote diagnostic plot to {output}")


def main() -> None:
    """Command-line entry point: read the CSVs, build the asset and optionally plot it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help=f"directory containing combined CSV files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fit-min-amp",
        type=float,
        default=0.4,
        help="minimum amplitude used to estimate full-scale power (default: 0.4)",
    )
    parser.add_argument(
        "--floor-tolerance-db",
        type=float,
        default=0.5,
        help="largest measured-minus-model excess still counted as following the law "
        "(default: 0.5 dB)",
    )
    parser.add_argument(
        "--diagnostic-plot",
        type=Path,
        help="optional diagnostic figure output (requires matplotlib)",
    )
    args = parser.parse_args()

    rows, source_files = read_measurements(args.input_dir)
    frequencies, full_scale_powers, amp_floors, noise_floors = build_calibration(
        rows, args.fit_min_amp, args.floor_tolerance_db
    )
    measured_amps = [amplitude for _, amplitude, _ in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frequency_ghz=frequencies,
        power_dbm_at_amp_1=full_scale_powers,
        amp_floor=amp_floors,
        floor_tolerance_db=args.floor_tolerance_db,
        noise_floor_dbm=noise_floors,
        fit_min_amp=args.fit_min_amp,
        measured_amp_min=min(measured_amps),
        measured_amp_max=max(measured_amps),
        source_files=np.asarray(source_files),
        model="power_dbm_at_amp_1 + 20*log10(|amp|)",
        measurement=MEASUREMENT_NOTES,
    )
    print(f"Wrote {len(frequencies)} frequencies from {len(rows)} points to {args.output}")
    print(
        "Verified amplitude floor: "
        f"{amp_floors.min():.4f} to {amp_floors.max():.4f} "
        f"(tolerance {args.floor_tolerance_db:g} dB)"
    )
    if args.diagnostic_plot is not None:
        save_diagnostic_plot(
            rows,
            frequencies,
            full_scale_powers,
            amp_floors,
            args.fit_min_amp,
            args.floor_tolerance_db,
            args.diagnostic_plot,
        )


if __name__ == "__main__":
    main()
