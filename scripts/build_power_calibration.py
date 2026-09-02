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


def save_diagnostic_plot(
    rows: list[tuple[float, float, float]],
    frequencies: np.ndarray,
    full_scale_powers: np.ndarray,
    fit_min_amp: float,
    output: Path,
) -> None:
    """Plot measured powers, the voltage-law model, and all-point residuals."""
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
    residual_summary = (
        f"amp ≥ {fit_min_amp:g}: median |residual| = {np.median(fit_residual):.3f} dB\n"
        f"95th percentile = {np.percentile(fit_residual, 95):.3f} dB"
    )
    ax_residual.text(
        0.03,
        0.97,
        residual_summary,
        transform=ax_residual.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    figure.suptitle("Presto-8 output-power calibration diagnostic", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote diagnostic plot to {output}")


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
    parser.add_argument(
        "--diagnostic-plot",
        type=Path,
        help="optional diagnostic figure output (requires matplotlib)",
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
    if args.diagnostic_plot is not None:
        save_diagnostic_plot(
            rows,
            frequencies,
            full_scale_powers,
            args.fit_min_amp,
            args.diagnostic_plot,
        )


if __name__ == "__main__":
    main()
