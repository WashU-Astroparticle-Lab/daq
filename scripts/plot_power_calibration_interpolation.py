"""Show how the power calibration behaves at frequencies it has no data for.

Top: the measured full-scale power ``P(1)`` and the verified amplitude floor at the calibrated
frequencies, the linear interpolation between them, and every frequency at which presto's
``recommended_dac_config`` changes the DAC configuration (see issue #69).  Below: nine off-grid
frequencies, each drawn between the two calibrated neighbours that define it, with the
neighbours' measurements for scale.  Solid is above the verified floor, dashed below (where the
conversions warn).

Usage::

    python scripts/plot_power_calibration_interpolation.py docs/power_calibration_interpolation.png
"""

import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from daq import calibrations as cal  # noqa: E402

OFF_GRID_GHZ = [2.5, 3.9, 4.8, 5.0, 6.1, 7.05, 7.15, 8.05, 8.9]


def dac_switch_points(f_lo: float, f_hi: float):
    """``[(freq_ghz, label)]`` where presto's recommended DAC config changes in ``[f_lo, f_hi]``."""
    try:
        from presto import utils
    except ImportError:  # presto 2.17.1 scan, 5 MHz steps, kept for machines without presto
        return [
            (2.400, "Mixed42/G8"),
            (3.000, "Mixed42/G10"),
            (4.270, "Mixed02/G6"),
            (4.900, "Mixed42/G8"),
            (5.605, "Mixed42/G10"),
            (7.005, "Mixed04/G8"),
            (7.105, "Mixed02/G6"),
            (8.000, "Mixed04/G10"),
        ]
    points, previous = [], None
    for f in np.arange(f_lo, f_hi + 0.0025, 0.005):
        config = utils.recommended_dac_config(f * 1e9)
        if previous is not None and config != previous:
            points.append((round(float(f), 3), f"{config[0].name}/{config[1].name}"))
        previous = config
    return points


def main() -> None:
    output = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO_ROOT / "docs" / "power_calibration_interpolation.png"
    )
    with np.load(REPO_ROOT / "daq" / "calibrations" / "power_calibration.npz") as asset:
        f_cal = np.asarray(asset["frequency_ghz"])
        p1 = np.asarray(asset["power_dbm_at_amp_1"])
        floor = np.asarray(asset["amp_floor"])
    measurements = defaultdict(list)
    for path in sorted((REPO_ROOT / "daq" / "calibrations" / "source_data").glob("*combined*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                measurements[float(row["freq_ghz"])].append(
                    (float(row["amp"]), float(row["power_dbm"]))
                )
    switches = dac_switch_points(float(f_cal[0]), float(f_cal[-1]))

    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    grid = fig.add_gridspec(5, 3, height_ratios=[2.2, 1.1, 1.6, 1.6, 1.6])
    ax_p1 = fig.add_subplot(grid[0, :])
    ax_floor = fig.add_subplot(grid[1, :], sharex=ax_p1)

    fine = np.linspace(f_cal[0], f_cal[-1], 2000)
    ax_p1.plot(
        fine,
        np.interp(fine, f_cal, p1),
        color="tab:blue",
        linewidth=1.5,
        label="np.interp between calibrated points",
    )
    ax_p1.plot(
        f_cal,
        p1,
        "o",
        color="tab:blue",
        markersize=6,
        label="measured P(1) at calibrated frequencies",
    )
    ax_p1.plot(
        OFF_GRID_GHZ,
        np.interp(OFF_GRID_GHZ, f_cal, p1),
        "s",
        color="black",
        markersize=7,
        markerfacecolor="white",
        label="off-grid frequencies drawn below",
    )
    for k, (f, label) in enumerate(switches):
        for ax in (ax_p1, ax_floor):
            ax.axvline(f, color="tab:red", linestyle="--", linewidth=1, alpha=0.7)
        ax_p1.text(
            f,
            -0.4 - 1.6 * (k % 2),
            f"{f:g} GHz → {label}",
            color="tab:red",
            fontsize=7.5,
            ha="center",
            va="top",
        )
    ax_p1.set_ylabel("Full-scale power P(1) (dBm)")
    ax_p1.set_title(
        "Full-scale power across the band: measured points, the interpolation used, and every DAC-configuration switch (red)"
    )
    ax_p1.grid(alpha=0.25)
    ax_p1.legend(fontsize=9, loc="lower left")
    ax_p1.set_ylim(-19, 0)

    floor_fine = np.array([cal.min_verified_amp(f) for f in fine])
    ax_floor.plot(
        fine,
        floor_fine,
        color="tab:green",
        linewidth=1.5,
        label="min_verified_amp (larger neighbour)",
    )
    ax_floor.plot(
        f_cal,
        floor,
        "v",
        color="tab:green",
        markersize=7,
        markerfacecolor="white",
        label="verified floor per calibrated frequency",
    )
    ax_floor.set_yscale("log")
    ax_floor.set_ylabel("Verified amplitude floor")
    ax_floor.set_xlabel("Frequency (GHz)")
    ax_floor.grid(alpha=0.25, which="both")
    ax_floor.legend(fontsize=9, loc="upper left")

    amps = np.logspace(-3, 0, 300)
    for k, f in enumerate(OFF_GRID_GHZ):
        ax = fig.add_subplot(grid[2 + k // 3, k % 3])
        i_hi = int(np.searchsorted(f_cal, f))
        i_lo = i_hi - 1
        f_lo, f_hi = float(f_cal[i_lo]), float(f_cal[i_hi])
        for f_n, color in ((f_lo, "tab:blue"), (f_hi, "tab:orange")):
            pts = np.array(sorted(measurements[f_n]))
            ax.scatter(
                pts[:, 0], pts[:, 1], s=9, color=color, alpha=0.55, label=f"{f_n:g} GHz measured"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", cal.CalibrationWarning)
                ax.plot(
                    amps, cal.amp_to_power_dbm(f_n, amps), color=color, linewidth=0.9, alpha=0.8
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", cal.CalibrationWarning)
            p = cal.amp_to_power_dbm(f, amps)
        fl = cal.min_verified_amp(f)
        above = amps >= fl
        ax.plot(amps[above], p[above], color="black", linewidth=2.2, label=f"{f:g} GHz calibration")
        ax.plot(amps[~above], p[~above], color="black", linewidth=1.6, linestyle=(0, (3, 2)))
        ax.plot(
            [fl],
            [cal.amp_to_power_dbm(f, fl)],
            "v",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="black",
            zorder=5,
        )
        inside = [s for s, _ in switches if f_lo < s < f_hi]
        title = (
            f"{f:g} GHz  (between {f_lo:g} and {f_hi:g} GHz, ΔP(1) = {p1[i_hi] - p1[i_lo]:+.2f} dB)"
        )
        if inside:
            title += (
                f"\nDAC switch inside the interval at {', '.join(f'{s:g}' for s in inside)} GHz"
            )
        ax.set_title(title, fontsize=9.5, color="tab:red" if inside else "black")
        ax.set_xscale("log")
        ax.set_ylim(-75, 0)
        ax.grid(alpha=0.25)
        if k % 3 == 0:
            ax.set_ylabel("Output power (dBm)")
        if k // 3 == 2:
            ax.set_xlabel("DAC amplitude (fraction of full scale)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles
            + [Line2D([], [], color="black", linestyle=(0, (3, 2)), label="below floor (warns)")],
            labels + ["below floor (warns)"],
            fontsize=7.5,
            loc="lower right",
        )

    fig.suptitle("Power calibration at frequencies with no calibration data", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
