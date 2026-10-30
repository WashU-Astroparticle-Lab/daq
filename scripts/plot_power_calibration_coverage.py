"""Compare current interpolation with a previous NPZ and plot all reopened intervals.

Run with --baseline pointing to the schema-4 asset from commit 19998f7. The plots
evaluate the public conversion functions; they are diagnostics, not new measurements.
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#176b89"
ORANGE = "#d96817"
GRAY = "#c3c8cc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("cal", ROOT / "daq/calibrations/__init__.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    warnings.simplefilter("ignore", cal.CalibrationWarning)
    with np.load(args.baseline, allow_pickle=False) as old, np.load(cal._DATA_PATH) as current:
        # Only coverage and provenance may change in this comparison.
        changed_metadata = {
            "interpolation_allowed",
            "calibration_id",
            "interpolation_policy_json",
            "source_sha256",
        }
        assert set(old.files) == set(current.files)
        for key in current.files:
            if key not in changed_metadata:
                assert np.array_equal(old[key], current[key]), f"Calibration data changed: {key}"
        f = current["frequency_ghz"].copy()
        p = current["power_dbm_at_amp_1"].copy()
        previous_allowed = old["interpolation_allowed"].copy()
        allowed = current["interpolation_allowed"].copy()
    assert np.all(allowed), "Expected continuous frequency coverage"
    reopened = np.flatnonzero(~previous_allowed & allowed)
    assert len(reopened), "No reopened intervals to plot"
    target, requested = 6.9968, -32.0
    target_power = cal.amp_to_power_dbm(target, 1.0)
    target_amp = cal.power_dbm_to_amp(target, requested)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    node_amp = np.array([cal.power_dbm_to_amp(float(x), requested) for x in f])
    for col, limits in enumerate([(0.1, 9.0), (6.9, 7.2)]):
        for row, node_values in enumerate([p, node_amp]):
            ax = axes[row, col]
            for i in range(len(f) - 1):
                if f[i + 1] < limits[0] or f[i] > limits[1]:
                    continue
                x = np.linspace(f[i], f[i + 1], 31)
                y = np.array(
                    [
                        (
                            cal.amp_to_power_dbm(float(v), 1.0)
                            if row == 0
                            else cal.power_dbm_to_amp(float(v), requested)
                        )
                        for v in x
                    ]
                )
                if previous_allowed[i]:
                    ax.plot(x, y, color=GRAY, lw=4, zorder=1)
                else:
                    ax.axvspan(f[i], f[i + 1], color=ORANGE, alpha=0.10, lw=0)
                ax.plot(x, y, color=BLUE if previous_allowed[i] else ORANGE, lw=1.6)
            visible = (f >= limits[0]) & (f <= limits[1])
            ax.scatter(
                f[visible],
                node_values[visible],
                s=12 if col == 0 else 32,
                color=BLUE,
                edgecolors="white",
                linewidths=0.4,
                zorder=4,
            )
            ax.set(xlim=limits, xlabel="Frequency (GHz)")
            ax.grid(alpha=0.18)
            if row == 1:
                ax.set_yscale("log")
            if col == 1:
                y = target_power if row == 0 else target_amp
                ax.axvline(target, color="#333333", lw=0.8, ls=":")
                ax.scatter([target], [y], marker="*", s=160, color="#333333", zorder=5)
                label = f"6.9968 GHz: {y:.4f} dBm" if row == 0 else f"6.9968 GHz: amp = {y:.8f}"
                ax.annotate(
                    label,
                    (target, y),
                    xytext=(10, -30 if row == 0 else 22),
                    textcoords="offset points",
                    fontsize=10,
                )
    axes[0, 0].set_title("Full calibrated band")
    axes[0, 1].set_title("Reported frequency and nearby DAC-switch regions")
    axes[0, 0].set_ylabel("Full-scale analyzer-input power (dBm)")
    axes[1, 0].set_ylabel("Amplitude for −32 dBm (fraction of full scale)")
    handles = [
        Line2D(
            [], [], color=BLUE, marker="o", label="Unchanged calibration nodes / prior coverage"
        ),
        Line2D([], [], color=ORANGE, label="Newly allowed interpolation"),
        Patch(facecolor=ORANGE, alpha=0.10, label="Previously rejected interval"),
    ]
    fig.suptitle(
        f"Power conversion now covers all 250 intervals ({len(reopened)} reopened)",
        fontsize=18,
        weight="bold",
        y=0.98,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.07,
        0.025,
        "Curves are API evaluations. Orange segments had no returned value before this change.\n"
        "Calibration nodes are unchanged; interpolation estimates transitions and does not resolve hardware steps.",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.85, hspace=0.3, wspace=0.2)
    fig.savefig(ROOT / "docs/power_calibration_coverage.png", dpi=170)
    plt.close(fig)

    rows = (len(reopened) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 2.55 * rows), squeeze=False)
    for ax, i in zip(axes.flat, reopened):
        x = np.linspace(f[i], f[i + 1], 101)
        y = [cal.amp_to_power_dbm(float(v), 1.0) for v in x]
        ax.axvspan(f[i], f[i + 1], color=ORANGE, alpha=0.09)
        ax.plot(x, y, color=ORANGE, lw=2)
        ax.scatter(f[i : i + 2], p[i : i + 2], color=BLUE, s=30, zorder=4)
        ax.set_title(f"{f[i]:g}–{f[i+1]:g} GHz  |  ΔP₁ = {p[i+1]-p[i]:+.3f} dB", fontsize=10)
        ax.set_xticks([f[i], (f[i] + f[i + 1]) / 2, f[i + 1]])
        ax.ticklabel_format(useOffset=False)
        ax.margins(x=0.08, y=0.2)
        ax.grid(alpha=0.2)
        if f[i] < target < f[i + 1]:
            ax.scatter([target], [target_power], marker="*", s=100, color="#333333", zorder=5)
            ax.annotate(
                "6.9968 GHz",
                (target, target_power),
                xytext=(-8, 18),
                textcoords="offset points",
                ha="right",
                fontsize=9,
            )
    for ax in list(axes.flat)[len(reopened) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Every previously rejected interval now interpolates", fontsize=19, weight="bold", y=0.985
    )
    fig.text(
        0.5,
        0.952,
        "Blue: unchanged endpoint calibration levels   •   Orange: new API result   •   ΔP₁: right minus left endpoint",
        ha="center",
        fontsize=11,
    )
    fig.supxlabel("Frequency (GHz)", y=0.045)
    fig.supylabel("Full-scale analyzer-input power (dBm)", x=0.015)
    fig.text(
        0.06,
        0.012,
        "Each panel uses its own y scale. These lines estimate transitions between existing nodes; no new hardware validation is implied.",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.09, hspace=0.6, wspace=0.3)
    fig.savefig(ROOT / "docs/power_calibration_reopened_intervals.png", dpi=170)
    plt.close(fig)
    print(f"Reopened {len(reopened)} intervals; all numerical calibration arrays unchanged.")
    print(f"power_dbm_to_amp({target}, {requested}) = {target_amp:.12f}")
    print(f"Full-scale power at {target} GHz = {target_power:.12f} dBm")


if __name__ == "__main__":
    main()
