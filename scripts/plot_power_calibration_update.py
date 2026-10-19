"""Plot corrected low-band curves and offset provenance (NumPy and matplotlib)."""

import csv
import importlib.util
from pathlib import Path
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "daq/calibrations/source_data/2026-10-16"


def read(name):
    with (SOURCE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    spec = importlib.util.spec_from_file_location("cal", ROOT / "daq/calibrations/__init__.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    warnings.simplefilter("ignore", cal.CalibrationWarning)
    corrected, anchors, checks, offsets = [
        read(n)
        for n in [
            "low_corrected_power.csv",
            "low_anchor.csv",
            "low_amplitude_checks.csv",
            "low_filter_offsets.csv",
        ]
    ]
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for f, ax in zip([0.1, 0.12, 0.5, 1.0, 2.1, 5.2], axes.flat):
        if f <= 2.1:
            rows = [r for r in corrected if float(r["frequency_mhz"]) == f * 1000]
            aa = np.array([float(r["amplitude"]) for r in rows])
            pp = np.array([float(r["median_peak_power_dbm"]) for r in rows])
            raw = np.array([float(r["original_filtered_power_dbm"]) for r in rows])
            ax.plot(aa, raw, color="#aaaaaa", linewidth=1.5)
            ax.scatter(aa, pp, color="#287eb0", s=13, zorder=3)
            bad = np.array([r["review_signal_or_timing"] == "True" for r in rows])
            ax.scatter(aa[bad], pp[bad], marker="x", color="#be5630", s=25, zorder=4)
            anchor = next(r for r in anchors if float(r["frequency_mhz"]) == f * 1000)
            ax.scatter(
                [0.1],
                [float(anchor["median_peak_power_dbm"])],
                marker="*",
                s=110,
                color="#22834a",
                zorder=6,
            )
            for row in checks:
                if float(row["frequency_mhz"]) == f * 1000:
                    ax.scatter(
                        float(row["amplitude"]),
                        float(row["median_peak_power_dbm"]),
                        marker="D",
                        s=45,
                        facecolors="none",
                        edgecolors="#b13d8d",
                        zorder=5,
                    )
            ax.set_title(f"{f:g} GHz — filter-offset corrected", fontsize=12)
        else:
            rows = [r for r in read("high_power.csv") if float(r["frequency_mhz"]) == f * 1000]
            ax.scatter(
                [float(r["amplitude"]) for r in rows],
                [float(r["median_peak_power_dbm"]) for r in rows],
                s=13,
                color="#287eb0",
            )
            for row in read("high_validation.csv"):
                if float(row["frequency_ghz"]) == f:
                    ax.scatter(
                        float(row["amplitude"]),
                        float(row["measured_power_dbm"]),
                        marker="D",
                        s=45,
                        facecolors="none",
                        edgecolors="#b13d8d",
                        zorder=5,
                    )
            ax.set_title(f"{f:g} GHz — direct unfiltered", fontsize=12)
        floor = cal.min_verified_amp(f)
        supported = np.geomspace(floor, 1, 250)
        ax.plot(supported, cal.amp_to_power_dbm(f, supported), color="#13577f", linewidth=1.1)
        tail = np.geomspace(0.001, floor, 30)
        ax.plot(tail, cal.amp_to_power_dbm(f, tail), "--", color="#7d8790", linewidth=1)
        ax.axvspan(0.001, floor, color="#eeeeee", zorder=0)
        ax.set(xscale="log", xlim=(0.00085, 1.15), ylim=(-72, 8))
        ax.set_xticks([0.001, 0.01, 0.1, 1], labels=["0.001", "0.01", "0.1", "1"])
        ax.grid(alpha=0.18)
    for ax in axes[-1]:
        ax.set_xlabel("Amplitude setting (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Analyzer-input power (dBm)")
    handles = [
        Line2D([], [], color="#aaaaaa", label="Original filtered curve"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#287eb0",
            label="Corrected points / direct high points",
        ),
        Line2D([], [], marker="*", linestyle="", color="#22834a", label="Direct unfiltered anchor"),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markerfacecolor="none",
            color="#b13d8d",
            label="Independent checks",
        ),
        Line2D(
            [], [], marker="x", linestyle="", color="#be5630", label="Original acquisition flag"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "0.1–2.1 GHz: calculated by filter offset", x=0.07, ha="left", fontsize=20, weight="bold"
    )
    fig.text(
        0.07,
        0.035,
        "Corrected curves estimate unfiltered power; they assume amplitude-independent loss and can include session/path drift.\nGray shading: outside source-quality support; dashed lines: warned extrapolation. No cable-loss correction.",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.81, bottom=0.14, hspace=0.27, wspace=0.13)
    fig.savefig(ROOT / "docs/power_calibration_october_diagnostic.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    f = np.array([float(r["frequency_mhz"]) / 1000 for r in offsets])
    d = np.array([float(r["filter_offset_db"]) for r in offsets])
    ax.plot(
        f,
        d,
        "o-",
        ms=3,
        color="#287eb0",
        label="Per-frequency correction applied to all 50 amplitudes",
    )
    ax.scatter(
        f[d < 0],
        d[d < 0],
        marker="x",
        s=60,
        color="#be5630",
        label="Negative offset retained",
        zorder=4,
    )
    ax.axhline(0, color="#777777", lw=1)
    ax.set(
        xlabel="Source frequency (GHz)",
        ylabel="Added power offset (dB)",
        xlim=(0.05, 2.15),
        title="Correction offsets: unfiltered anchor minus original filtered curve at A=0.1",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=10)
    fig.text(
        0.1,
        0.035,
        "Offsets include possible filter loss, path changes and session drift. They are not isolated insertion-loss measurements.\nNegative offsets at 140, 880 and 960 MHz are flagged; the correction is not forced positive.",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.22, top=0.9)
    fig.savefig(ROOT / "docs/low_band_filter_offsets.png", dpi=170)
    plt.close(fig)
    print("Updated corrected-curve and offset figures.")


if __name__ == "__main__":
    main()
