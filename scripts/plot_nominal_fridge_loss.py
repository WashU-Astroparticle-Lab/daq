"""Compare nominal_fridge_loss against the original port-2 S21 measurement.

Run with ``python scripts/plot_nominal_fridge_loss.py`` from the repository root.
The output is ``docs/nominal_fridge_loss.png``. All calls to the function use Hz.
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibrations", ROOT / "daq/calibrations/__init__.py")
calibrations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrations)
nominal_fridge_loss = calibrations.nominal_fridge_loss

# Independently read the original file; columns are explicitly labelled in its header.
source = ROOT / "daq/calibrations/source_data/Empty_Line_port_2_-_2026-06-15.s2p"
data = np.loadtxt(source, comments=["#", "!"])
frequency_hz, s21_db = data[:, 0], data[:, 3]
dense_hz = np.linspace(frequency_hz[0], frequency_hz[-1], 20001)
residual_db = nominal_fridge_loss(frequency_hz) + s21_db
midpoints_hz = (frequency_hz[:-1] + frequency_hz[1:]) / 2
midpoint_error = nominal_fridge_loss(midpoints_hz) + (s21_db[:-1] + s21_db[1:]) / 2
assert np.max(np.abs(residual_db)) < 1e-10
assert np.max(np.abs(midpoint_error)) < 1e-10

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(12, 9), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
full = fig.add_subplot(grid[0, :])
zoom = fig.add_subplot(grid[1, 0])
residual = fig.add_subplot(grid[1, 1])
fig.suptitle("Nominal fridge loss from the measured port-2 transmission", fontsize=18)

full.plot(
    dense_hz,
    nominal_fridge_loss(dense_hz),
    color="#126a96",
    lw=1.5,
    label="nominal_fridge_loss(frequency_hz)",
)
full.scatter(
    frequency_hz,
    -s21_db,
    s=10,
    facecolors="none",
    edgecolors="#db7b23",
    linewidths=0.6,
    label=f"Original file: −S21 (all {len(frequency_hz):,} points)",
    zorder=3,
)
full.set(title="Full measured band: 100 kHz to 10 GHz", ylabel="Loss [dB]", xlabel="Frequency [Hz]")
full.legend(loc="upper left")

selected = (frequency_hz >= 2.77e9) & (frequency_hz <= 2.84e9)
zoom_hz = np.linspace(2.78e9, 2.83e9, 1001)
zoom.plot(
    zoom_hz, nominal_fridge_loss(zoom_hz), color="#126a96", lw=2, label="Linear interpolation in dB"
)
zoom.scatter(
    frequency_hz[selected],
    -s21_db[selected],
    s=55,
    facecolors="none",
    edgecolors="#db7b23",
    linewidths=1.7,
    zorder=3,
    label="Measured −S21",
)
example_hz = 2.8e9
example_db = nominal_fridge_loss(example_hz)
zoom.scatter([example_hz], [example_db], color="#934ba2", s=50, zorder=4)
zoom.annotate(
    f"f = 2.8e9 Hz\nloss = {example_db:.6f} dB",
    (example_hz, example_db),
    xytext=(0.05, 0.85),
    textcoords="axes fraction",
    fontsize=10,
    arrowprops={"arrowstyle": "-", "color": "#934ba2"},
)
zoom.set(
    title="Zoom: measured points and interpolation",
    xlabel="Frequency [Hz]",
    ylabel="Loss [dB]",
    xlim=(2.78e9, 2.83e9),
)
zoom.legend(loc="lower right", fontsize=9)

residual.plot(frequency_hz, residual_db, color="#126a96", lw=1.5)
residual.set(
    title="Agreement at every measured frequency",
    xlabel="Frequency [Hz]",
    ylabel="Function − measured loss [dB]",
    ylim=(-1e-10, 1e-10),
)
residual.text(
    0.05,
    0.94,
    f"Max measured-point error: {np.max(np.abs(residual_db)):.1e} dB\n"
    f"Max midpoint error: {np.max(np.abs(midpoint_error)):.1e} dB",
    transform=residual.transAxes,
    va="top",
    fontsize=10,
)
for ax in (full, zoom, residual):
    ax.grid(alpha=0.2)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=False)
fig.supxlabel(
    "Source: Empty_Line_port_2_-_2026-06-15.s2p • loss = −S21_dB\n"
    "Validates file conversion and interpolation; no separation of input/output line losses.",
    fontsize=10,
)
output = ROOT / "docs/nominal_fridge_loss.png"
fig.savefig(output, dpi=180)
print(f"Saved {output}")
print(f"Measured-point maximum error: {np.max(np.abs(residual_db)):.3g} dB")
print(f"Midpoint maximum error: {np.max(np.abs(midpoint_error)):.3g} dB")
