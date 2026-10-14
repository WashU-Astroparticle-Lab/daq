"""Validate both fridge-loss sources and the red screenshot trace digitization.

Run ``python scripts/plot_nominal_fridge_loss.py`` in the presto environment.
Writes ``docs/nominal_fridge_loss.png``. Public function calls use Hz.
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from digitize_fridge_loss import BOTTOM, LEFT, RIGHT, SOURCE, TOP, digitize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibrations", ROOT / "daq/calibrations/__init__.py")
calibrations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrations)
nominal_fridge_loss = calibrations.nominal_fridge_loss

input_data = np.loadtxt(ROOT / "daq/calibrations/source_data/fridge_loss_to_mc.csv", delimiter=",")
loopback_data = np.loadtxt(
    ROOT / "daq/calibrations/source_data/Empty_Line_port_2_-_2026-06-15.s2p",
    comments=["#", "!"],
)
digitized_hz, digitized_db, red_x, red_y = digitize()
assert np.array_equal(input_data[:, 0], digitized_hz)
assert np.max(np.abs(input_data[:, 1] - digitized_db)) <= 5e-7

for include_output, frequency_hz, loss_db in (
    (False, input_data[:, 0], input_data[:, 1]),
    (True, loopback_data[:, 0], -loopback_data[:, 3]),
):
    raw = nominal_fridge_loss(frequency_hz, smooth=False, include_output=include_output)
    midpoint_hz = (frequency_hz[:-1] + frequency_hz[1:]) / 2
    midpoint_db = nominal_fridge_loss(midpoint_hz, smooth=False, include_output=include_output)
    assert np.max(np.abs(raw - loss_db)) < 1e-10
    assert np.max(np.abs(midpoint_db - (loss_db[:-1] + loss_db[1:]) / 2)) < 1e-10
    print(f"Raw source and midpoint recovery verified: {include_output=}")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(13, 11), layout="constrained")
grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.15, 1])
full = fig.add_subplot(grid[0, :])
overlay = fig.add_subplot(grid[1, :])
zoom = fig.add_subplot(grid[2, 0])
adjustment = fig.add_subplot(grid[2, 1])
fig.suptitle("Nominal fridge loss: input line to MC by default", fontsize=19)

frequency_hz, input_db = input_data.T
dense_hz = np.linspace(frequency_hz[0], frequency_hz[-1], 20001)
full.plot(
    dense_hz / 1e9,
    nominal_fridge_loss(dense_hz),
    color="#c72736",
    lw=2,
    label="Default: input to MC, measured at room temperature",
)
full.scatter(
    frequency_hz / 1e9, input_db, s=9, color="#c72736", alpha=0.4, label="Digitized red trace"
)
full.plot(
    dense_hz / 1e9,
    nominal_fridge_loss(dense_hz, include_output=True),
    color="#126a96",
    lw=1.8,
    label="include_output=True: previous cold loopback (input + output)",
)
full.plot(loopback_data[:, 0] / 1e9, -loopback_data[:, 3], color="#126a96", lw=0.5, alpha=0.25)
full.set(xlabel="Frequency [GHz]", ylabel="Positive loss [dB]", xlim=(0, 10))
full.legend(fontsize=10, loc="upper left")

# Crop only the displayed viewport, retaining exact source-image coordinates. The
# recovered 201-point curve is drawn on the actual screenshot for visual validation.
overlay.imshow(plt.imread(SOURCE), origin="upper")
sample_x = LEFT + (frequency_hz - 1e5) / (1e10 - 1e5) * (RIGHT - LEFT)
sample_y = TOP + (input_db - 45) / 50 * (BOTTOM - TOP)
overlay.plot(
    sample_x,
    sample_y,
    color="#56dcec",
    lw=0.6,
    marker="o",
    markersize=2,
    markerfacecolor="none",
    markeredgewidth=0.5,
    label="Extracted red trace (201 points)",
)
overlay.set(
    xlim=(LEFT - 5, RIGHT + 5),
    ylim=(875, 410),
    aspect="auto",
    title="Digitization check on the original screenshot (red and yellow retained)",
)
overlay.set_xticks(LEFT + np.arange(11) / 10 * (RIGHT - LEFT), labels=np.arange(11))
loss_ticks = np.arange(60, 77, 5)
overlay.set_yticks(TOP + (loss_ticks - 45) / 50 * (BOTTOM - TOP), labels=-loss_ticks)
overlay.set(xlabel="Frequency [GHz]", ylabel="Screenshot S21 [dB]")
overlay.legend(loc="upper right", fontsize=9)

zoom_hz = np.linspace(2.5e9, 4.5e9, 2001)
zoom.plot(
    zoom_hz / 1e9, nominal_fridge_loss(zoom_hz), color="#c72736", lw=2, label="Smoothed default"
)
zoom.plot(
    zoom_hz / 1e9,
    nominal_fridge_loss(zoom_hz, smooth=False),
    color="#5b5555",
    lw=1,
    linestyle="--",
    label="smooth=False",
)
example_db = nominal_fridge_loss(2.8e9)
zoom.scatter([2.8], [example_db], color="#c72736", s=25, zorder=3)
zoom.annotate(
    f"2.8 GHz: {example_db:.2f} dB",
    (2.8, example_db),
    xytext=(2.6, 68.8),
    arrowprops={"arrowstyle": "-", "color": "#c72736"},
)
zoom.set(
    title="Input-to-MC loss near the operating band",
    xlabel="Frequency [GHz]",
    ylabel="Positive loss [dB]",
    xlim=(2.5, 4.5),
)
zoom.legend(loc="lower right", fontsize=9)

input_adjustment = nominal_fridge_loss(frequency_hz) - input_db
adjustment.plot(frequency_hz / 1e9, input_adjustment, color="#c72736", lw=1)
adjustment.axhline(0, color="#5b5555", lw=0.6)
adjustment.set(
    title="Input-trace smoothing adjustment (~200 MHz)",
    xlabel="Frequency [GHz]",
    ylabel="Smoothed − digitized loss [dB]",
)
adjustment.text(
    0.04,
    0.96,
    f"RMS change: {np.sqrt(np.mean(input_adjustment**2)):.3f} dB\n"
    f"Maximum change: {np.max(np.abs(input_adjustment)):.3f} dB",
    transform=adjustment.transAxes,
    va="top",
    fontsize=10,
)
for ax in (full, zoom, adjustment):
    ax.grid(alpha=0.2)
fig.supxlabel(
    "Red-trace calibration is approximate: ~0.04 dB/pixel; short occlusions are interpolated.\n"
    "Room-temperature input and cold loopback are separate measurements; their difference is not an output-line calibration.",
    fontsize=10,
)
output = ROOT / "docs/nominal_fridge_loss.png"
fig.savefig(output, dpi=180)
print(f"Saved {output}")
print(
    f"At 2.8 GHz: default {example_db:.6f} dB; loopback {nominal_fridge_loss(2.8e9, include_output=True):.6f} dB"
)
