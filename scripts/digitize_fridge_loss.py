"""Reproduce the input-line calibration from the supplied red screenshot trace.

Run ``python scripts/digitize_fridge_loss.py`` in the presto environment. Requires
Pillow and numpy. Coordinates refer to the committed 2558 x 1397 PNG, converted
losslessly from ``Safari 2026-10-13 18.59.15.tiff``. The user identifies the red
trace as the room-temperature input-line measurement down to the mixing chamber.
The screenshot's date display is ``6/1/2026 18:01``; its date convention is unknown.
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/fridge_loss_to_mc_source.png"
OUTPUT = ROOT / "daq/calibrations/source_data/fridge_loss_to_mc.csv"

# Plot borders: x=75..2398, y=75..1338. Displayed axes are linear frequency,
# Start 100 kHz / Stop 10 GHz, and S21=-45..-95 dB (5 dB per division).
LEFT, RIGHT, TOP, BOTTOM = 75, 2398, 75, 1338
START_HZ, STOP_HZ = 1e5, 1e10
N_POINTS = 201  # Point count displayed in the analyzer's bottom status bar.


def digitize():
    """Return frequency, positive loss, and the visible red pixel centerline."""
    with Image.open(SOURCE) as source:
        if source.size != (2558, 1397):
            raise ValueError("Digitization coordinates require the original image dimensions.")
        rgb = np.asarray(source.convert("RGB"))
    plot = rgb[TOP : BOTTOM + 1, LEFT : RIGHT + 1]
    # Red-only selection excludes the yellow trace, grid, labels and right-side marker.
    red = (plot[:, :, 0] > 150) & (plot[:, :, 1] < 80) & (plot[:, :, 2] < 80)
    columns = np.flatnonzero(red.any(axis=0))
    x_pixels = LEFT + columns
    y_pixels = TOP + np.array([np.median(np.flatnonzero(red[:, x])) for x in columns])
    if x_pixels[0] != LEFT or x_pixels[-1] != RIGHT:
        raise ValueError("The red trace must cover both frequency endpoints.")
    # Yellow sometimes occludes red; bridge only short missing spans by interpolation.
    # The longest gap in this source has 16 missing columns, about 69 MHz.
    if np.max(np.diff(x_pixels)) > 17:
        raise ValueError("Unexpectedly large gap in the red trace.")
    sample_x = np.linspace(LEFT, RIGHT, N_POINTS)
    sample_y = np.interp(sample_x, x_pixels, y_pixels)
    frequency_hz = np.linspace(START_HZ, STOP_HZ, N_POINTS)
    loss_db = 45 + (sample_y - TOP) * 50 / (BOTTOM - TOP)
    return frequency_hz, loss_db, x_pixels, y_pixels


if __name__ == "__main__":
    frequency_hz, loss_db, x_pixels, _ = digitize()
    np.savetxt(
        OUTPUT,
        np.column_stack((frequency_hz, loss_db)),
        delimiter=",",
        fmt=["%.1f", "%.6f"],
        header=(
            "Room-temperature input-line loss down to MC; red screenshot trace.\n"
            "Source: docs/fridge_loss_to_mc_source.png (user-supplied screenshot).\n"
            "Reproduce: python scripts/digitize_fridge_loss.py\n"
            "201 uniformly spaced points; linear x=75..2398 -> 100 kHz..10 GHz.\n"
            "y=75..1338 -> S21=-45..-95 dB; loss=-S21.\n"
            "Red centerline interpolated across occlusions (max 16 missing columns).\n"
            "Approximate digitization: 0.0396 dB/vertical pixel, 4.305 MHz/horizontal pixel.\n"
            "Decimal precision is for reproducibility, not measurement accuracy.\n"
            "frequency_hz,loss_db"
        ),
    )
    print(f"Saved {N_POINTS} digitized points to {OUTPUT}")
    print(f"Visible red columns: {len(x_pixels)}; max gap: {np.max(np.diff(x_pixels)) - 1}")
