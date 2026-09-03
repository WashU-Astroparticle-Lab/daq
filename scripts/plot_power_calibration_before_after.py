import csv, io, subprocess, sys, warnings, importlib.util
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

W = Path(sys.argv[1]); OUT = Path(sys.argv[2])
# --- old calibration exactly as main loaded it
old = np.load(io.BytesIO(subprocess.check_output(["git", "show", "origin/main:daq/calibrations/power_calibration.npz"])))
f_grid, a_grid, Z = old["f_grid"], old["a_grid"], old["Z"]
mono = lambda M: np.all(np.diff(M, axis=1) >= 0) or np.all(np.diff(M, axis=1) <= 0)
if not mono(Z): Z = Z.T
old_interp = RegularGridInterpolator((f_grid, a_grid), Z, method="linear")
old_p = lambda f, a: old_interp(np.column_stack([np.full_like(a, f), a]))
old_amp_32 = brentq(lambda a: old_p(2.8, np.array([a]))[0] + 32.0, 0.001, 1.0)
# --- new calibration from the branch
spec = importlib.util.spec_from_file_location("cal", W / "daq" / "calibrations" / "__init__.py")
cal = importlib.util.module_from_spec(spec); spec.loader.exec_module(cal)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", cal.CalibrationWarning)
    new_amp_32 = cal.power_dbm_to_amp(2.8, -32.0)
# --- measurements
meas = defaultdict(list)
for p in sorted((W / "daq" / "calibrations" / "source_data").glob("*combined*.csv")):
    for r in csv.DictReader(p.open(newline="", encoding="utf-8-sig")):
        meas[float(r["freq_ghz"])].append((float(r["amp"]), float(r["power_dbm"])))

freqs = [2.35, 2.8, 4.95, 6.95, 8.2, 9.0]
cmap = plt.colormaps["viridis"]; norm = Normalize(2.35, 9.0)
amps = np.logspace(-3, 0, 300)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)
for ax in (ax0, ax1):
    ax.axvspan(1e-3, 0.4, color="0.85", alpha=0.35, zorder=0)
    ax.axvline(0.4, color="0.4", linestyle="--", linewidth=1)
    ax.axhline(-32, color="tab:red", linestyle=":", linewidth=1)
    ax.set_xscale("log"); ax.grid(alpha=0.25)
    ax.set_xlabel("DAC amplitude (fraction of full scale)")
for f in freqs:
    c = cmap(norm(f)); pts = np.array(sorted(meas[f]))
    ax0.scatter(pts[:, 0], pts[:, 1], s=14, color=c, alpha=0.8)
    ax1.scatter(pts[:, 0], pts[:, 1], s=14, color=c, alpha=0.8)
    ax0.plot(amps, old_p(f, amps), color=c, linewidth=2.2, label=f"{f:g} GHz")
    floor = cal.min_verified_amp(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cal.CalibrationWarning)
        p_new = cal.amp_to_power_dbm(f, amps)
    hi = amps >= floor
    ax1.plot(amps[hi], p_new[hi], color=c, linewidth=2.2, label=f"{f:g} GHz")
    ax1.plot(amps[~hi], p_new[~hi], color=c, linewidth=1.6, linestyle=(0, (3, 2)), alpha=0.8)
    ax1.plot([floor], [cal.amp_to_power_dbm(f, floor)], marker="v", markersize=9,
             markerfacecolor="white", markeredgecolor=c, linestyle="none", zorder=5)
for ax, a32, txt in ((ax0, old_amp_32, f"−32 dBm → amp {old_amp_32:.4f}"), (ax1, new_amp_32, f"−32 dBm → amp {new_amp_32:.4f}")):
    ax.plot([a32], [-32], marker="X", markersize=13, color="tab:red", zorder=6)
    ax.annotate(txt, (a32, -32), xytext=(-150, 14), textcoords="offset points", color="tab:red", fontsize=11)
ax0.set_ylabel("Output power (dBm)")
ax0.set_title("Before: four-corner grid, linear in amplitude")
ax1.set_title("After: 20 log₁₀(amp) law, warned below the verified floor")
ax0.legend(fontsize=9, ncol=2, loc="lower right")
extra = [Line2D([], [], color="0.3", linewidth=2.2, label="verified region"),
         Line2D([], [], color="0.3", linewidth=1.6, linestyle=(0, (3, 2)), label="extrapolated (CalibrationWarning)"),
         Line2D([], [], marker="v", markerfacecolor="white", markeredgecolor="0.3", linestyle="none", markersize=9, label="verified floor")]
ax1.legend(handles=ax1.get_legend_handles_labels()[0] + extra, fontsize=9, ncol=2, loc="lower right")
fig.suptitle("Power calibration before and after the fix\n"
             "dots = spectrum-analyzer measurements; lines = packaged calibration; gray = below the 0.4 fit cutoff", fontsize=13)
fig.savefig(OUT, dpi=170); print("wrote", OUT, "old amp", old_amp_32, "new amp", new_amp_32)
