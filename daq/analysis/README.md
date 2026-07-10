# Analysis Guide

This guide covers the analysis tools in `daq.analysis` with practical examples.

## Contents

- [Noise PSD](#noise-psd)
- [Parity PSD fit (random-telegraph model)](#parity-psd-fit-random-telegraph-model)
- [Averaged PSD from repeated TimeStreams](#averaged-psd-from-repeated-timestreams)
- [Electronic to Resonator Basis](#electronic-to-resonator-basis)
- [I/Q Comparison Plot](#iq-comparison-plot)
- [Correlated Noise Removal](#correlated-noise-removal)
  - [Batch cleaning of interleaved streams](#batch-cleaning-of-interleaved-streams)
- [Mattis-Bardeen Fitting](#mattis-bardeen-fitting)
- [Helper Functions](#helper-functions)

---

## Noise PSD

`compute_psd` computes the one-sided Power Spectral Density of a real-valued time series using the periodogram method (direct FFT, no windowing).

### Basic usage

```python
import numpy as np
from daq.analysis import compute_psd

# Suppose you have a time series sampled at 1 MHz
fs = 1e6  # Hz
t = np.arange(0, 1.0, 1 / fs)
signal = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(len(t))

f, psd = compute_psd(signal, fs)
```

`f` is the frequency array in Hz and `psd` has units of (signal units)^2 / Hz.

### Welch's method

By default `compute_psd` uses the bare periodogram. Pass `welch=True` to use Welch's method (`scipy.signal.welch`), which averages the periodograms of overlapping windowed segments to reduce variance at the cost of frequency resolution:

```python
# Default Welch parameters (Hann window, nperseg=256, 50% overlap)
f, psd = compute_psd(signal, fs, welch=True)

# Customize the segmentation
f, psd = compute_psd(signal, fs, welch=True, nperseg=4096, noverlap=2048)
```

The `nperseg`, `noverlap`, `window`, and `detrend` parameters are only used when `welch=True`. Note that the Welch path defaults to `detrend="constant"` (mean removal), unlike the periodogram path which applies no detrending; pass `detrend=False` to match it.

### Plotting the PSD

```python
import matplotlib.pyplot as plt

f, psd = compute_psd(signal, fs)

plt.figure()
plt.loglog(f[1:], psd[1:])  # skip DC bin
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [V$^2$/Hz]")
plt.title("Noise Power Spectral Density")
plt.grid(True, which="both", alpha=0.3)
plt.show()
```

### Using with TimeStream data

Load a saved TimeStream measurement and compute the PSD of the I or Q quadrature:

```python
from daq.measurements import TimeStream
from daq.analysis import compute_psd

ts = TimeStream.load("00000042-Resonator_A-timestream.h5")

# USB sideband data, shape: (pixel_counts, n_tones)
# Extract the I quadrature for the first tone
i_data = np.real(ts.usb[:, 0])

# The sampling rate is the lock-in demodulation bandwidth
fs = ts.df  # Hz

f, psd = compute_psd(i_data, fs)
```

### Batch PSD (multiple time series)

For 2-D input, each row is treated as an independent time series:

```python
# Stack multiple time series as rows
data_2d = np.vstack([i_data, q_data])  # shape: (2, N)

f, psd = compute_psd(data_2d, fs)
# psd.shape == (2, len(f))
```

---

## Parity PSD fit (random-telegraph model)

`fit_parity_psd` fits the PSD of a parity (random-telegraph) time-stream to Eqn. 18 of [arXiv:2601.16261](https://arxiv.org/pdf/2601.16261):

```
PSD(f) = F^2 * 4*Gamma_p / ((2*Gamma_p)^2 + (2*pi*f)^2) + (1 - F^2) / f_bw
```

The first term is the Lorentzian of the parity-switching process; the second is a white noise floor set by the readout fidelity `F` and the sampling bandwidth `f_bw`. The fit extracts the fidelity `F` and the characteristic parity-switching rate `Gamma_p` (in Hz); `f_bw` is held **fixed** — pass the acquisition sample rate (e.g. `TimeStream.df`) for it.

The function takes the `(f, psd)` output of `compute_psd` directly:

```python
import numpy as np
from daq.analysis import compute_psd, fit_parity_psd, parity_psd_model

# `parity` is the projected parity time-stream (e.g. IQ data projected onto the
# maximal-separation axis), sampled at fs.
fs = ts.df  # Hz -- this is also f_bw
f, psd = compute_psd(parity, fs)

res = fit_parity_psd(f, psd, f_bw=fs)
print(f"fidelity F = {res['fidelity']:.3f} +/- {res['fidelity_err']:.3f}")
print(f"Gamma_p    = {res['gamma_p']:.2f} +/- {res['gamma_p_err']:.2f} Hz")
print(f"f_corner   = {res['f_corner']:.2f} Hz")   # Lorentzian half-power = Gamma_p / pi
```

`res` is a dict with `fidelity`, `gamma_p` (Hz), their `*_err`, the Lorentzian half-power frequency `f_corner` (= `Gamma_p / pi`), the fixed `f_bw`, the raw `popt`/`pcov`, a `model` array (the fitted curve evaluated at every input `f`), and a `success` flag.

### Plotting the fit

```python
import matplotlib.pyplot as plt

plt.figure()
plt.loglog(f[1:], psd[1:], ".", ms=2, label="Data")
plt.loglog(f[1:], res["model"][1:], "-", label="Eqn. 18 fit")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [a.u.$^2$/Hz]")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()
```

### Weighting, DC bin, and initial guess

- By default a PSD-proportional weighting (`relative_weight=True`) is used so the multi-decade dynamic range does not let the low-frequency plateau dominate the fit. Pass explicit `sigma` (e.g. `1/sqrt(num_averages)` scaled errors) to override, or set `relative_weight=False` for uniform weighting.
- The `f == 0` DC bin is dropped by default (`drop_dc=True`), since it is meaningless after mean removal.
- Initial guesses for `F` and `Gamma_p` are estimated from the spectrum automatically; pass `p0=(F0, gamma0)` to override. Extra keyword arguments (e.g. `maxfev`) are forwarded to `scipy.optimize.curve_fit`.

### Low-frequency 1/f noise

Real parity time-streams often have a `1/f`-like excess at low frequency (drift, two-level-system noise). Left unmodelled it is absorbed into `Gamma_p`/`F` and biases them badly — the plain two-term fit can even collapse. Set `fit_onef=True` to add a `A / f^alpha` term:

```
PSD(f) = F^2 * 4*Gamma_p/((2*Gamma_p)^2 + (2*pi*f)^2) + (1 - F^2)/f_bw + A / f^alpha
```

```python
res = fit_parity_psd(f, psd, f_bw=fs, fit_onef=True)   # alpha fixed at 1.0
print(res["a_onef"], res["alpha"])          # 1/f amplitude and exponent
```

By default the exponent `alpha` is held fixed at `1.0` (pure `1/f`). Change the fixed value with `alpha=...`, or let it float as a free parameter with `fit_alpha=True`:

```python
# Fix a steeper slope
res = fit_parity_psd(f, psd, f_bw=fs, fit_onef=True, alpha=1.5)

# Or fit the slope too (needs fit_onef=True)
res = fit_parity_psd(f, psd, f_bw=fs, fit_onef=True, fit_alpha=True)
print(res["alpha"], res["alpha_err"])
```

Floating `alpha` and `Gamma_p` together can be weakly identifiable (the `1/f` slope and the Lorentzian shoulder trade off), so prefer a fixed `alpha` unless the data clearly warrant fitting it. The returned dict always carries `a_onef`, `a_onef_err`, `alpha`, `alpha_err` (with `a_onef = 0` and `nan` errors when a term is held fixed), and `res["model"]` includes the `1/f` contribution. When you pass your own `p0`, give it in the order `(fidelity, gamma_p[, a_onef[, alpha]])` matching the enabled terms.

### Batch fitting (2-D PSD)

If you pass a 2-D `psd` of shape `(n_rows, n_freqs)` — for example the per-tone PSDs from `averaged_psd_timestream` — each row is fit independently and a **list** of result dicts is returned:

```python
f, psd_a, psd_b, streams = averaged_psd_timestream(...)
results = fit_parity_psd(f, psd_a, f_bw=streams[0].df)
for ch, r in enumerate(results):
    print(ch, r["fidelity"], r["gamma_p"])
```

`parity_psd_model(f, fidelity, gamma_p, f_bw)` is exposed separately if you want to evaluate the model directly (e.g. for overplotting or simulation).

---

## Averaged PSD from repeated TimeStreams

`averaged_psd_timestream` wraps `TimeStream` for the common "take data, then show averaged PSD" workflow. It builds a multi-tone `TimeStream` with the configuration you provide, runs it `num_averages` times (e.g. 100), computes a per-tone PSD for every acquisition, and returns the PSDs averaged across acquisitions. Averaging is done as a running mean, so the raw time streams are not all held in memory at once. Each `run()` still writes its own HDF5 file and MongoDB document, just like running the measurement by hand.

The function returns `(f, psd_a, psd_b, streams)`:

- `f` — PSD frequency axis (Hz).
- `psd_a`, `psd_b` — averaged PSDs, each of shape `(n_tones, n_freqs)`.
- `streams` — the list of executed `TimeStream` objects (each already saved).

### Resonator basis (dissipation / frequency)

Pass one fitted `Sweep` per tone (aligned with `if_freqs`). Each tone is projected with `from_elec_to_reson`, so `psd_a` is the dissipation (radial) PSD and `psd_b` is the frequency (arc-length) PSD:

```python
import numpy as np
from daq.measurements import Sweep
from daq.analysis import averaged_psd_timestream

# One fitted Sweep per tone, in the same order as if_freqs
frs = np.array([sw.fit_results["fr"] for sw in sweeps])
lo = frs[0] - 5e5  # center the LO just below the tones

f, psd_diss, psd_freq, streams = averaged_psd_timestream(
    num_averages=100,
    lo_freq=lo,
    if_freqs=frs - lo,          # multiple IF tones are fine
    df=10e3,                    # sample rate (Hz)
    pixel_counts=int(10e3 * 20),
    amp=0.01,
    output_port=1,
    input_port=1,
    sweeps=sweeps,              # -> resonator-basis PSDs
    device="B260416-NG-D1",
    notes="No bias; for PSDs",
)
# psd_diss.shape == psd_freq.shape == (len(frs), len(f))
```

Plot the averaged dissipation and frequency PSDs per tone:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ch in range(psd_diss.shape[0]):
    axes[0].loglog(f[1:], psd_diss[ch, 1:], label=f"Ch {ch}")
    axes[1].loglog(f[1:], psd_freq[ch, 1:], label=f"Ch {ch}")
axes[0].set(xlabel="Frequency [Hz]", ylabel="Dissipation PSD [1/Hz]")
axes[1].set(xlabel="Frequency [Hz]", ylabel="Frequency PSD [1/Hz]")
axes[0].legend()
plt.tight_layout()
plt.show()
```

### Raw I/Q (no sweeps)

If you omit `sweeps`, no projection is applied and the two returned PSDs are the averaged PSDs of the raw I (real) and Q (imaginary) quadratures:

```python
f, psd_i, psd_q, streams = averaged_psd_timestream(
    num_averages=100,
    lo_freq=3e9,
    if_freqs=[0.0, 1e5, 2e5],
    df=1e3,
    pixel_counts=int(1e3 * 20),
    amp=0.01,
    output_port=1,
    input_port=1,
)
```

### Discarding leading junk

The first tens of milliseconds of an acquisition are often startup junk. `TimeStream` owns the trimming via its `discard_start_ms` field (default `25.0`); `averaged_psd_timestream` simply forwards it. Override it to change the amount, or set `0` to keep everything:

```python
f, psd_diss, psd_freq, streams = averaged_psd_timestream(
    num_averages=100,
    lo_freq=lo,
    if_freqs=frs - lo,
    df=10e3,
    pixel_counts=int(10e3 * 20),
    amp=0.01,
    output_port=1,
    input_port=1,
    sweeps=sweeps,
    discard_start_ms=25.0,   # drop the first 25 ms of every acquisition (default)
)
```

The number of samples dropped is `round(discard_start_ms * 1e-3 * fs)` using the actual hardware sample rate. `TimeStream` applies the cut to its in-memory time-axis arrays (`signal`, `usb`, `lsb`, `pixel_i`, `pixel_q`) after `run()` (and again after `load()`), so both the PSD input and the returned `TimeStream` objects reflect the analysed window. The HDF5 file saved by each `run()` still holds the full, untrimmed acquisition.

Welch's method and its parameters (`welch`, `nperseg`, `noverlap`, `window`, `detrend`) are forwarded to `compute_psd`, and `is_usb` is forwarded to `TimeStream` for per-tone sideband selection.

---

## Electronic to Resonator Basis

`from_elec_to_reson` transforms raw I/Q time-stream data from the electronic measurement basis into the resonator coordinate system using calibration parameters from a fitted `Sweep`. It returns the complex resonator-basis coordinate along with the dissipation (radial) and frequency (arc-length) responses.

### Basic usage

```python
from daq.measurements import Sweep, TimeStream
from daq.analysis import from_elec_to_reson

# Load a fitted sweep and a time-stream measurement at the same resonance
sw = Sweep.load("00000042-Resonator_A-sweep.h5")
ts = TimeStream.load("00000043-Resonator_A-timestream.h5")

# Transform the USB time-stream for the first tone
tsz, rad, arc = from_elec_to_reson(ts.usb[:, 0], sw)
# tsz: complex resonator-basis coordinate
# rad: dissipation (radial) response
# arc: frequency (arc-length) response
```

### Computing PSD in the resonator basis

Combine with `compute_psd` to obtain the noise spectrum in dissipation and frequency channels:

```python
from daq.analysis import compute_psd

fs = ts.df  # sampling rate (Hz)

f, psd_rad = compute_psd(rad, fs)
f, psd_arc = compute_psd(arc, fs)
```

---

## I/Q Comparison Plot

`plot_iq_comparison` overlays a time-stream I/Q cloud on the smooth fitted resonator sweep circle in the complex plane. It re-fits the sweep internally (via `resonator_tools`) so the smooth trace and the calibration parameters (`environmental_term`, `phi0`, `fr`) come from one self-consistent fit, then projects the time stream, the sweep trace, and optional QC-trace calibration points into a common display basis. Markers highlight the resonance `fr` and `fr ± freq_shift`, and the sweep trace is coloured by frequency detuning.

### Basic usage

```python
from daq.measurements import Sweep, TimeStream
from daq.analysis import plot_iq_comparison

sw = Sweep.load("00000042-Resonator_A-sweep.h5")
ts = TimeStream.load("00000043-Resonator_A-timestream.h5")

# Overlay the first-tone time stream on the sweep, in the resonator basis
ax = plot_iq_comparison(
    ts.signal[:, 0],
    sw,
    basis="resonator",
    device="Resonator_A",
    power_dbm=-95,
)
```

### Choosing a basis and a density style

The `basis` argument selects the display coordinates:

- `"electronic"` — raw I/Q (default),
- `"fractional"` — environment divided out,
- `"resonator"` — recentred on the resonance circle.

The `density` argument controls how the (typically large) time-stream cloud is rendered:

- `"scatter"` — light scatter points (default),
- `"kde"` — scatter plus 1σ / 2σ Gaussian-KDE contours (accurate but slowest),
- `"contour"` — scatter plus fast 1σ / 2σ contours from a 2-D histogram (levels bound the innermost 68.3% / 95.4% of the counts); the fast way to get contour lines on a large cloud,
- `"hexbin"` — hexagonal density bins (fastest for big clouds),
- `"hist2d"` — 2-D histogram.

For big clouds (millions of points) `"contour"` gives KDE-like σ-contours roughly 5× faster than `"kde"`, since it bins with `numpy.histogram2d` instead of evaluating a kernel per point.

```python
# Fast density view for a very large time stream, with a QC calibration trace
ax = plot_iq_comparison(
    ts.signal[:, 0],
    sw,
    qc=qc_trace,              # optional complex calibration points (red circles)
    basis="fractional",
    density="hexbin",
    freq_shift=200e3,         # fr ± 200 kHz marker diamonds
    savefig="iq_comparison.png",
)
```

Pass an existing `ax` to compose subplots, `xlim`/`ylim` to zoom, or `title` to override the auto-generated label. The function returns the matplotlib axis.

The scatter defaults (`scatter_size=0.05`, `scatter_alpha=0.005`) are tuned for million-point clouds and render nearly invisibly on small time streams. For a modest cloud, raise them:

```python
# A few thousand points: make the scatter visible
ax = plot_iq_comparison(ts.signal[:, 0], sw, scatter_size=2, scatter_alpha=0.2)
```

---

## Correlated Noise Removal

`remove_correlated_noise` subtracts correlated electronics noise (gain drift, LO phase noise) from an on-resonance tone using a simultaneously acquired off-resonance reference. It implements the cleaning procedure of Eqn 7.44–7.45 in Wen (2025), working in the gain / arc-length basis.

### Basic usage

```python
from daq.measurements import TimeStream
from daq.analysis import remove_correlated_noise

ts = TimeStream.load("00000050-Resonator_A-timestream.h5")

# on_res: tone index 0 (on resonance), off_res: tone index 1 (off resonance)
on_res = ts.usb[:, 0]
off_res = ts.usb[:, 1]

cleaned, x_r, x_rho = remove_correlated_noise(on_res, off_res, fs=ts.df)
print(f"Cleaning coefficients: x_r={x_r:.4f}, x_rho={x_rho:.4f}")
```

### Time-windowed cleaning coefficients

Restrict the time window used to compute cleaning coefficients (subtraction still applies to the full array):

```python
# Use only the first 10 seconds to compute coefficients
cleaned, x_r, x_rho = remove_correlated_noise(
    on_res, off_res, fs=ts.df, max_t_s=10.0
)

# Use a window from 5 s to 15 s
cleaned, x_r, x_rho = remove_correlated_noise(
    on_res, off_res, fs=ts.df, min_t_s=5.0, max_t_s=15.0
)
```

### Getting cleaned r/rho and computing PSDs

Set `return_r_rho=True` to also get the mean-subtracted gain and arc-length components, which can be passed directly to `compute_psd`:

```python
from daq.analysis import compute_psd

cleaned, x_r, x_rho, cleaned_r, cleaned_rho = remove_correlated_noise(
    on_res, off_res, fs=ts.df, return_r_rho=True
)

f, psd_r = compute_psd(cleaned_r, ts.df)
f, psd_rho = compute_psd(cleaned_rho, ts.df)
```

### Before/after PSD comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from daq.analysis import compute_psd, remove_correlated_noise

cleaned, x_r, x_rho, cleaned_r, cleaned_rho = remove_correlated_noise(
    on_res, off_res, fs=ts.df, return_r_rho=True
)

# Original gain fluctuations
r_original = np.abs(on_res) - np.mean(np.abs(on_res))
f, psd_before = compute_psd(r_original, ts.df)
f, psd_after = compute_psd(cleaned_r, ts.df)

plt.figure()
plt.loglog(f[1:], psd_before[1:], label="Before cleaning")
plt.loglog(f[1:], psd_after[1:], label="After cleaning")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [arb$^2$/Hz]")
plt.title("Gain noise: before vs. after correlated noise removal")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()
```

### Batch cleaning of interleaved streams

`clean_correlated_streams` applies `remove_correlated_noise` across a whole list of `TimeStream` acquisitions — for example the `streams` returned by `averaged_psd_timestream` — when the tones are interleaved as `[signal, reference, signal, reference, ...]`. Each *reference* tone sits off resonance and cleans its neighbouring on-resonance *signal* tone. It returns only the cleaned signal tones.

```python
import numpy as np
from daq.analysis import averaged_psd_timestream, clean_correlated_streams

# frs_interleaved = [TONE1, CLEAN1, TONE2, CLEAN2, ...]
lo = frs_interleaved[0] - 5e5
_, _, _, streams = averaged_psd_timestream(
    num_averages=100,
    lo_freq=lo,
    if_freqs=frs_interleaved - lo,
    df=10e3,
    pixel_counts=int(10e3 * 20),
    amp=amps_interleaved,   # per-tone array (one amp per tone)
    output_port=1,
    input_port=1,
)

# Default pairing: signals = even indices, references = odd indices.
cleaned, freqs = clean_correlated_streams(streams)
# cleaned.shape == (n_streams, n_samples, n_signal_tones)  (complex)
# freqs         == physical frequencies of the signal tones
```

### From cleaned tones to averaged PSDs

`averaged_psd_cleaned` is the PSD stage that follows `clean_correlated_streams`. It takes the `cleaned` array, computes a per-tone PSD for every acquisition, and averages them across acquisitions (running mean). It mirrors `averaged_psd_timestream`: pass one fitted `Sweep` per **signal** tone to get resonator-basis dissipation/frequency PSDs, or omit `sweeps` for raw I/Q PSDs.

```python
from daq.analysis import averaged_psd_cleaned

cleaned, freqs = clean_correlated_streams(streams)
fs = streams[0].df

# Resonator basis: one Sweep per signal tone (aligned with `freqs`)
f, psd_diss, psd_freq = averaged_psd_cleaned(cleaned, fs, sweeps=signal_sweeps)
# psd_diss.shape == psd_freq.shape == (n_signal_tones, len(f))

# Or raw I/Q PSDs when you have no sweeps
f, psd_i, psd_q = averaged_psd_cleaned(cleaned, fs)
```

Plot the cleaned, averaged PSDs per signal tone:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ch in range(psd_diss.shape[0]):
    axes[0].loglog(f[1:], psd_diss[ch, 1:], label=f"{freqs[ch]/1e9:.4f} GHz")
    axes[1].loglog(f[1:], psd_freq[ch, 1:])
axes[0].set(xlabel="Frequency [Hz]", ylabel="Dissipation PSD [1/Hz]")
axes[1].set(xlabel="Frequency [Hz]", ylabel="Frequency PSD [1/Hz]")
axes[0].legend()
plt.tight_layout()
plt.show()
```

For non-interleaved layouts (e.g. a single shared reference tone), pass the pairing explicitly:

```python
# Signals at tones 0, 2, 4 all cleaned against one shared reference at tone 6
cleaned, freqs = clean_correlated_streams(
    streams, signal_indices=[0, 2, 4], reference_indices=[6, 6, 6]
)

# Also return the per-stream, per-pair cleaning coefficients
cleaned, freqs, x_r, x_rho = clean_correlated_streams(streams, return_coeffs=True)
# x_r.shape == x_rho.shape == (n_streams, n_signal_tones)
```

The `min_t_s` / `max_t_s` window arguments are forwarded to `remove_correlated_noise`. All streams must share the same tone layout and sample count.

---

## Mattis-Bardeen Fitting

The `MB_fitter` function fits temperature-dependent resonator data (resonant frequency and internal quality factor vs. temperature) to Mattis-Bardeen superconductor theory. It uses `iminuit` for chi-squared minimization and extracts material parameters like the superconducting gap and kinetic inductance fraction.

### Preparing temperature sweep data

You need arrays of temperature, internal quality factor, resonant frequency, and their variances. These typically come from fitting a set of `Sweep` measurements taken at different temperatures:

```python
import numpy as np
from daq.measurements import Sweep

# List of (temperature_K, filename) pairs from your temperature sweep
temp_files = [
    (0.050, "00000101-Resonator_A-sweep.h5"),
    (0.100, "00000102-Resonator_A-sweep.h5"),
    (0.150, "00000103-Resonator_A-sweep.h5"),
    (0.200, "00000104-Resonator_A-sweep.h5"),
    (0.250, "00000105-Resonator_A-sweep.h5"),
    (0.300, "00000106-Resonator_A-sweep.h5"),
]

T_arr = []
fr_arr = []
Qi_arr = []
fr_err_arr = []
Qi_err_arr = []

for T, filename in temp_files:
    sweep = Sweep.load(filename)
    fr = sweep.analyze(batch=True)  # returns resonant frequency via fit

    # Access fit results
    T_arr.append(T)
    fr_arr.append(sweep.fit_results["fr"])
    Qi_arr.append(sweep.fit_results["Qi_dia_corr"])
    fr_err_arr.append(sweep.fit_results["fr_err"])
    Qi_err_arr.append(sweep.fit_results["Qi_dia_corr_err"])

T_arr = np.array(T_arr)          # K
fr_arr = np.array(fr_arr)        # Hz
Qi_arr = np.array(Qi_arr)        # dimensionless
var_f = np.array(fr_err_arr)**2  # variance of frequency
var_Qi = np.array(Qi_err_arr)**2 # variance of Qi
```

### Running the fit

```python
from daq.analysis import MB_fitter

f0, Delta0, alpha, Qi0, chi2_dof, f0_err, Delta0_err, alpha_err, Qi0_err = MB_fitter(
    T_arr, Qi_arr, fr_arr, var_Qi, var_f
)

print(f"f0     = {f0:.6f} +/- {f0_err:.6f} GHz")
print(f"Delta0 = {Delta0:.4f} +/- {Delta0_err:.4f} meV")
print(f"alpha  = {alpha:.4f} +/- {alpha_err:.4f}")
print(f"Qi0    = {Qi0:.0f} +/- {Qi0_err:.0f}")
print(f"chi2/dof = {chi2_dof:.3f}")
```

**Return values:**

| Value | Unit | Description |
|---|---|---|
| `f0` | GHz | Resonant frequency at T = 0 |
| `Delta0` | meV | Superconducting gap energy |
| `alpha` | dimensionless | Kinetic inductance fraction |
| `Qi0` | dimensionless | Internal quality factor at T = 0 |
| `chi2_dof` | dimensionless | Reduced chi-squared of the fit |
| `*_err` | (same as value) | 1-sigma uncertainties from Hesse |

### Fixing parameters

Pass keyword arguments to fix any parameter during the fit:

```python
# Fix the superconducting gap to the BCS value for aluminum (0.18 meV)
f0, Delta0, alpha, Qi0, chi2_dof, *errs = MB_fitter(
    T_arr, Qi_arr, fr_arr, var_Qi, var_f,
    Delta0=1.8e-4  # eV (0.18 meV)
)
```

### Plotting fit results

```python
import matplotlib.pyplot as plt
from daq.analysis import f_T, Qi_T

T_dense = np.linspace(T_arr.min(), T_arr.max(), 200)

# Convert fitted values back to SI for the model functions
f0_Hz = f0 * 1e9
Delta0_eV = Delta0 * 1e-3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Frequency vs temperature
ax1.errorbar(T_arr * 1e3, fr_arr * 1e-9, yerr=np.sqrt(var_f) * 1e-9,
             fmt="o", label="Data")
ax1.plot(T_dense * 1e3, f_T(T_dense, f0_Hz, Delta0_eV, alpha) * 1e-9,
         "--", label="MB fit")
ax1.set_xlabel("Temperature [mK]")
ax1.set_ylabel("Frequency [GHz]")
ax1.legend()

# Qi vs temperature
ax2.errorbar(T_arr * 1e3, Qi_arr, yerr=np.sqrt(var_Qi), fmt="o", label="Data")
ax2.plot(T_dense * 1e3, Qi_T(T_dense, f0_Hz, Qi0, Delta0_eV, alpha),
         "--", label="MB fit")
ax2.set_xlabel("Temperature [mK]")
ax2.set_ylabel("$Q_i$")
ax2.set_yscale("log")
ax2.legend()

plt.tight_layout()
plt.show()
```

---

## Helper Functions

The analysis module also exposes several physical model functions that can be used independently.

### Quasiparticle density: `n_qp(T, Delta0)`

Thermal quasiparticle number density (m^-3) at temperature `T` (K) for gap `Delta0` (eV):

```python
from daq.analysis import n_qp

# Quasiparticle density at 100 mK for aluminum (Delta0 ~ 0.18 meV)
nqp = n_qp(0.1, 1.8e-4)
print(f"n_qp = {nqp:.2e} m^-3")
```

### Mattis-Bardeen kernel functions: `S_1`, `S_2`

The dimensionless Mattis-Bardeen integrals that govern dissipation (`S_1`) and frequency shift (`S_2`):

```python
from daq.analysis import S_1, S_2

fr = 5e9     # 5 GHz resonator
T = 0.15     # 150 mK
Delta = 1.8e-4  # eV

s1 = S_1(fr, T, Delta)
s2 = S_2(fr, T, Delta)
print(f"S_1 = {s1:.6f}")
print(f"S_2 = {s2:.6f}")
```

### Loss and frequency shift kernels: `kappa_1`, `kappa_2`

These give the quasiparticle-induced loss rate and fractional frequency shift per quasiparticle:

```python
from daq.analysis import kappa_1, kappa_2

k1 = kappa_1(0.15, 5e9, 1.8e-4)  # (T, f0, Delta0)
k2 = kappa_2(0.15, 5e9, 1.8e-4)

print(f"kappa_1 = {k1:.4e}")
print(f"kappa_2 = {k2:.4e}")
```

### Signed log: `signed_log10`

A utility for plotting quantities that span positive and negative values on a log scale:

```python
from daq.analysis import signed_log10

x = np.array([-100, -1, 0.01, 1, 100])
print(signed_log10(x))
# [-2., -0.,  -2.,  0.,  2.]
```

---

## Physical Constants

The Mattis-Bardeen module uses the following built-in constants (aluminum):

| Constant | Value | Unit | Description |
|---|---|---|---|
| `Boltz_k` | 8.617e-5 | eV/K | Boltzmann constant |
| `N_0` | 1.72e28 | m^-3 eV^-1 | Single-spin density of states |
| `Planck_h` | 4.136e-15 | eV s | Planck constant |
