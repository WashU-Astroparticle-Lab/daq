# Analysis Guide

This guide covers the analysis tools in `daq.analysis` with practical examples.

## Contents

- [Noise PSD](#noise-psd)
- [Electronic to Resonator Basis](#electronic-to-resonator-basis)
- [Correlated Noise Removal](#correlated-noise-removal)
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
