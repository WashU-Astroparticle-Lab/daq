# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DAQ is a Python library providing stable wrappers for data acquisition using Intermodulation Product Presto-8 quantum measurement hardware at the WashU Astroparticle Lab. It handles measurement execution, HDF5 data storage, and MongoDB metadata logging.

## Environment Setup

Always activate the conda environment before running any commands:
```bash
conda activate presto
```

## References
The `presto` package is documented here: https://www.intermod.pro/manuals/presto/index.html, and this should be the major reference to be read before searching externally.

## Commands

**Install:**
```bash
pip install -e .
```

**Format code:**
```bash
black --line-length 100 daq/
docformatter --style sphinx --wrap-summaries 100 --wrap-descriptions 100 daq/
```

There are no automated tests or CI workflows in this repo.

## Configuration

Runtime settings are loaded from environment variables (see `daq/config.py`):

| Variable | Default |
|---|---|
| `DAQ_PRESTO_ADDRESS` | `172.23.20.29` |
| `DAQ_PRESTO_PORT` | Presto default |
| `DAQ_DATA_FOLDER` | `<repo>/data` |
| `DAQ_MONGODB_URI` | `mongodb://localhost:27017` |
| `DAQ_MONGODB_DB_NAME` | `WashU_Astroparticle_Detector` |
| `DAQ_MONGODB_COLLECTION_NAME` | `measurement` |

Settings are cached after first access via `get_settings()`. Call `reload_settings()` to pick up environment changes.

## Architecture

### Measurement Classes (`daq/measurements/`)

All measurement classes inherit from `Base` (`daq/_base.py`). Each runs a hardware acquisition via the `presto` library, optionally fits the data, saves results to HDF5, and logs metadata to MongoDB.

- **`Sweep`** — Single-tone frequency sweep. Auto-fits resonator parameters (fr, Qi, Qc, Ql, kappa) using `resonator_tools`.
- **`TimeStream`** — Time-domain multi-tone acquisition. Supports multiple IF frequencies with per-tone USB/LSB sideband selection via the `is_usb` bool array (`True` → `LO + IF`, `False` → `LO - IF`; default all-USB). `amp` accepts a per-tone array or a single scalar that is broadcast to every tone (equal drive); a scalar is *not* split across tones. This broadcast guards a presto footgun — `set_amplitudes` drives only tone 0 and silently zeroes the rest when given fewer amplitudes than tones — and `check_amp` validates both that `amp` length matches `if_freqs` and that the per-tone sum stays below DAC full scale (`< 1.0`). Single-sideband output phases are derived automatically. Each `|IF| < 500 MHz`, so centering the LO between tones lets two tones up to 1 GHz apart be read at once. After `run()`, `signal`/`signal_freqs` give the per-tone selected-sideband data and physical frequencies. Phase reset is gated: disabled for non-zero IF.
- **`SweepPower`** — 2D sweep over frequency × drive power. Accepts `device`/`filter`/`notes` for database logging (required by `_save`, like `Sweep`). With `auto_fit=True` (default), `run()` fits the resonator once per drive amplitude (via `resonator_tools`, centered on each row's amplitude minimum and cropped to half the span) and stores the per-amp `fitresults` dicts as a list in `self.fit_results` (`None` for any amp whose fit fails). This list lives on the object only — it is not written to HDF5 or MongoDB. An optional `attenuation_db` (default `None`) shifts the plotted drive-power axis to the device input (`drive power - attenuation_db`, labelled "Drive power at device"). `analyze()` shows the response-amplitude map plus the best-fit `fr` and `Qi` (diagonally corrected) versus drive power, each with fit-error bars (`fr_err`, `Qi_dia_corr_err`); it lazily fits via `_perform_fit()` when `fit_results` is unset (e.g. after `load()`).
- **`SweepFreqAndDC`** — 2D sweep over frequency × DC bias (JPA modulation curves).
- **`TwoTonePower`** — Two-tone spectroscopy: pump power/frequency vs. fixed probe frequency.

### Base Class (`daq/_base.py`)

Provides `_save()` (writes HDF5 + inserts MongoDB document) and `_build_document()`. Defines hardware constants: `DAC_CURRENT = 40_500` μA, `ADC_ATTENUATION = 0` dB.

### Database (`daq/db/database.py`)

MongoDB integration. Key functions:
- `get_next_number()` — Returns next 8-digit cumulative measurement number (falls back to timestamp if DB unavailable).
- `insert_measurement(document)` — Insert measurement metadata.
- `select_runs(**kwargs)` — Rich query with filtering, regex, and time ranges; returns a `pandas.DataFrame`.
- `list_devices()` — List unique devices and measurement counts.

### Data Storage

Files are saved as `{number}-{device}-{type}.h5` (e.g., `00000042-Resonator_A-sweep.h5`) under `DAQ_DATA_FOLDER`. Each HDF5 file stores the acquisition script source, all measurement parameters as attributes, data arrays, and fit results.

### Calibrations (`daq/calibrations/`)

Power calibration module. Translates between DAC full-scale amplitude (`amp`) and output power in dBm via packaged calibration grid data (`power_calibration.npz`). Key functions:
- `amp_to_power_dbm(freq_ghz, amp)` — Forward conversion (used by `SweepPower` and `TwoTonePower` plots).
- `power_dbm_to_amp(freq_ghz, power_dbm)` — Inverse conversion via `scipy.optimize.brentq`.

### Analysis (`daq/analysis/`)

- **`compute_psd`** (`noise.py`) — Noise PSD for real-valued time series (1-D or 2-D). Uses the bare periodogram by default; pass `welch=True` for Welch's method (`scipy.signal.welch`).
- **`averaged_psd_timestream`** (`noise.py`) — Wraps `TimeStream` for the "take data, then show averaged PSD" workflow: builds a multi-tone `TimeStream`, runs it `num_averages` times (each run saved as usual), computes a per-tone PSD each time, and returns the running-mean-averaged PSDs `(f, psd_a, psd_b, streams)` with `psd_*` shape `(n_tones, n_freqs)`. Pass one fitted `Sweep` per tone via `sweeps` to get resonator-basis dissipation/frequency PSDs (via `from_elec_to_reson`); otherwise returns raw I/Q PSDs. `discard_start_s` drops leading junk (e.g. the first 0.2 ms) from both the PSD input and the returned in-memory `TimeStream` arrays; the saved HDF5 keeps the full acquisition.
- **`from_elec_to_reson`** (`noise.py`) — Transform raw I/Q time-stream data from electronic to resonator basis using a fitted Sweep.
- **`remove_correlated_noise`** (`noise.py`) — Subtract correlated electronics noise (gain drift, LO phase noise) using an off-resonance reference tone. Implements Eqn 7.44–7.45 from Wen (2025) in the gain / arc-length basis.
- **`clean_correlated_streams`** (`noise.py`) — Batch wrapper that applies `remove_correlated_noise` across a list of `TimeStream` acquisitions (e.g. the `streams` from `averaged_psd_timestream`) whose tones are interleaved as `[signal, reference, ...]`. Defaults to pairing even-indexed signal tones with odd-indexed reference tones (override via `signal_indices`/`reference_indices`), and returns only the cleaned signal tones as `(cleaned, freqs)` with `cleaned` shape `(n_streams, n_samples, n_signal_tones)`.
- **`averaged_psd_cleaned`** (`noise.py`) — PSD stage following `clean_correlated_streams`: takes its `cleaned` array and returns per-signal-tone PSDs averaged across acquisitions (running mean) as `(f, psd_a, psd_b)`. Mirrors `averaged_psd_timestream` — pass one `Sweep` per signal tone for resonator-basis dissipation/frequency PSDs, else raw I/Q.
- **`plot_iq_comparison`** (`plotting.py`) — Overlays a `TimeStream` I/Q cloud on the smooth fitted resonator sweep circle in the complex plane. Re-fits the `Sweep` internally (via `resonator_tools`) so the smooth `z_data_sim` trace and the calibration parameters (`environmental_term`, `phi0`, `fr`) come from one self-consistent fit, then projects the time stream, sweep trace, and optional QC-trace points into a common `basis` (`"electronic"`/`"fractional"`/`"resonator"`). Renders the cloud via `density` (`"scatter"`/`"kde"`/`"contour"`/`"hexbin"`/`"hist2d"`; `"contour"` gives fast histogram-based 1σ/2σ rings, ~5× faster than the KDE path on large clouds), marks `fr` and `fr ± freq_shift`, colours the sweep trace by detuning, and returns the matplotlib axis. `device`/`power_dbm` feed the auto-title (replacing the previously hardcoded globals).
- **`MB_fitter`** (`mattis_bardeen.py`) — Mattis-Bardeen superconductor theory fit for temperature-dependent resonant frequency and internal quality factor using `iminuit`.
- Helper functions: `n_qp`, `f_T`, `Qi_T`, `kappa_1`, `kappa_2`, `S_1`, `S_2`, `signed_log10`.

Usage examples are in `daq/analysis/README.md`.

## Documentation

When adding or modifying measurement classes or analysis modules, update the corresponding documentation:

- **Measurement classes** — Document new classes in the Architecture > Measurement Classes section of this file.
- **Analysis tools** — Add usage examples to `daq/analysis/README.md` and update the Architecture > Analysis section of this file.

## Style Conventions

- Sphinx-style docstrings for all public classes and functions.
- Complete type annotations for public APIs; use `Optional[T]` for optional parameters.
- Black formatting with 100-character line length.
