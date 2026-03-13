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
- **`TimeStream`** — Time-domain multi-tone acquisition. Supports multiple IF frequencies with USB/LSB sideband separation. Phase reset is gated: disabled for non-zero IF.
- **`SweepPower`** — 2D sweep over frequency × drive power.
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

### Analysis (`daq/analysis/mattis_bardeen.py`)

Implements Mattis-Bardeen superconductor theory for temperature-dependent quality factor fitting using `iminuit`. Entry point: `MB_fitter()`.

## Style Conventions

- Sphinx-style docstrings for all public classes and functions.
- Complete type annotations for public APIs; use `Optional[T]` for optional parameters.
- Black formatting with 100-character line length.
