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
| `DAQ_FGEN_RESOURCE` | `None` (autodiscover) |
| `DAQ_LED_RESOURCE` | `None` (autodiscover) |
| `DAQ_VISA_BACKEND` | `""` (system NI-VISA) |

Settings are cached after first access via `get_settings()`. Call `reload_settings()` to pick up environment changes.

## Architecture

### Measurement Classes (`daq/measurements/`)

All measurement classes inherit from `Base` (`daq/_base.py`). Each runs a hardware acquisition via the `presto` library, optionally fits the data, saves results to HDF5, and logs metadata to MongoDB.

- **`Sweep`** — Single-tone frequency sweep. Auto-fits resonator parameters (fr, Qi, Qc, Ql, kappa) using `resonator_tools`.
- **`TimeStream`** — Time-domain multi-tone acquisition. Supports multiple IF frequencies with per-tone USB/LSB sideband selection via the `is_usb` bool array (`True` → `LO + IF`, `False` → `LO - IF`; default all-USB). `amp` accepts a per-tone array or a single scalar that is broadcast to every tone (equal drive); a scalar is *not* split across tones. This broadcast guards a presto footgun — `set_amplitudes` drives only tone 0 and silently zeroes the rest when given fewer amplitudes than tones — and `check_amp` validates both that `amp` length matches `if_freqs` and that the per-tone sum stays below DAC full scale (`< 1.0`). Single-sideband output phases are derived automatically. Each `|IF| < 500 MHz`, so centering the LO between tones lets two tones up to 1 GHz apart be read at once. After `run()`, `signal`/`signal_freqs` give the per-tone selected-sideband data and physical frequencies. Phase reset is gated: disabled for non-zero IF. `discard_start_ms` (default `25.0`, set `0` to disable) drops that many leading milliseconds of startup junk from the in-memory time-axis arrays (`signal`/`usb`/`lsb`/`pixel_i`/`pixel_q`) after `run()` and `load()`, using `round(discard_start_ms * 1e-3 * df)` samples at the tuned rate; the saved HDF5 keeps the full acquisition. This trimming lives on `TimeStream` alone — `averaged_psd_timestream` forwards the field rather than re-implementing it. `external_trigger` selects **which Presto digital output ports** assert a trigger, since presto's `set_trigger_out` takes one state per port (element *i* → port *i+1*; `0` off, `1` every lock-in window, `2` every sum window): `False` triggers nothing, `True` is shorthand for `[1]` (port 1, where the 33220A gate is wired), `[0, 1]` gates port 2 (the DC2200 modulation input) and `[1, 1]` fires both. `resolve_trigger_states()` normalises the argument and rejects out-of-range states, non-integral states (a state computed as `0.999…` would otherwise silently disable the port) and more than four ports (presto packs the states two bits at a time into a uint8). Stored — and round-tripped through HDF5/MongoDB — as the resolved states array, with `load()` still accepting the older scalar-bool files. The constructor argument is backward compatible, but the *attribute* is now an array: test it with `.any()`/`.size`, since a bare `if ts.external_trigger:` raises on an empty or multi-element array. The MongoDB field likewise changed from a bool to a list, so a `select_runs(external_trigger=...)` filter matches only records of the matching vintage. Two properties of the presto layer shape how this can be used: the `delay`/`width` pair is **global**, not per port, so ports enabled together are gated identically; and the trigger re-asserts at the start of every lock-in window, so `TRIGGER_WIDTH_S = 0.03` (≫ `1/df`) leaves the line **high for the whole acquisition** rather than emitting a 30 ms one-shot. A width below `1/df` would chop it into a pulse train at the sample rate; a genuine short pulse needs `presto.pulsed`'s `output_digital_marker`, not this class. (The continuous-high behaviour is inferred from the presto implementation and from `QCTrace`'s gated ramp working, not yet scope-verified.)
- **`SweepPower`** — 2D sweep over frequency × drive power. Accepts `device`/`filter`/`notes` for database logging (required by `_save`, like `Sweep`). With `auto_fit=True` (default), `run()` fits the resonator once per drive amplitude (via `resonator_tools`, centered on each row's amplitude minimum and cropped to half the span) and stores the per-amp `fitresults` dicts as a list in `self.fit_results` (`None` for any amp whose fit fails). This list lives on the object only — it is not written to HDF5 or MongoDB. An optional `attenuation_db` (default `None`) shifts the plotted drive-power axis to the device input (`drive power - attenuation_db`, labelled "Drive power at device"). `analyze()` shows the response-amplitude map plus the best-fit `fr` and `Qi` (diagonally corrected) versus drive power, each with fit-error bars (`fr_err`, `Qi_dia_corr_err`); it lazily fits via `_perform_fit()` when `fit_results` is unset (e.g. after `load()`).
- **`SweepFreqAndDC`** — 2D sweep over frequency × DC bias (JPA modulation curves).
- **`TwoTonePower`** — Two-tone spectroscopy: pump power/frequency vs. fixed probe frequency.
- **`QCTrace`** — Quantum-capacitance trace: the charge-parity / quasiparticle-tunnelling routine on one device. Unlike the classes above it drives no hardware of its own — it *composes* `Sweep`, several `TimeStream`s and an `Agilent33220A` gate bias, and is the one deliberate exception to "no per-experiment measurement classes" (see Instruments) because the four steps and their ordering are the measurement. `run()` does: (1) a `Sweep` with auto-fit to locate `fr`, which every later step reads out at — raises `RuntimeError` if the fit gives no usable `fr` (the sweep's own file is saved, and its path is in the message); (2) the **QC trace** — a gated sawtooth plus an externally-triggered `TimeStream` spanning `num_periods` whole ramp periods, folded into one period by `fold_timestream`; (3) a **bias hunt** — one constant-bias `TimeStream` per entry of `bias_voltages`, keeping the largest parity contrast `std(|signal|)`; (4) one `TimeStream` under a **free-running** (`gated=False`) ramp, same length as the tries. Step 4 must not be gated: the Presto trigger is deasserted there, so a gated ramp would sit at its burst start level and record a static bias. Folding uses `period_s=1/ramp_freq_hz` at the *tuned* `df` rather than dividing the record by `num_periods`, so a slightly shifted `df` drops a leftover sample instead of smearing the average across blocks. The bias voltages are drawn (uniform over the ramp's own voltage range, `seed`-able) in `__init__`, so the object and its saved record fully pin down what was measured; pass `bias_voltages` explicitly (e.g. a `linspace`) to scan instead of sample. Every step saves its own HDF5/MongoDB record via the normal path with `attach(bias=...)` applied, so each raw acquisition stays individually loadable; `QCTrace` saves one further `qc_trace` record holding the derived products (`fr`, `time_ms`/`avg_iq`, `parity_contrast`, `best_bias`) and the constituent file paths (`sweep_file`, `qc_file`, `best_bias_file`, `ramp_file`). The sweep's `fitresults` is copied to `self.fit_results`, so the composite record also carries `fit_fr`/`fit_Qi`/`fit_Qc`/`fit_Ql`/`fit_kappa` in MongoDB. The constituent objects hang off read-only properties (`sweep`, `qc_stream`, `bias_streams`, `best_bias_stream`, `ramp_stream`) so `Base._save` skips them; `load()` restores the derived record but not those objects (reload them from the saved paths). The bias output is forced off before the sweep and again on exit — including on exception — and a caller-supplied `bias` instrument is de-energised but never closed. `analyze()` plots parity contrast versus gate bias with the winner marked, above the folded I/Q trace drawn by `plot_qc_trace`.

### Base Class (`daq/_base.py`)

Provides `_save()` (writes HDF5 + inserts MongoDB document) and `_build_document()`. Defines hardware constants: `DAC_CURRENT = 40_500` μA, `ADC_ATTENUATION = 0` dB.

`attach(**instruments)` records auxiliary instrument state on a measurement: each instrument's read-back `settings()` mapping is flattened onto the object as `<prefix>_<key>` scalar attributes, which `_save()`/`_build_document()` then pick up for free (no per-instrument code in `_base.py`). Call it before `run()` so bias voltages and LED parameters land in HDF5 and MongoDB — and become `select_runs()`-queryable — instead of surviving only inside a `notes` string. Accepts anything with a `settings()` method, or a plain dict; `None` values are skipped since they have no HDF5 representation.

### Instruments (`daq/instruments/`)

Drivers for non-Presto benchtop hardware reached over VISA/SCPI. These are **instruments, not measurements** — they produce no data arrays and do not subclass `Base`; you compose them with the ordinary measurement classes in a notebook and record their state via `Base.attach()`. Deliberately almost no per-experiment measurement classes: improvised combinations stay in user code. The one exception is `QCTrace`, where the sequence of steps and their bias states *is* the measurement and getting the order wrong silently produces the wrong data.

- **`VisaInstrument`** (`_visa.py`) — shared base. Lazy `pyvisa` import (so `import daq` works with no VISA runtime); resource resolution by explicit argument → env var → autodiscovery filtered on `*IDN?`, **raising** on zero or multiple matches rather than taking `list_resources()[0]`. Discovery needs no configuration in the normal case — the rule is one match *per model*, so unrelated instruments on the same computer are simply ignored; each driver's `RESOURCE_HINTS` (USB VID/PID) further narrows probing so unrelated devices are not opened at all, falling back to probing everything if nothing matches the hint. `probe_visa_resources()` lists every visible resource with its `*IDN?` (or the error explaining why it did not answer) and is the first thing to run when something will not connect; `visa_backend_info()` names the loaded VISA library, which distinguishes a vendor VISA from the pure-Python backend (the latter cannot see USB instruments without `pyusb`/`libusb`). Failed discovery reports the same per-resource detail, so a model mismatch, a busy resource and a timeout are distinguishable — and when no USB/GPIB resource is visible at all it says outright that the likeliest cause is an unplugged or unpowered instrument, which is the most common failure by a wide margin. every write bracketed by a `SYST:ERR?` drain and check that raises `InstrumentError`; context manager whose `__exit__` always forces `safe_state()`; optional `transcript_path` SCPI log. Subclasses set `IDN_KEYWORDS`/`RESOURCE_HINTS`, override `env_resource()` and `safe_state()`, and extend `settings()`.
- **`Agilent33220A`** (`function_generator.py`) — gate-bias source. `constant(offset_v)` for DC; `sawtooth(vpp, freq_hz, ...)` for a ramp, gated on the Presto trigger by default (pair with `TimeStream(external_trigger=True)`, i.e. digital output port 1, where the gate input is wired), with `offset_v` defaulting to `vpp/2` (unipolar-positive). Both setters are **hermetic** — each writes the full state its mode depends on, and `constant()` disables burst *before* selecting the DC carrier, since the 33220A rejects that combination. `samples_for_periods(n_periods, sample_rate)` returns the `pixel_counts` spanning whole ramp periods plus the `TimeStream` discard window, keeping generator and acquisition in step. Amplitudes below the instrument minimum (20 mVpp into high-Z, 10 mVpp into 50 Ω) raise.
- **`DC2200`** (`dc2200.py`) — Thorlabs LED driver. Every SCPI header comes from the DC2200 Operation Manual v1.8 (29-Nov-2023) §4.3.2, so nothing is guessed. The instrument offers seven modes; four are wrapped, distinguished by what generates the timing:
  - `configure_cc(current_a)` — steady current (`SOURCE1:CCURENT:CURRENT`, Thorlabs' own misspelling, required verbatim).
  - `configure_pwm(current_a, freq_hz, duty_pct, count)` / `pwm_train(...)` — burst defined by a **duty cycle**, amplitude in **amps**. `pwm_train` is the blocking configure → on → wait → off form; `configure_pwm` returns immediately. Instrument ranges: freq `0.1 Hz–20 kHz`, duty `0.1–99.9 %`, count `1–1000` or `0` = infinite. The **0.1 % duty floor** couples width to rate — the narrowest PWM pulse is `0.001 / freq_hz`, i.e. 10 µs only at 100 Hz, 500 µs at 2 Hz.
  - `configure_pulse(on_time_s, off_time_s=|freq_hz=|period_s=, brightness_pct=|current_a=, count)` — burst defined by explicit **ON and OFF times**, so width and rate are independent; this is the only way to get a short pulse at a slow rate (10 µs at 2 Hz). Each time spans `0.001 ms–10 s`, giving `0.05–500 Hz`. Note the amplitude is **brightness as a percent of the configured current limit**, not a current: `current_a` is accepted and converted via the queried limit, but with a high limit a small absolute current is a tiny percentage that the instrument's resolution may not reach — lower the LED current limit for fine control near the bottom.
  - `configure_ttl(current_a)` — LED follows the rear-panel SMA modulation input ("TTL" in the plain digital-logic sense: low `0–0.8 V`, high `2.0–5.0 V`, 10 kΩ), the only mode whose timing is synchronised to an acquisition. Note it yields an illumination **window** spanning the record, not a pulse within it: the Presto re-asserts its trigger at the start of every lock-in window, so with `TRIGGER_WIDTH_S` far above `1/df` the line is high from the first sample to the last. The LED is wired to **digital output port 2** in the lab's default setup, so pass `external_trigger=[0, 1]` (`True` would gate port 1, the 33220A, and leave the LED dark); `[1, 1]` fires both. Set **`discard_start_ms=0`**, since the LED comes up with the RF drive at `t = 0` and the default 25 ms trim would drop the turn-on transient. A short *synchronised* flash needs `presto.pulsed` (not wrapped); `configure_pwm`/`configure_pulse` give real short pulses at a software-start offset instead.

  Configuring is *not* arming: every `configure_*` defaults to `output=False`, so it can never illuminate the LED as a side effect. In TTL mode, enabling the output arms the stage and the LED then follows the input, so arm immediately before `run()` — the idle level of the Presto trigger line between acquisitions is not controlled here and has not been measured. In PWM and pulse mode the train starts on the software write that enables the output, so neither is synchronised to an acquisition. Currents are validated against the queried `SOURCE1:CURRENT:LIMIT?`. Note the DC2200 has **two output terminals** (LED1 10 A/12-pin, LED2 2 A/4-pin) and every source setting applies to the selected one — the `terminal` property reads and sets it rather than silently inheriting the front-panel choice. `protection_status()` reports the current-limit, interlock and driver/head over-temperature trip flags, which shut the output down independently of the settings. In PWM and pulse mode the SMA connector *outputs* the internal modulation as a TTL signal, so the LED timing can be fed back to the Presto as a marker.

Setup, SCPI gotchas and troubleshooting are in `daq/instruments/README.md`. `pyvisa` is an optional extra (`pip install daq[instruments]`).

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
- **`averaged_psd_timestream`** (`noise.py`) — Wraps `TimeStream` for the "take data, then show averaged PSD" workflow: builds a multi-tone `TimeStream`, runs it `num_averages` times (each run saved as usual), computes a per-tone PSD each time, and returns the running-mean-averaged PSDs `(f, psd_a, psd_b, streams)` with `psd_*` shape `(n_tones, n_freqs)`. Pass one fitted `Sweep` per tone via `sweeps` to get resonator-basis dissipation/frequency PSDs (via `from_elec_to_reson`); otherwise returns raw I/Q PSDs. `discard_start_ms` (default `25.0`) is forwarded to `TimeStream`, which owns the leading-junk trim (see above); both the PSD input and the returned in-memory `TimeStream` arrays reflect the trimmed window, while the saved HDF5 keeps the full acquisition.
- **`parity_psd_model`** / **`fit_parity_psd`** (`noise.py`) — Random-telegraph (RTS) parity-timestream PSD model and fitter for Eqn. 18 of arXiv:2601.16261: `PSD(f) = F² · 4Γp/((2Γp)² + (2πf)²) + (1-F²)/f_bw`, a switching-rate Lorentzian plus a fidelity-limited white floor. `fit_parity_psd` takes the `(f, psd)` output of `compute_psd` and fits the readout fidelity `F` and parity-switching rate `Γp` (Hz) with the sampling bandwidth `f_bw` held fixed (pass the sample rate, e.g. `TimeStream.df`). The fit is done in **log-log space** (the natural PSD representation): the spectrum is first averaged onto `n_bins` (default 60) logarithmically-spaced frequency bins — geometric-mean frequency, linear-mean power per bin — so every decade is represented equally and the fit is not dictated by the densely-sampled high-frequency structure, then the residual of `log10(PSD)` vs `log10(model)` is minimized. `bin_weighting` sets the weighting: `"uniform"` (default) weights every bin equally so each decade contributes equally (best for Welch/averaged PSDs and for not letting high-f dominate), `"count"` weights by `~1/sqrt(m)` (m = points/bin, statistically optimal, better for a bare periodogram whose low-f bins are sparse); a true per-point `sigma` (on the linear PSD) overrides both. The `f=0` DC bin (and any non-positive frequency) is always excluded (`drop_dc` retained but inert). The fit runs on `iminuit` (`iminuit.cost.LeastSquares` + `MIGRAD`/`HESSE`), so errors come from Minuit's Hesse step. Returns a `fit_results` dict with a best fit and error per term (`fidelity`/`fidelity_err`, `gamma_p`/`gamma_p_err`, `a_onef`/`a_onef_err`, `alpha`/`alpha_err`, `f_corner=Γp/π`/`f_corner_err`) plus `f_bw`, `chi2`, `ndof`, `reduced_chi2` (over the log-binned points), `resid_dex_rms` (RMS log10 data-vs-model residual in decades — weighting-independent goodness-of-fit, `~0.1` good/`~1` bad; prefer it over `reduced_chi2` for flagging bad fits since uniform weighting makes `reduced_chi2` unreliable), `model` (evaluated at every input `f`), the log-binned points fit (`f_binned`, `psd_binned`) and non-empty-bin count `n_bins`, the `minuit` object, and `success` (`Minuit.valid`). Fitting a 1/f-dominated spectrum (e.g. electronic-basis I/Q) with `fit_onef=False` makes the two-term model collapse to the flat white floor — use `fit_onef=True` for 1/f-like low-frequency rise. `parity_psd_model` and the fit take an overall amplitude `sigma^2` (signal variance) scaling the parity term; `fit_parity_psd` fits it by default (`fit_amplitude=True`), which is required for **un-normalized** input (raw electronic I/Q, variance ≪ 1) — Eqn. 18 as written assumes a normalized ±1 signal (variance 1), so with `amplitude=1` its floor `(1-F²)/f_bw ≈ 1/f_bw` is orders of magnitude above a small-variance PSD and the fit collapses (`F→1`, corner out of band). With `sigma²` free, `F` comes from the Lorentzian/floor ratio and `Γp` from the corner (both scale-free). Set `fit_amplitude=False` only for already-normalized data (recovers strict two-parameter Eqn. 18); results carry `amplitude`/`amplitude_err`. Because the default weighting is only relative, errors are rescaled by `sqrt(chi2/ndof)` unless `absolute_sigma=True` (pass a true `sigma` for that). A 2-D `psd` (one PSD per row) is fit row-by-row, returning a list of such dicts. An optional low-frequency `1/f` term `A/f^α` (drift / TLS noise) is added with `fit_onef=True` (amplitude `A` becomes free; exponent `α` fixed at `alpha`, default `1.0`, or fit too with `fit_alpha=True`); held-fixed 1/f terms report `nan` errors (`a_onef=0` when disabled). `parity_psd_model` gains matching `a_onef`/`alpha` kwargs (default `0`/`1.0`, i.e. pure Eqn. 18) and guards the `f=0` divergence.
- **`from_elec_to_reson`** (`noise.py`) — Transform raw I/Q time-stream data from electronic to resonator basis using a fitted Sweep.
- **`remove_correlated_noise`** (`noise.py`) — Subtract correlated electronics noise (gain drift, LO phase noise) using an off-resonance reference tone. Implements Eqn 7.44–7.45 from Wen (2025) in the gain / arc-length basis.
- **`clean_correlated_streams`** (`noise.py`) — Batch wrapper that applies `remove_correlated_noise` across a list of `TimeStream` acquisitions (e.g. the `streams` from `averaged_psd_timestream`) whose tones are interleaved as `[signal, reference, ...]`. Defaults to pairing even-indexed signal tones with odd-indexed reference tones (override via `signal_indices`/`reference_indices`), and returns only the cleaned signal tones as `(cleaned, freqs)` with `cleaned` shape `(n_streams, n_samples, n_signal_tones)`.
- **`averaged_psd_cleaned`** (`noise.py`) — PSD stage following `clean_correlated_streams`: takes its `cleaned` array and returns per-signal-tone PSDs averaged across acquisitions (running mean) as `(f, psd_a, psd_b)`. Mirrors `averaged_psd_timestream` — pass one `Sweep` per signal tone for resonator-basis dissipation/frequency PSDs, else raw I/Q.
- **`plot_iq_comparison`** (`plotting.py`) — Overlays a `TimeStream` I/Q cloud on the smooth fitted resonator sweep circle in the complex plane. Re-fits the `Sweep` internally (via `resonator_tools`) so the smooth `z_data_sim` trace and the calibration parameters (`environmental_term`, `phi0`, `fr`) come from one self-consistent fit, then projects the time stream, sweep trace, and optional QC-trace points into a common `basis` (`"electronic"`/`"fractional"`/`"resonator"`). Renders the cloud via `density` (`"scatter"`/`"kde"`/`"contour"`/`"hexbin"`/`"hist2d"`; `"contour"` gives fast histogram-based 1σ/2σ rings, ~5× faster than the KDE path on large clouds), marks `fr` and `fr ± freq_shift`, colours the sweep trace by detuning, and returns the matplotlib axis. `device`/`power_dbm` feed the auto-title (replacing the previously hardcoded globals).
- **`fold_timestream`** (`folding.py`) — block-average a periodically-driven time stream into a single drive period (the sawtooth-biased "QC trace"): the bias ramp repeats at a fixed rate while the stream records continuously, so averaging in blocks of one period beats uncorrelated noise down by `sqrt(n_periods)`. Accepts a `TimeStream` (uses `signal`, already trimmed by `discard_start_ms`) or a raw complex array; specify the period either as `period_s` or as `n_periods` (exactly one). Leftover samples after the last whole period are dropped. Returns `(time_ms, avg_iq)` with `avg_iq` shape `(2, n_samples)` holding averaged I and Q.
- **`plot_qc_trace`** (`plotting.py`) — plot the `fold_timestream` output as I and Q versus time over one drive period, optionally overlaying the unfolded stream via `raw=` to show what the averaging bought.
- **`MB_fitter`** (`mattis_bardeen.py`) — Mattis-Bardeen superconductor theory fit for temperature-dependent resonant frequency and internal quality factor using `iminuit`.
- Helper functions: `n_qp`, `f_T`, `Qi_T`, `kappa_1`, `kappa_2`, `S_1`, `S_2`, `signed_log10`.

Usage examples are in `daq/analysis/README.md`.

## Documentation

When adding or modifying measurement classes or analysis modules, update the corresponding documentation:

- **Measurement classes** — Document new classes in the Architecture > Measurement Classes section of this file.
- **Analysis tools** — Add usage examples to `daq/analysis/README.md` and update the Architecture > Analysis section of this file.
- **Instrument drivers** — Add usage and SCPI gotchas to `daq/instruments/README.md` and update the Architecture > Instruments section of this file.

## Style Conventions

- Sphinx-style docstrings for all public classes and functions.
- Complete type annotations for public APIs; use `Optional[T]` for optional parameters.
- Black formatting with 100-character line length.
