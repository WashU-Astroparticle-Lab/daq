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

**Two documents are authoritative for anything about the Presto. Treat them as the bible.**

| | |
|---|---|
| **`presto` Python API manual** | https://www.intermod.pro/manuals/presto/index.html |
| **Presto spec sheet (hardware)** | https://intermod.pro/res/docs/presto_spec_sheet.pdf |
| **`presto` package source** | your env's `site-packages/presto/*.py` |

Our hardware is a **Presto-8**. The rules:

1. **Read these before searching externally**, and before reasoning from
   `site-packages/presto/*.py`.
2. **When the documents and an inference from the source disagree, the documents win.** The
   source shows *what the Python layer sends*; it does not show what the FPGA does with it.
   Several wrong conclusions in this repo's history came from reasoning about hardware
   behaviour purely from the wrapper — including a "30 ms trigger pulse" that the hardware
   never emits, and a Pulsed sampling-window figure that was off by exactly 2× until the spec
   sheet settled it.
3. Reading the source is still right for **exact call ordering, argument validation and
   constants** — things the documents do not cover. Cite `file:line` when you do.
4. **If neither document settles a hardware-timing claim, say so and mark it unverified.**
   Do not promote an inference to a fact because it is plausible.
5. **When in doubt, look it up — do not guess.** All three references are cheap to read and
   the failures they prevent are silent ones. An answer marked "unverified" is fine; a
   plausible guess presented as a fact is not.

The spec sheet is a scanned-layout PDF: `WebFetch` cannot read it, but the `Read` tool can
(pass `pages`). Do not conclude it is unavailable.

**On an analysis machine with no `presto` install** (the offline test suites skip there, and
`import daq` fails), unpack the wheel instead of guessing — the source is a plain zip:

```bash
mkdir -p ~/.local/share/presto-source/2.17.1 && cd $_ && unzip -o /path/to/intermod_presto-*.whl
```

The notes under *Established behaviour* below were written against **2.16.0**; check
`presto/version.py` before treating one as current.

Page map for the API manual (relative to `https://www.intermod.pro/manuals/presto/`):

| Topic | Page |
|---|---|
| Digital / trigger ports | `special/digital_ports.html` |
| Direct vs Mixed converter modes | `special/direct_mixed.html` |
| Mixer math, synchronising mixers | `special/mixer.html`, `special/sync_mixers.html` |
| HF outputs, DC outputs | `special/HF_outputs.html`, `special/DC_outputs.html` |
| Phase encoding, DAC config, sequence plots | `special/XY_pulses.html`, `special/recommended_dac_config.html`, `special/plot_sequences.html` |
| API modules | `source/{lockin,pulsed,hardware,dcbias,spectral,utils,ssh,version}.html` |

### Presto-8 hardware, from the spec sheet (2024-07-15)

| | Presto-8 |
|---|---|
| RF outputs | 8× SMA, 14-bit DAC up to 10 GS/s, output power range 0–24 dB |
| RF inputs | 8× SMA, 14-bit ADC up to 5 GS/s, input attenuation 0–27 dB in 1-dB steps |
| IF bandwidth | 1 GHz (−3 dB at 964 MHz) — this is what caps `\|IF\| < 500 MHz` per sideband |
| NCO | 48-bit frequency (40 µHz), 18-bit phase (30 µrad); fully digital IQ mixer |
| **Digital markers / triggers** | **4 inputs, 4 outputs**; input impedance **10 kΩ**; output impedance **50 Ω**; **output 3.3 V**; rise 470 ps (20–80 %) / 820 ps (10–90 %), fall 620 ps / 1060 ps, typical **into 50 Ω** |
| Clock reference | internal TCXO, ±50 ppb; external programmable reference **input up to 750 MHz and output up to 3 GHz, 10 MHz default** |
| HF outputs | 2 ports, 10 MHz – 15 GHz, 8 dBm @ 8 GHz (CW pump tones, no flexible phase sync) |
| DC bias output | 16 channels (4× SMA + 1× DSub-25), ranges 0–3.33 V / 0–6.67 V / ±3.33 V / ±6.67 V / ±10 V, 16-bit, output impedance 1 kΩ, compliance 10 mA |
| Continuous-wave mode | 192 generators and 192 demodulators, distributable between ports |
| Pulse mode | contiguous sampling window **max 524 µs (2¹⁹ samples)**; total sample memory **268 ms (2²⁹ samples)**; FPGA averaging up to 65k (2¹⁶) windows; template matching **128 templates, max length 1 µs**; sequencer **10 736 events**, **event time resolution 2 ns**; feedback latency 184–254 ns |
| Comms / power | Gigabit Ethernet; 100–250 V, 50–60 Hz; 2 U rack, 6 kg |

Two consequences worth keeping in mind. The **digital outputs are 3.3 V into 50 Ω** — driving
an unterminated high-impedance input (the DC2200's TTL input is 10 kΩ) can roughly double
that, above a 5 V TTL maximum, so terminate at the far end. And the **external clock reference
output (10 MHz default)** means any external generator with a 10 MHz reference input can be
locked to the Presto, removing free-running-crystal drift between the two.

### Established behaviour, from the API manual and `presto` 2.16.0 source

Provenance is marked because it determines how much weight a claim carries.

- **`Lockin`'s only digital control is `set_trigger_out`**, and it is *window-locked*: the
  trigger re-asserts at the start of every lock-in window, i.e. at rate `df`. One state per
  port (`0` off, `1` every lock-in window, `2` every sum window), packed two bits at a time
  into a uint8, so at most four ports. `delay`/`width` are **global, not per port** — a single
  pair sent alongside `df` in `Msg__LckSetDf` — so ports enabled together are gated
  identically. (source)
- **Consequence: a `TimeStream` cannot make the Presto emit a short pulse at a slow rate.**
  Width ≥ `1/df` holds the line high for the whole acquisition; width < `1/df` chops it into a
  pulse train at the sample rate. There is no width that yields one short pulse in a long
  record. The sum-window divider (`state=2`) does not rescue this either: `nsum` is capped at
  1023, so a 100 ms repetition needs `df ≤ ~10 kHz`, which no longer resolves a 0.1 ms
  feature. Getting a genuinely short, acquisition-locked pulse needs either an external pulse
  generator gated by the trigger line, or `presto.pulsed`. (source)
- **`output_digital_marker` is `Pulsed`-only** — arbitrary duration at a defined time in a
  sequence, ports 1–4, at the 3.3 V / 50 Ω levels above. (manual + spec)
- **Digital inputs only start a `Pulsed` sequence.** They are reachable exclusively as
  `TriggerSource.DigitalIn1..4` passed to `Pulsed.run()`, which reaches the hardware via
  `hardware.set_run()` — called *only* from `pulsed.py`. There is **no API to read a digital
  input**, and a `Lockin` acquisition cannot be armed from one: it arms with `Msg__LckGet` +
  `Msg__RfdcSync` and never calls `set_run`. So "let the external hardware free-run and trigger
  the acquisition off its marker" is not available to `TimeStream`. (source)
- **A `Lockin` acquisition starts inside `get_pixels()`, not at `apply_settings()`.**
  `apply_settings` starts signal *generation*; `_get_pixels_common` sends `Msg__LckGet`, calls
  `hardware.sync()`, then blocks on an 8-byte server handshake, and the pixel window aligns to
  the next SYSREF edge. This is why `TimeStream.run(on_acquire=)` genuinely runs before sample
  zero — and equally why its offset is a software/network latency over Gigabit Ethernet, not a
  hardware one, so it cannot be trusted below the ~10 ms scale. (source)
- **`Pulsed` is windowed, not continuous, and cannot replace `TimeStream` for multi-tone
  streams.** The contiguous window is 524 µs at full rate (spec sheet; note `MAX_STORE_LEN` in
  the source is 2²⁰ = 1 048 576, twice 2¹⁹, because it counts I and Q separately — trust the
  spec sheet). Because a store holds *raw* samples, window length and usable tone span trade
  off against each other at a fixed budget, scaling with `Pulsed`'s `downsampling` argument `D`:

  | `D` | `fs` | tone span | contiguous window |
  |---|---|---|---|
  | 1 | 1000 MS/s | ±500 MHz | 524 µs |
  | 8 | 125 MS/s | ±62.5 MHz | 4.2 ms |
  | 32 | 31.2 MS/s | ±15.6 MHz | 16.8 ms |
  | 128 | 7.8 MS/s | ±3.9 MHz | 67 ms |

  So covering a 100 ms period needs `D ≈ 256`, confining every tone to ±2 MHz, while the
  ±500 MHz spread `TimeStream`'s `is_usb` machinery exists to exploit allows only a 524 µs
  window. `Lockin` has no such coupling because the FPGA demodulates *before* storage — it
  keeps `n_tones` complex numbers per `1/df` instead of raw samples, a reduction of
  `fs / (n_tones × df)`, order 10⁴ for two tones at `df = 50 kHz`. **That reduction is the only
  reason a multi-tone multi-second stream is possible at all.** (spec + source)
- **`Pulsed`'s hardware demodulator is template matching, and it is sized for qubit readout,
  not for streaming.** `setup_template_matching_pair` + `match()` + `get_template_matching_data`
  compute an inner product on the FPGA, but templates cap at 1 µs (128 template slots per the
  spec sheet; `MAX_TEMPLATE_LEN = 2044` points, halved in Mixed mode, longer ones split across
  slots) and each `match()` yields **one complex number per event** against a whole-sequence
  budget of 10 736 events. A 5 s × 50 kHz single-tone stream would need 250 000 match events —
  23× the entire sequencer capacity. Do not propose template matching as a lock-in substitute.
  (spec + source)
- **`CLK_T` is configuration-dependent**, not a fixed board constant: 2.5 ns (`CLK_F` 400 MHz)
  when `adc_fsample` is `G3_2`, otherwise 2 ns (500 MHz). Any timing figure quoted in clock
  counts — e.g. the 24-bit trigger-width ceiling — moves with it. (source)
- **`Sweep`'s explicit DAC config and `TimeStream`'s automatic one agree.** The sweep classes
  pass `recommended_dac_config(freq_center)`; `TimeStream` passes bare `DC_PARAMS`
  (`dac_mode=Mixed`, no `dac_fsample`), which is the manual's documented *automatic* path
  since API 2.12.0. They are not two different configurations: `hardware.py:995` routes the
  automatic path through `utils.compatible_dac_configs()`, whose candidate list
  (`G6`/`G8`/`G10` × `Mixed02`/`Mixed04`/`Mixed42`) and `spur_score` are **identical** to
  `utils.recommended_dac_config`'s — the code is duplicated verbatim — and `hardware.py:1832`
  takes the top-scored entry the same way. So the two select the same `(dac_mode,
  dac_fsample)` at the same frequency, and a `Sweep` and a `TimeStream` a few hundred kHz
  apart cannot land on different DAC settings (the score is a spur-distance measure on a
  MHz-to-GHz scale). **This was checked because a gain/phase mismatch between the two would
  present exactly as a time-stream cloud sitting off a fitted resonance circle** — it is not
  that. One real difference remains: `_autoconfig` *intersects* candidates across ports
  sharing a DAC tile (`hardware.py:1821`), so a multi-port acquisition can be pushed onto a
  compromise config that a single-port one would not choose. (source, 2.17.1)

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

There is no CI. An offline verification suite lives in `tests/` and runs with **no hardware, no VISA runtime and no MongoDB**:

```bash
python tests/test_instruments.py     # daq.instruments vs a simulated VISA backend (no VISA runtime needed)
python tests/test_timestream_run.py  # TimeStream.run() ordering/muting vs a mocked Lockin (skips if presto absent)
python tests/test_resonator.py       # daq.analysis.resonator vs a synthetic resonator (skips if resonator_tools absent)
python tests/test_qc_trace.py        # QCTrace trigger routing and folding vs a mocked TimeStream (skips if presto absent)
python tests/test_bias_hunt.py       # BiasHunt ungated acquisition and contrast ranking vs a mocked TimeStream (skips if presto absent)
python tests/test_plotting.py        # plot_iq_comparison's single-frequency normalisation vs a synthetic resonator, and plot_psd's bases/panels/per-panel fits (no optional deps)
```

Each prints one PASS/FAIL line per check and exits non-zero on failure. They are standalone scripts, not pytest suites — `tests/test_instruments.py` must inject its fake `pyvisa` before importing `daq.instruments` (and still needs `presto` importable, since `daq/__init__.py` pulls in the measurement classes; the drivers themselves have no presto dependency), and `tests/test_resonator.py` loads `daq/analysis/resonator.py` by path (bypassing `daq/__init__.py`, which pulls in `presto`) so the analysis layer stays verifiable on a machine with no presto install. `tests/test_plotting.py` does the same for `daq/analysis/plotting.py`, but under a synthetic parent package so the module's relative `from .resonator import ...` still resolves; it needs neither `presto` nor `resonator_tools` and so is the one suite that always runs (its `plot_psd` fit checks skip themselves if `iminuit` is absent, and it forces matplotlib's `Agg` backend so nothing opens a window). Run them as scripts. Extend them when touching the instruments layer, `TimeStream.run()`, `QCTrace`'s trigger routing, `BiasHunt`'s ranking, the resonator fitting adapter, or the basis transformations and PSD plotting in `plotting.py`. Note both measurement suites swap `TimeStream` in `daq/measurements/_gate_bias.py`, not in the measurement's own module — that is where the shared readout builder lives.

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
| `DAQ_FGEN_TRIGGER_PORT` | `None` (driver default: port 1) |
| `DAQ_LED_TRIGGER_PORT` | `None` (driver default: port 2) |
| `DAQ_VISA_BACKEND` | `""` (system NI-VISA) |

Settings are cached after first access via `get_settings()`. Call `reload_settings()` to pick up environment changes.

## Architecture

### Measurement Classes (`daq/measurements/`)

All measurement classes inherit from `Base` (`daq/_base.py`). Each runs a hardware acquisition via the `presto` library, optionally fits the data, saves results to HDF5, and logs metadata to MongoDB.

- **`Sweep`** — Single-tone frequency sweep. Auto-fits resonator parameters (fr, Qi, Qc, Ql, kappa) via `daq.analysis.resonator.fit_notch` (see Analysis), so `fit_results` also carries the environmental term.
- **`TimeStream`** — Time-domain multi-tone acquisition. Supports multiple IF frequencies with per-tone USB/LSB sideband selection via the `is_usb` bool array (`True` → `LO + IF`, `False` → `LO - IF`; default all-USB). `amp` accepts a per-tone array or a single scalar that is broadcast to every tone (equal drive); a scalar is *not* split across tones. This broadcast guards a presto footgun — `set_amplitudes` drives only tone 0 and silently zeroes the rest when given fewer amplitudes than tones — and `check_amp` validates both that `amp` length matches `if_freqs` and that the per-tone sum stays below DAC full scale (`< 1.0`). Single-sideband output phases are derived automatically. Each `|IF| < 500 MHz`, so centering the LO between tones lets two tones up to 1 GHz apart be read at once. After `run()`, `signal`/`signal_freqs` give the per-tone selected-sideband data and physical frequencies. Phase reset is gated: disabled for non-zero IF. `discard_start_ms` (default `25.0`, set `0` to disable) drops that many leading milliseconds of startup junk from the in-memory time-axis arrays (`signal`/`usb`/`lsb`/`pixel_i`/`pixel_q`) after `run()` and `load()`, using `round(discard_start_ms * 1e-3 * df)` samples at the tuned rate; the saved HDF5 keeps the full acquisition. This trimming lives on `TimeStream` alone — `averaged_psd_timestream` forwards the field rather than re-implementing it. `run(on_acquire=...)` takes an optional callable invoked once after `apply_settings()` (any configured trigger already asserted) and immediately before `get_pixels()` — the place to software-start side hardware (e.g. a DC2200 pulse train) so it lands ms-scale from sample zero instead of the seconds-scale, per-run-variable offset of anything started before `run()`; the offset's sign and size are not yet bench-measured, so read them off the data (first recorded pulse position modulo the pulse period). The hook is a `run()` argument, not an attribute, so it never reaches HDF5/MongoDB; if it raises, the acquisition is abandoned, the Presto outputs are muted on the way out (the mute runs in a `finally`, also covering `get_pixels` failures), and the exception propagates. `external_trigger` selects **which Presto digital output ports** assert a trigger, since presto's `set_trigger_out` takes one state per port (element *i* → port *i+1*; `0` off, `1` every lock-in window, `2` every sum window): `False` triggers nothing, `True` is shorthand for `[1]` (port 1, where the 33220A gate is wired), `[0, 1]` gates port 2 (the DC2200 modulation input) and `[1, 1]` fires both. Prefer `daq.triggers.trigger_for(bias, led)` over writing those numbers into a measurement — it reads each instrument's `trigger_port`, so a rig wired differently gates the right hardware unedited (see Instruments). `resolve_trigger_states()` (now owned by `daq/triggers.py`, whose module body imports only numpy so the instrument layer can share the port semantics without depending on `presto`; `TimeStream.resolve_trigger_states` is a thin alias) normalises the argument and rejects out-of-range states, non-integral states (a state computed as `0.999…` would otherwise silently disable the port) and more than four ports (presto packs the states two bits at a time into a uint8). Stored — and round-tripped through HDF5/MongoDB — as the resolved states array, with `load()` still accepting the older scalar-bool files. The constructor argument is backward compatible, but the *attribute* is now an array: test it with `.any()`/`.size`, since a bare `if ts.external_trigger:` raises on an empty or multi-element array. The MongoDB field likewise changed from a bool to a list, so a `select_runs(external_trigger=...)` filter matches only records of the matching vintage. Two properties of the presto layer shape how this can be used: the `delay`/`width` pair is **global**, not per port, so ports enabled together are gated identically; and the trigger re-asserts at the start of every lock-in window, so `TRIGGER_WIDTH_S = 0.03` (≫ `1/df`) leaves the line **high for the whole acquisition** rather than emitting a 30 ms one-shot. A width below `1/df` would chop it into a pulse train at the sample rate; a genuine short pulse needs `presto.pulsed`'s `output_digital_marker`, not this class. (The continuous-high behaviour is inferred from the presto implementation and is consistent with the archived bench routine `QCTrace` packages having taken usable QC traces under this configuration; `QCTrace` itself is not yet bench-run, so it is not independent evidence, and none of this is scope-verified.)
- **`SweepPower`** — 2D sweep over frequency × drive power. Accepts `device`/`filter`/`notes` for database logging (required by `_save`, like `Sweep`). With `auto_fit=True` (default), `run()` fits the resonator once per drive amplitude (via `daq.analysis.resonator.fit_notch`, centered on each row's amplitude minimum and cropped to half the span) and stores the per-amp `fitresults` dicts as a list in `self.fit_results` (`None` for any amp whose fit fails). This list lives on the object only — it is not written to HDF5 or MongoDB. An optional `attenuation_db` (default `None`) shifts the plotted drive-power axis to the device input (`drive power - attenuation_db`, labelled "Drive power at device"). `analyze()` shows the response-amplitude map plus the best-fit `fr` and `Qi` (diagonally corrected) versus drive power, each with fit-error bars (`fr_err`, `Qi_dia_corr_err`); it lazily fits via `_perform_fit()` when `fit_results` is unset (e.g. after `load()`).
- **`SweepFreqAndDC`** — 2D sweep over frequency × DC bias (JPA modulation curves).
- **`TwoTonePower`** — Two-tone spectroscopy: pump power/frequency vs. fixed probe frequency.
- **`QCTrace`** — Quantum-capacitance trace: a gated gate-voltage ramp, folded into one period. Drives no Presto hardware of its own — it composes one `TimeStream` and an `Agilent33220A` gate bias. `run()` puts the generator into a **gated** sawtooth (`ramp_vpp`/`ramp_freq_hz`/`ramp_offset_v`/`ramp_symmetry_pct`), acquires an externally-triggered `TimeStream` spanning `num_periods` whole ramp periods (`Agilent33220A.samples_for_periods` sizes it), and calls `fold()`. **It does not locate the resonance**: pass `readout_freq` (normally the `fr` from a separately-run `Sweep` with `auto_fit=True`). `fold(stream=None, *, period_s=None, n_periods=None, tone=0)` is a method wrapping `fold_timestream` that knows the ramp period and sample rate, so the usual call is a bare `qct.fold()`; it defaults to `period_s=1/ramp_freq_hz` at the *tuned* `df` rather than dividing the record by `num_periods`, so a slightly shifted `df` drops a leftover sample instead of smearing the average across blocks. It is re-callable — on a different period, or on a stream reloaded from `qc_file` (`qct.fold(TimeStream.load(qct.qc_file))`) — and each call replaces `time_ms`/`avg_iq`. It requires an object carrying `df` (the tuned rate) and raises `TypeError` on a bare array: substituting the *requested* `sampling_frequency` would fold at the wrong rate, which is the failure folding on `stream.df` exists to prevent — call `fold_timestream(array, fs, …)` directly for that. `num_periods_folded` records the blocks actually averaged — the `N` in the `sqrt(N)` noise gain — which falls below `num_periods` whenever the record does not divide evenly, and is not otherwise recoverable from the file. **Folding cuts the record into `round(period_s × fs)`-sample blocks, so a `sampling_frequency` that is not a whole multiple of `ramp_freq_hz` makes every block start a fraction of a sample late and the error accumulate**: at 50 kHz with a 300 Hz ramp (166.67 samples/period) a sharp feature loses ~90% of its contrast over 200 periods, and the resulting trace looks unremarkable. `__init__` warns when the ratio is non-integral; the cure is picking a sample rate that divides evenly. (This is longstanding behaviour of `fold_timestream`, not new here — the warning is.) The acquisition is gated by default on whichever port the bias generator says it is wired to (`Agilent33220A.trigger_port`, port 1 by default) — re-read on **every** run, so a rewired rig needs no change here and fixing a port then re-running the same object gates the new one. `trigger_states=` overrides the routing for one measurement and is validated in `__init__`; a routing that gates **no** port is refused outright (an ungated ramp sits at its burst start level and silently records a static bias) and one that leaves the generator's own port unasserted warns. The resolved states are printed at the start of the run and saved to HDF5/MongoDB; `load()` restores them for inspection but does not pin them (re-running a loaded measurement re-reads the generator). The `TimeStream` saves its own HDF5/MongoDB record via the normal path with `attach(bias=...)` applied, so the raw acquisition stays individually loadable; `QCTrace` saves one further `qc_trace` record holding `time_ms`/`avg_iq`, `num_periods_folded`, the resolved `trigger_states` and `qc_file`. The stream hangs off a read-only `qc_stream` property so `Base._save` skips it; `load()` restores the derived record but not the stream. The bias output is forced off on exit — including on exception — and a caller-supplied `bias` instrument is de-energised but never closed. `analyze()` draws the folded I/Q trace via `plot_qc_trace`.
- **`BiasHunt`** — The companion to `QCTrace`: find the gate bias with the largest charge-parity contrast. Parks the generator at each entry of `bias_voltages` via `Agilent33220A.constant` and records one `ts_duration_s` `TimeStream` per voltage at `readout_freq` (again caller-supplied — no `Sweep`), ranking them by `std(|signal|)`, exposed as the static `parity_contrast_of(stream)`. Every acquisition is **ungated** — the bias is a DC level already written over SCPI, so there is nothing for a trigger to start, and asserting one would gate whatever else is on that port for the whole hunt. This is the exact inverse of `QCTrace`, and the two failure modes are symmetric, which is why each class's offline suite asserts its own. The voltages are drawn (uniform over `[v_min, v_max]`, `seed`-able) in `__init__`, so the object and its saved record fully pin down what was measured before hardware is touched; pass `bias_voltages` explicitly (e.g. a `linspace`) to scan instead of sample. **There is no default gate range**: omitting `v_min`/`v_max` without `bias_voltages` raises rather than putting an invented voltage on a device; passing both, `v_min`/`v_max` are ignored and the attributes report the explicit list's span. Every result is cleared at the top of `run()`, so a re-run that fails part-way cannot leave the previous run's `parity_contrast` beside this run's shorter `bias_streams` — `best_bias_stream` indexes one by the argmax of the other, and the mismatch would report a new stream under an old bias voltage. Each try saves its own HDF5/MongoDB record with `attach(bias=...)` applied; `BiasHunt` saves one further `bias_hunt` record with `parity_contrast`, `best_bias`, `best_contrast`, `best_bias_file` and every try's path in `bias_files`. `bias_streams`/`best_bias_stream` are read-only properties; `load()` restores the derived record but not the streams. Bias off on exit, including on exception. `average_psd(streams=None, *, quantity="abs", welch=False, …)` averages one `compute_psd` per try into `psd_freqs`/`psd_avg` — each try is short, so its own periodogram is noisy and the mean over `n_bias_try` beats that down by `sqrt(n_bias_try)`. It takes the spectrum of the **mean-subtracted** projection (`"abs"` by default, matching the contrast metric; `"real"`/`"imag"` available), since the parity signal is the fluctuation and not the operating point it sits on, and it uses the streams' **tuned** `df` rather than the requested `sampling_frequency`. `fit_psd(**kwargs)` fits that average with `fit_parity_psd` (`f_bw` held at the tuned rate, `kwargs` forwarded — `fit_onef=True` for a 1/f-like rise) into `fit_results`. **Averaging across operating points is the intended use**, not a compromise: the switching rate is expected to be common across the gate range (set by quasiparticle dynamics) while the telegraph *amplitude* is what varies with bias (through the quantum-capacitance slope), and the model is linear in that amplitude — so the average is the same Lorentzian with `sigma^2` replaced by its mean, leaving `gamma_p` intact and only improving precision. Simulated over 20 realisations of six tries spanning 100× in amplitude, averaging all six and averaging only the top three by contrast agree (130.2±7.6 vs 130.8±7.7 Hz) and both beat one stream by 2.3× (146.9±17.2 Hz, `sqrt(6)` expected); low-contrast tries cost nothing. Two caveats survive: the fitted `fidelity` is **diluted** by averaging (it is the Lorentzian-to-floor ratio, and a dead try adds floor without signal — read it as an ensemble property, not the readout fidelity at the best point), and `gamma_p` **reads ~10% high on short records** (6×2 s; ~5% at 4M samples, 0.4% on an exact analytic spectrum — a log-periodogram small-sample effect, not a fitter defect). If the rate genuinely varies with bias the average is a mixture of Lorentzians; fit tries individually via `hunt.average_psd([stream])` to check. Re-averaging clears a stale `fit_results`, so a fit never describes a spectrum it did not see. Both are lazy, cached, and — like `SweepPower`'s per-amp fits — live on the object only: they derive from streams the saved record does not contain, and `Base` skips `fit_results` by name anyway (this one holds a live `iminuit` object). `analyze(psd=True, fit=True, title=None, **fit_kwargs)` draws parity contrast versus gate bias with the winner marked, above the averaged spectrum on log-log axes with the fit overlaid and annotated (`Γp`, `f_corner`, `F`, `resid_dex_rms`). The spectrum panel plots the raw periodogram faintly, the **log-binned points the fit actually used** as markers, and frames the y-axis on those — the bare periodogram scatters over ten decades and reads as a bad fit even when it is a good one. An existing `fit_results` is reused, so `analyze()` after an explicit `fit_psd(fit_onef=True)` draws that fit. A fit that raises is reported on the panel rather than costing the spectrum, and a measurement with no streams (i.e. after `load()`) silently degrades to the contrast panel alone with a note on how to get the spectrum back.

### Base Class (`daq/_base.py`)

Provides `_save()` (writes HDF5 + inserts MongoDB document) and `_build_document()`. Defines hardware constants: `DAC_CURRENT = 40_500` μA, `ADC_ATTENUATION = 0` dB.

`attach(**instruments)` records auxiliary instrument state on a measurement: each instrument's read-back `settings()` mapping is flattened onto the object as `<prefix>_<key>` scalar attributes, which `_save()`/`_build_document()` then pick up for free (no per-instrument code in `_base.py`). Call it before `run()` so bias voltages and LED parameters land in HDF5 and MongoDB — and become `select_runs()`-queryable — instead of surviving only inside a `notes` string. Accepts anything with a `settings()` method, or a plain dict; `None` values are skipped since they have no HDF5 representation.

### Instruments (`daq/instruments/`)

Drivers for non-Presto benchtop hardware reached over VISA/SCPI. These are **instruments, not measurements** — they produce no data arrays and do not subclass `Base`; you compose them with the ordinary measurement classes in a notebook and record their state via `Base.attach()`. Deliberately almost no per-experiment measurement classes: improvised combinations stay in user code. The exceptions are `QCTrace` and `BiasHunt`, where the bias state the acquisition runs under *is* the measurement and getting it wrong — a gated ramp with no port asserted, a trigger asserted during a DC hunt — silently produces plausible-looking wrong data. They stop there: neither locates its own resonance, and neither sequences the other.

- **`VisaInstrument`** (`_visa.py`) — shared base. Lazy `pyvisa` import (so `import daq` works with no VISA runtime); resource resolution by explicit argument → env var → autodiscovery filtered on `*IDN?`, **raising** on zero or multiple matches rather than taking `list_resources()[0]`. Discovery needs no configuration in the normal case — the rule is one match *per model*, so unrelated instruments on the same computer are simply ignored; each driver's `RESOURCE_HINTS` (USB VID/PID) further narrows probing so unrelated devices are not opened at all, falling back to probing everything if nothing matches the hint. `probe_visa_resources()` lists every visible resource with its `*IDN?` (or the error explaining why it did not answer) and is the first thing to run when something will not connect; `visa_backend_info()` names the loaded VISA library, which distinguishes a vendor VISA from the pure-Python backend (the latter cannot see USB instruments without `pyusb`/`libusb`). Failed discovery reports the same per-resource detail, so a model mismatch, a busy resource and a timeout are distinguishable — and when no USB/GPIB resource is visible at all it says outright that the likeliest cause is an unplugged or unpowered instrument, which is the most common failure by a wide margin. every write bracketed by a `SYST:ERR?` drain and check that raises `InstrumentError`; context manager whose `__exit__` always forces `safe_state()`; optional `transcript_path` SCPI log. Subclasses set `IDN_KEYWORDS`/`RESOURCE_HINTS`/`TRIGGER_PORT`, override `env_resource()`, `env_trigger_port()` and `safe_state()`, and extend `settings()`. Each instrument also carries **`trigger_port`** — the Presto digital output port its gate/modulation input is wired to, resolved explicit argument → env var → class default and validated to 1–4. That makes the wiring a property of the bench rather than of every measurement: `daq.triggers.trigger_for(bias, led)` builds the per-port states a `TimeStream` wants (`[1]`, `[0, 1]`, `[1, 1]`), and `settings()` reports `trigger_port` so `attach()` records which port was expected to gate what. This exists because the failure is silent — a gated instrument on an unasserted port simply never fires and the data looks like a dead detector.
- **`Agilent33220A`** (`function_generator.py`) — gate-bias source. `constant(offset_v)` for DC; `sawtooth(vpp, freq_hz, ...)` for a ramp, gated on the Presto trigger by default (pair with `TimeStream(external_trigger=trigger_for(bias))`, which asserts the generator's own `trigger_port` — 1 in the lab's wiring, overridable per instrument or via `DAQ_FGEN_TRIGGER_PORT`), with `offset_v` defaulting to `vpp/2` (unipolar-positive). Both setters are **hermetic** — each writes the full state its mode depends on, and `constant()` disables burst *before* selecting the DC carrier, since the 33220A rejects that combination. `samples_for_periods(n_periods, sample_rate)` returns the `pixel_counts` spanning whole ramp periods plus the `TimeStream` discard window, keeping generator and acquisition in step. Amplitudes below the instrument minimum (20 mVpp into high-Z, 10 mVpp into 50 Ω) raise.
- **`DC2200`** (`dc2200.py`) — Thorlabs LED driver. Every SCPI header comes from the DC2200 Operation Manual v1.8 (29-Nov-2023) §4.3.2, so nothing is guessed. The instrument offers seven modes; four are wrapped, distinguished by what generates the timing:
  - `configure_cc(current_a)` — steady current (`SOURCE1:CCURENT:CURRENT`, Thorlabs' own misspelling, required verbatim).
  - `configure_pwm(current_a, freq_hz, duty_pct, count)` / `pwm_train(...)` — burst defined by a **duty cycle**, amplitude in **amps**. `pwm_train` is the blocking configure → on → wait → off form; `configure_pwm` returns immediately. Instrument ranges: freq `0.1 Hz–20 kHz`, duty `0.1–99.9 %`, count `1–1000` or `0` = infinite. The **0.1 % duty floor** couples width to rate — the narrowest PWM pulse is `0.001 / freq_hz`, i.e. 10 µs only at 100 Hz, 500 µs at 2 Hz.
  - `configure_pulse(on_time_s, off_time_s=|freq_hz=|period_s=, brightness_pct=|current_a=, count)` — burst defined by explicit **ON and OFF times**, so width and rate are independent; this is the only way to get a short pulse at a slow rate (10 µs at 2 Hz). Each time spans `0.001 ms–10 s`, giving `0.05–500 Hz`. Note the amplitude is **brightness as a percent of the configured current limit**, not a current: `current_a` is accepted and converted via the queried limit, but with a high limit a small absolute current is a tiny percentage that the instrument's resolution may not reach — lower the LED current limit for fine control near the bottom.
  - `configure_ttl(current_a)` — LED follows the rear-panel SMA modulation input ("TTL" in the plain digital-logic sense: low `0–0.8 V`, high `2.0–5.0 V`, 10 kΩ), the only mode whose timing is synchronised to an acquisition. Note it yields an illumination **window** spanning the record, not a pulse within it: the Presto re-asserts its trigger at the start of every lock-in window, so with `TRIGGER_WIDTH_S` far above `1/df` the line is high from the first sample to the last. The LED is wired to **digital output port 2** in the lab's default setup (`TRIGGER_PORT = 2`, overridable per instrument or via `DAQ_LED_TRIGGER_PORT`), so gate it with `external_trigger=trigger_for(led)` rather than a hand-written list — a plain `True` gates port 1, the 33220A, and leaves the LED dark for the whole run; `trigger_for(led, bias)` fires both. Set **`discard_start_ms=0`**, since the LED comes up with the RF drive at `t = 0` and the default 25 ms trim would drop the turn-on transient. A short *synchronised* flash needs `presto.pulsed` (not wrapped); `configure_pwm`/`configure_pulse` give real short pulses at a software-start offset instead.

  Configuring is *not* arming: every `configure_*` defaults to `output=False`, so it can never illuminate the LED as a side effect. In TTL mode, enabling the output arms the stage and the LED then follows the input, so arm immediately before `run()` — the idle level of the Presto trigger line between acquisitions is not controlled here and has not been measured. In PWM and pulse mode the train starts on the software write that enables the output, so neither is synchronised to an acquisition. Currents are validated against the queried `SOURCE1:CURRENT:LIMIT?`. Note the DC2200 has **two output terminals** (LED1 10 A/12-pin, LED2 2 A/4-pin) and every source setting applies to the selected one — the `terminal` property reads and sets it rather than silently inheriting the front-panel choice. `protection_status()` reports the current-limit, interlock and driver/head over-temperature trip flags, which shut the output down independently of the settings. In PWM and pulse mode the SMA connector *outputs* the internal modulation as a TTL signal, so the LED timing can be fed back to the Presto as a marker.

Setup, SCPI gotchas and troubleshooting are in `daq/instruments/README.md`. `pyvisa` is an optional extra (`pip install daq[instruments]`).

### Trigger routing (`daq/triggers.py`)

Owns which Presto digital output port gates which instrument, for both the measurement and the instrument layers (it imports only numpy, so `daq.instruments` can use it on a machine with no `presto`). `resolve_trigger_states()` normalises a `TimeStream(external_trigger=...)` argument to presto's per-port states; `trigger_for(*instruments_or_ports)` builds those states from each instrument's `trigger_port`, which is the form to prefer — a port number written into a measurement is wrong the moment the bench is rewired, and wrong silently. It requires at least one source (an empty `trigger_for()` would be a silently ungated acquisition inside the helper written to prevent them). `validate_trigger_port()` bounds ports to 1–4 and `describe_trigger_states()` renders them for log lines. Exported as `daq.trigger_for` / `daq.resolve_trigger_states`.

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

- **`fit_notch`** (`resonator.py`) — The single entry point for resonator circle fitting; every fit in the repo goes through it (`Sweep`, `SweepPower`, `plot_iq_comparison`). It runs the **stock upstream** `resonator_tools` `notch_port.autofit()` — upstream owns the whole algorithm — and then adds the calibration quantities upstream computes as locals and discards, merging them into the returned port's `fitresults`: `environmental_term` (the complex `a·e^{iα}·e^{-2πifτ}` of Eqn. 1), `environmental_baseline`, and the scalars `environmental_amp_norm`/`environmental_alpha`/`environmental_delay`/`environmental_A2`/`environmental_frcal`. The return value is a genuine `notch_port`, so `z_data_sim`, `f_data` and the usual keys (`fr`, `Ql`, `absQc`, `Qi_dia_corr`, `phi0`, errors) behave exactly as before. **This module exists to kill a latent bug**: `environmental_term` is *not* an upstream key, so `plot_iq_comparison` and `from_elec_to_reson` used to `KeyError` on any stock install — the repo silently required a private fork (`FaroutYLq/resonator_tools`) that was itself 50 commits behind upstream, and nothing in the repo declared or checked that. The fork turned out to be unnecessary: `notch_port.do_calibration()` is public API returning `(delay, amp_norm, alpha, fr, Ql, A2, frcal)`, i.e. every value the fork saved, so `fit_notch` just re-runs that deterministic calibration and rebuilds the term analytically. Two upstream facts make this exact rather than approximate: `get_delay()` pins `A2 = 0.0` in *both* branches of its `ignoreslope` test (so the baseline is identically zero), and `do_normalization()` therefore reduces to `z_norm = z_raw / environmental_term`. `fit_notch` **checks both on every call** — it raises `ResonatorFitError` if `A2` is ever non-zero, and if the recovered term fails to reproduce upstream's normalization — so a future change to the calibration convention surfaces loudly instead of silently corrupting the basis transformation. The `A2` check is the load-bearing one: the consumers divide by `environmental_term` alone and ignore `environmental_baseline`, so validating only the normalization identity would let a non-zero baseline through unnoticed. Verified bit-for-bit identical (relative drift `0.0`) to the old fork's values, so migrating changes no previously published number. Companion helpers: `readout_environmental_term(fit, readout_freq, ...)` is the **single owner** of "which environmental term does single-frequency data need" — `plot_iq_comparison` and `from_elec_to_reson` both go through it, since two independent implementations of the same transform is how they drifted apart in the first place; it rebuilds the term analytically from the fit's own scalars (so an `f_ro` outside the swept span is still exact), validates the frequency via `validate_readout_freq` (rejecting `nan`, which would otherwise pass a bare `<= 0` test and poison everything downstream), and owns the on-resonance fallback warning. `environmental_term(freq, amp_norm, alpha, delay)` evaluates the term standalone, and `resonator_tools_available()` gates the optional dependency (fits are skipped, not fatal, when it is missing).
- **`compute_psd`** (`noise.py`) — Noise PSD for real-valued time series (1-D or 2-D). Uses the bare periodogram by default; pass `welch=True` for Welch's method (`scipy.signal.welch`).
- **`averaged_psd_timestream`** (`noise.py`) — Wraps `TimeStream` for the "take data, then show averaged PSD" workflow: builds a multi-tone `TimeStream`, runs it `num_averages` times (each run saved as usual), computes a per-tone PSD each time, and returns the running-mean-averaged PSDs `(f, psd_a, psd_b, streams)` with `psd_*` shape `(n_tones, n_freqs)`. Pass one fitted `Sweep` per tone via `sweeps` to get resonator-basis dissipation/frequency PSDs (via `from_elec_to_reson`); otherwise returns raw I/Q PSDs. It passes each tone's own `signal_freqs[ch]` into that transform automatically — the stream knows its tone frequencies, so there is nothing here for a caller to get wrong. `discard_start_ms` (default `25.0`) is forwarded to `TimeStream`, which owns the leading-junk trim (see above); both the PSD input and the returned in-memory `TimeStream` arrays reflect the trimmed window, while the saved HDF5 keeps the full acquisition.
- **`parity_psd_model`** / **`fit_parity_psd`** (`noise.py`) — Random-telegraph (RTS) parity-timestream PSD model and fitter for Eqn. 18 of arXiv:2601.16261: `PSD(f) = F² · 4Γp/((2Γp)² + (2πf)²) + (1-F²)/f_bw`, a switching-rate Lorentzian plus a fidelity-limited white floor. `fit_parity_psd` takes the `(f, psd)` output of `compute_psd` and fits the readout fidelity `F` and parity-switching rate `Γp` (Hz) with the sampling bandwidth `f_bw` held fixed (pass the sample rate, e.g. `TimeStream.df`). The fit is done in **log-log space** (the natural PSD representation): the spectrum is first averaged onto `n_bins` (default 60) logarithmically-spaced frequency bins — geometric-mean frequency, linear-mean power per bin — so every decade is represented equally and the fit is not dictated by the densely-sampled high-frequency structure, then the residual of `log10(PSD)` vs `log10(model)` is minimized. `bin_weighting` sets the weighting: `"uniform"` (default) weights every bin equally so each decade contributes equally (best for Welch/averaged PSDs and for not letting high-f dominate), `"count"` weights by `~1/sqrt(m)` (m = points/bin, statistically optimal, better for a bare periodogram whose low-f bins are sparse); a true per-point `sigma` (on the linear PSD) overrides both. The `f=0` DC bin (and any non-positive frequency) is always excluded (`drop_dc` retained but inert). The fit runs on `iminuit` (`iminuit.cost.LeastSquares` + `MIGRAD`/`HESSE`), so errors come from Minuit's Hesse step. Returns a `fit_results` dict with a best fit and error per term (`fidelity`/`fidelity_err`, `gamma_p`/`gamma_p_err`, `a_onef`/`a_onef_err`, `alpha`/`alpha_err`, `f_corner=Γp/π`/`f_corner_err`) plus `f_bw`, `chi2`, `ndof`, `reduced_chi2` (over the log-binned points), `resid_dex_rms` (RMS log10 data-vs-model residual in decades — weighting-independent goodness-of-fit, `~0.1` good/`~1` bad; prefer it over `reduced_chi2` for flagging bad fits since uniform weighting makes `reduced_chi2` unreliable), `model` (evaluated at every input `f`), the log-binned points fit (`f_binned`, `psd_binned`) and non-empty-bin count `n_bins`, the `minuit` object, and `success` (`Minuit.valid`). Fitting a 1/f-dominated spectrum (e.g. electronic-basis I/Q) with `fit_onef=False` makes the two-term model collapse to the flat white floor — use `fit_onef=True` for 1/f-like low-frequency rise. `parity_psd_model` and the fit take an overall amplitude `sigma^2` (signal variance) scaling the parity term; `fit_parity_psd` fits it by default (`fit_amplitude=True`), which is required for **un-normalized** input (raw electronic I/Q, variance ≪ 1) — Eqn. 18 as written assumes a normalized ±1 signal (variance 1), so with `amplitude=1` its floor `(1-F²)/f_bw ≈ 1/f_bw` is orders of magnitude above a small-variance PSD and the fit collapses (`F→1`, corner out of band). With `sigma²` free, `F` comes from the Lorentzian/floor ratio and `Γp` from the corner (both scale-free). Set `fit_amplitude=False` only for already-normalized data (recovers strict two-parameter Eqn. 18); results carry `amplitude`/`amplitude_err`. Because the default weighting is only relative, errors are rescaled by `sqrt(chi2/ndof)` unless `absolute_sigma=True` (pass a true `sigma` for that). A 2-D `psd` (one PSD per row) is fit row-by-row, returning a list of such dicts. An optional low-frequency `1/f` term `A/f^α` (drift / TLS noise) is added with `fit_onef=True` (amplitude `A` becomes free; exponent `α` fixed at `alpha`, default `1.0`, or fit too with `fit_alpha=True`); held-fixed 1/f terms report `nan` errors (`a_onef=0` when disabled). `parity_psd_model` gains matching `a_onef`/`alpha` kwargs (default `0`/`1.0`, i.e. pure Eqn. 18) and guards the `f=0` divergence.
- **`from_elec_to_reson`** (`noise.py`) — Transform raw I/Q time-stream data from electronic to resonator basis using a fitted Sweep. Takes the same **`readout_freq`** as `plot_iq_comparison` (`TimeStream.signal_freqs[i]` for tone *i*), and for the same reason — both now share `daq.analysis.resonator.readout_environmental_term`, which is what keeps them from drifting apart again. Here the uncorrected rotation `θ = 2πΔτ` **mixes the two returned axes, asymmetrically**: since `rad = Re(tsz)/q` but `arc = Im(tsz)/(-2q)`, `rad_bad = cosθ·rad − 2sinθ·arc` and `arc_bad = cosθ·arc + (sinθ/2)·rad`, so the dissipation channel picks up `4sin²θ` of the frequency channel's power while the frequency channel picks up only `sin²θ/4` of the dissipation channel's — a factor of 16 apart. Both are second order in `θ`, but each is multiplied by the *ratio* of the two noise powers, and the point of splitting them is that one usually dominates: with frequency noise 20 dB above dissipation noise, `θ = 0.094 rad` makes the dissipation PSD read ~4.5× high. `None` keeps the historical `fr` behaviour and warns.
- **`remove_correlated_noise`** (`noise.py`) — Subtract correlated electronics noise (gain drift, LO phase noise) using an off-resonance reference tone. Implements Eqn 7.44–7.45 from Wen (2025) in the gain / arc-length basis.
- **`clean_correlated_streams`** (`noise.py`) — Batch wrapper that applies `remove_correlated_noise` across a list of `TimeStream` acquisitions (e.g. the `streams` from `averaged_psd_timestream`) whose tones are interleaved as `[signal, reference, ...]`. Defaults to pairing even-indexed signal tones with odd-indexed reference tones (override via `signal_indices`/`reference_indices`), and returns only the cleaned signal tones as `(cleaned, freqs)` with `cleaned` shape `(n_streams, n_samples, n_signal_tones)`.
- **`averaged_psd_cleaned`** (`noise.py`) — PSD stage following `clean_correlated_streams`: takes its `cleaned` array and returns per-signal-tone PSDs averaged across acquisitions (running mean) as `(f, psd_a, psd_b)`. Mirrors `averaged_psd_timestream` — pass one `Sweep` per signal tone for resonator-basis dissipation/frequency PSDs, else raw I/Q. Unlike that function it holds a bare array rather than the streams, so it takes **`readout_freqs`** (one per signal tone) to normalise each tone at its own frequency; this is exactly the `freqs` `clean_correlated_streams` already returns beside `cleaned`, so the usual call passes it straight through. A length mismatch raises rather than silently normalising tones at each other's frequencies; `None` keeps the historical `fr` behaviour and warns.
- **`plot_iq_comparison`** (`plotting.py`) — Overlays a `TimeStream` I/Q cloud on the smooth fitted resonator sweep circle in the complex plane. Re-fits the `Sweep` internally (via `daq.analysis.resonator.fit_notch`) so the smooth `z_data_sim` trace and the calibration parameters (`environmental_term`, `phi0`, `fr`) come from one self-consistent fit, then projects the time stream, sweep trace, and optional QC-trace points into a common `basis` (`"electronic"`/`"fractional"`/`"resonator"`). Renders the cloud via `density` (`"scatter"`/`"kde"`/`"contour"`/`"hexbin"`/`"hist2d"`; `"contour"` gives fast histogram-based 1σ/2σ rings, ~5× faster than the KDE path on large clouds), marks `fr` and `fr ± freq_shift`, colours the sweep trace by detuning, and returns the matplotlib axis. `device`/`power_dbm` feed the auto-title (replacing the previously hardcoded globals). **`readout_freq` is the frequency the cloud and the QC points were acquired at** (`TimeStream.signal_freqs[i]` for tone *i* — which already accounts for that tone's USB/LSB choice and equals `lo_freq` only at zero IF; `QCTrace.readout_freq`, whose stream is zero-IF by construction) and matters more than it looks: the sweep spans frequencies and is normalised point by point, but single-frequency data needs the environmental term evaluated *at that frequency*, and the term carries the cable delay `e^{-2πifτ}`. Normalising data taken at `f_ro` by `env(fr)` leaves `S21·exp(-2πi(f_ro-fr)τ)` — a rigid rotation of the cloud about the origin, which moves it **off** the fitted circle rather than along it, by roughly `4πΔτ/(Ql/|Qc|)` ring radii while `2πΔτ ≪ 1` (≈1.9 radii for `Δ = 300 kHz`, `τ = 50 ns`, circle diameter `Ql/|Qc|` = 0.1; under 0.2 on a deep dip). Detuning alone can never leave the circle — the circle *is* the locus over detuning — so an off-ring cloud is a normalisation mismatch, not physics; and because a large rotation can land the cloud near a *different* arc, looking on-ring is not evidence of correct normalisation. The term is rebuilt analytically from the fit's own scalars rather than interpolated, so an `f_ro` outside the swept span is still exact. Omitting it keeps the historical behaviour (normalise at `fr`) and warns in the `"fractional"`/`"resonator"` bases with the fit's own sensitivity; `"electronic"` never divides the term out, so it stays silent there.
- **`fold_timestream`** (`folding.py`) — block-average a periodically-driven time stream into a single drive period (the sawtooth-biased "QC trace"): the bias ramp repeats at a fixed rate while the stream records continuously, so averaging in blocks of one period beats uncorrelated noise down by `sqrt(n_periods)`. Accepts a `TimeStream` (uses `signal`, already trimmed by `discard_start_ms`) or a raw complex array; specify the period either as `period_s` or as `n_periods` (exactly one). Leftover samples after the last whole period are dropped. Returns `(time_ms, avg_iq)` with `avg_iq` shape `(2, n_samples)` holding averaged I and Q.
- **`plot_psd`** (`plotting.py`) — the drawing half of the PSD workflow: takes the `(f, psd_a, psd_b)` that `averaged_psd_timestream`, `averaged_psd_cleaned` or a bare `compute_psd` produce, draws them on log-log axes, and fits each panel with `fit_parity_psd`. The two channels mean different things in different bases, and `basis` is what names them: `"resonator"` (default) is dissipation (radial) / frequency (arc-length) in `1/Hz`, `"electronic"` is I / Q in `FS²/Hz`. `labels`/`units` override both for a projection that is neither (`BiasHunt`'s magnitude spectrum). Deliberately narrower than `plot_iq_comparison`'s three-valued `basis`: nothing projects a *spectrum* into the fractional basis. **Each panel is fit from the array passed in**, which is the whole point — a `fit_results` lifted off a measurement object describes only whichever single channel was last computed, so overlaying it on both panels compares a spectrum against a model of a different spectrum. Accepts 1-D or 2-D input (one PSD per row, one row per tone) and returns `(axes, fits)` with `fits` keyed `"a"`/`"b"`, following `fit_parity_psd`'s own convention of a dict for 1-D and a list for 2-D; a channel that is absent, unfitted or whose fit raised is `None`, and a failed fit is reported on its panel rather than costing the spectrum. Like `BiasHunt.analyze`'s panel it plots the raw periodogram faintly, the log-binned points the fit used as markers, and **frames the y-axis on those** — a bare periodogram scatters over ten decades and reads as a bad fit even when it is a good one. `f_bw` should be the *tuned* `df`; it defaults to `2 * f[-1]`, exact to within one frequency bin and affecting only the height of the white floor. Returning `fits` alongside the axes is the one place this departs from the module's return-the-axes convention.
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
