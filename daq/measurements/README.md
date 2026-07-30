# Supported Measurements

This directory contains all measurement classes for the DAQ system. Each 
measurement class inherits from `Base` and provides automatic database 
logging, file management, and data analysis capabilities.

## Overview

All measurements support:
- **MongoDB Integration**: Automatic logging to MongoDB Atlas
- **File Management**: Automatic filename generation and organization
- **Data Persistence**: HDF5 format with metadata storage
- **Analysis Tools**: Built-in visualization methods

## Measurement Classes

### 1. Sweep (`sweep.py`)

**Purpose**: Single-tone frequency sweep measurement for resonator 
characterization.

**Key Features**:
- 1D frequency sweep (center frequency ± span)
- Automatic resonator fitting (optional, enabled by default)
- Fit results stored in database (fr, Qi, Qc, Ql, kappa)

**Key Parameters**:
- `freq_center`: Center frequency (Hz)
- `freq_span`: Frequency span (Hz)
- `df`: Frequency resolution (Hz)
- `num_averages`: Number of averages per point
- `amp`: Drive amplitude (fraction of full scale)
- `output_port`: DAC output port
- `input_port`: ADC input port
- `auto_fit`: Enable automatic fitting (default: True)
- `device`: Device name (required for DB)
- `filter`: Filter name (optional)
- `notes`: Measurement notes (optional)

**Usage Example**:
```python
from daq import Sweep

sweep = Sweep(
    freq_center=5e9,
    freq_span=100e6,
    df=1e3,
    num_averages=100,
    amp=0.1,
    output_port=1,
    input_port=1,
    device="Resonator_A",
    auto_fit=True
)
filepath = sweep.run()
sweep.analyze()
```

**Analysis**: Provides visualization with optional resonator fitting 
results.

---

### 2. TimeStream (`timestream.py`)

**Purpose**: Time-domain measurement with multiple simultaneous tones.

**Key Features**:
- Multi-tone measurement (multiple IF frequencies)
- Time-domain data acquisition
- I/Q data streams
- Per-tone USB/LSB sideband selection via the `is_usb` flag

**Key Parameters**:
- `lo_freq`: LO frequency (Hz)
- `if_freqs`: Array of IF frequencies (Hz), each with `|IF| < 500 MHz`
- `df`: Sample rate (Hz)
- `pixel_counts`: Number of samples
- `amp`: Array of amplitudes for each tone
- `output_port`: DAC output port
- `input_port`: ADC input port
- `is_usb`: Optional per-tone bool array — `True` (default) puts the tone on the
  upper sideband (`LO + IF`), `False` on the lower sideband (`LO - IF`). A single
  bool applies to every tone. The single-sideband output phases are derived
  automatically, so you never set them by hand.
- `discard_start_ms`: Milliseconds of startup junk dropped from the in-memory
  time-axis arrays after `run()`/`load()` (default `25.0`; set `0` to keep
  everything). The saved HDF5 keeps the full, untrimmed acquisition.
- `external_trigger`: Which Presto digital output ports assert a trigger during the
  acquisition, used to gate external instruments. `False` (default) triggers nothing;
  `True` is shorthand for `[1]` (port 1 only). Prefer `trigger_for(bias, led)`, which
  reads the ports off the instruments themselves — see **Gating external instruments**
  below.
- `device`: Device name (required for DB)
- `filter`: Filter name (optional)
- `notes`: Measurement notes (optional)

**Sideband selection** — the hardware limits each IF magnitude to `< 500 MHz`, so
USB-only readout can only reach `LO … LO + 500 MHz`. To cover tones on both sides
of the LO (up to 1 GHz apart), center the LO between them and mark each tone USB or
LSB. For example, two tones 600 MHz apart at `f_lo = LO - 300 MHz` and
`f_hi = LO + 300 MHz`:

```python
ts = TimeStream(
    lo_freq=6.0e9,
    if_freqs=[300e6, 300e6],   # magnitudes only; sign is set by is_usb
    is_usb=[False, True],      # tone 0 -> LO - 300 MHz, tone 1 -> LO + 300 MHz
    df=1e3,
    pixel_counts=10000,
    amp=[0.05, 0.05],
    output_port=1,
    input_port=1,
    device="Detector_B",
    notes="Two tones 600 MHz apart",
)
filepath = ts.run()
ts.analyze()
```

**Gating external instruments** — `external_trigger` maps directly onto presto's
`set_trigger_out`, which takes one state **per digital output port**: element *i*
configures port *i+1*, with `0` = off, `1` = trigger on every lock-in window, `2` = on
every sum window. Which instrument you gate is therefore decided by what is wired where:

| `external_trigger` | ports triggered |
|---|---|
| `False` | none |
| `True`, `[1]`, `[1, 0]` | port 1 |
| `[0, 1]` | port 2 |
| `[1, 1]` | ports 1 and 2, fired together |

The Presto has four digital output ports, so at most four states; a longer list raises, as do
non-integral states (a state computed as `0.999…` would otherwise silently disable the port).

**Don't write the port numbers by hand.** Each driver carries the port it is wired to, and
`trigger_for` turns instruments into the states list — so a rig wired differently from the lab
default gates the right hardware without editing the measurement:

```python
from daq import DC2200, Agilent33220A, TimeStream, trigger_for

with Agilent33220A() as bias, DC2200() as led:
    ts = TimeStream(..., external_trigger=trigger_for(bias, led))   # -> [1, 1]
```

Override the wiring per instrument (`Agilent33220A(trigger_port=3)`) or per rig, with
`DAQ_FGEN_TRIGGER_PORT` / `DAQ_LED_TRIGGER_PORT`. `attach()` records each instrument's
`trigger_port`, so the saved measurement says which port was expected to gate what.

> **`ts.external_trigger` is an array, not a bool.** The constructor argument still accepts
> `True`/`False` and means exactly what it always did, but the attribute now stores the
> resolved per-port states. Test it with `.any()` or `.size` — a bare
> `if ts.external_trigger:` raises `ValueError` on an empty or multi-element array. This also
> applies to objects returned by `TimeStream.load()` on pre-existing files.

Two constraints worth knowing before you plan a measurement around this:

- **The trigger is high for the whole acquisition, not a pulse.** presto re-asserts it at
  the start of every lock-in window (every `1 / df`), and `TimeStream.TRIGGER_WIDTH_S`
  (30 ms) is far longer than any window used here, so the line goes high when acquisition
  starts and stays high until it ends. That is what a gated bias ramp or a TTL-driven LED
  wants; it is *not* a timed one-shot. This is **inferred from the presto implementation and
  not yet scope-verified** — see the call-out under *Routing the trigger* in
  `daq/instruments/README.md`.
- **Timing is shared across ports.** presto sends a single global `delay`/`width` pair
  alongside `df`, so ports enabled in the same acquisition are gated identically. Only the
  on/off state is per port. Independent timing needs separate acquisitions.

See `daq/instruments/README.md` for the wiring, the arming order, and the worked
Agilent-33220A and DC2200 recipes.

**Outputs**: after `run()`, use the ready-made per-tone fields rather than the raw
sidebands:
- `ts.signal` — complex I/Q array of shape `(n_samples, n_tones)`; column `i` is the
  tone's selected sideband.
- `ts.signal_freqs` — physical frequency (Hz) of each tone (`LO ± IF`).
- `ts.is_usb` — which sideband each tone used.

(The raw `ts.usb` / `ts.lsb` and `ts.freqs_usb` / `ts.freqs_lsb` are still available.)

**Usage Example** (single-tone, all-USB default):
```python
from daq import TimeStream

ts = TimeStream(
    lo_freq=6e9,
    if_freqs=[10e6, 20e6, 30e6],
    df=1e3,
    pixel_counts=10000,
    amp=[0.05, 0.05, 0.05],
    output_port=1,
    input_port=1,
    device="Detector_B",
    notes="Noise measurement"
)
filepath = ts.run()
ts.analyze()
```

**Analysis**: Plots I/Q streams for each tone, labelled with its sideband (USB/LSB).

---

### 3. SweepPower (`sweep_power.py`)

**Purpose**: 2D sweep of drive power and frequency.

**Key Features**:
- 2D parameter sweep (frequency × power)
- Power-dependent resonator characterization
- Per-power resonator fits with `fr` and `Qi` vs. drive power

**Key Parameters**:
- `freq_center`: Center frequency (Hz)
- `freq_span`: Frequency span (Hz)
- `df`: Frequency resolution (Hz)
- `num_averages`: Number of averages per point
- `amp_arr`: Array of drive amplitudes (fraction of full scale; use `power_dbm_to_amp` to convert from dBm)
- `output_port`: DAC output port
- `input_port`: ADC input port
- `device`: Device name (required for DB)
- `filter`: Filter name (optional)
- `notes`: Measurement notes (optional)
- `attenuation_db`: Line attenuation in dB (optional). When set, plots reference the drive power to the device input (drive power − attenuation)

**Usage Example**:
```python
from daq import SweepPower

sp = SweepPower(
    freq_center=5.5e9,
    freq_span=100e6,
    df=1e3,
    num_averages=100,
    amp_arr=[0.01, 0.05, 0.1, 0.2],
    output_port=1,
    input_port=1,
    device="Resonator_C",
    notes="Power sweep"
)
filepath = sp.run()
sp.analyze(norm=True, portrait=True)
```

**Analysis**: 2D response heatmap (with the fitted `fr` overlaid as a scatter),
plus best-fit `fr` and `Qi` (diag. corrected) vs. drive power with error bars.
Drive powers whose fit fails are omitted automatically.

---

### 4. SweepFreqAndDC (`sweep_freq_and_dc.py`)

**Purpose**: 2D sweep of DC bias and frequency for JPA modulation curve 
characterization.

**Key Features**:
- 2D parameter sweep (frequency × DC bias)
- Automatic DC bias ramping
- JPA characterization

**Key Parameters**:
- `freq_center`: Center frequency (Hz)
- `freq_span`: Frequency span (Hz)
- `df`: Frequency resolution (Hz)
- `num_averages`: Number of averages per point
- `amp`: Drive amplitude
- `bias_arr`: Array of DC bias values (V)
- `output_port`: DAC output port
- `input_port`: ADC input port
- `bias_port`: DC bias port
- `bias_ramp_rate`: Ramp rate for bias (V/s)
- `device`: Device name (required for DB)
- `filter`: Filter name (optional)
- `notes`: Measurement notes (optional)

**Usage Example**:
```python
from daq import SweepFreqAndDC

sf = SweepFreqAndDC(
    freq_center=6e9,
    freq_span=200e6,
    df=1e3,
    num_averages=100,
    amp=0.1,
    bias_arr=[0.0, 0.5, 1.0, 1.5, 2.0],
    output_port=1,
    input_port=1,
    bias_port=1,
    device="JPA_A",
    notes="Modulation curve"
)
filepath = sf.run()
sf.analyze(quantity="amplitude")
```

**Analysis**: 2D heatmap with various quantity options (amplitude, phase, 
dB, group delay, dpdb).

---

### 5. TwoTonePower (`two_tone_power.py`)

**Purpose**: Two-tone spectroscopy with 2D sweep of pump power and 
frequency, fixed probe.

**Key Features**:
- Two-tone spectroscopy measurement
- Fixed probe frequency (readout)
- Variable pump frequency and power
- Interactive visualization with linecuts

**Key Parameters**:
- `readout_freq`: Fixed probe frequency (Hz)
- `control_freq_center`: Pump center frequency (Hz)
- `control_freq_span`: Pump frequency span (Hz)
- `df`: Frequency resolution (Hz)
- `readout_amp`: Probe amplitude
- `control_amp_arr`: Array of pump amplitudes (fraction of full scale; use `power_dbm_to_amp` to convert from dBm)
- `readout_port`: Probe output port
- `control_port`: Pump output port
- `input_port`: ADC input port
- `num_averages`: Number of averages per point
- `device`: Device name (required for DB)
- `filter`: Filter name (optional)
- `notes`: Measurement notes (optional)

**Usage Example**:
```python
from daq import TwoTonePower

tt = TwoTonePower(
    readout_freq=6e9,
    control_freq_center=5e9,
    control_freq_span=100e6,
    df=1e3,
    readout_amp=0.1,
    control_amp_arr=[0.01, 0.05, 0.1],
    readout_port=1,
    control_port=2,
    input_port=1,
    num_averages=100,
    device="Device_C",
    notes="Two-tone spectroscopy"
)
filepath = tt.run()
tt.analyze(quantity="quadrature", linecut=True)
```

**Analysis**: 2D heatmap with optional linecuts. Quantity options: 
"quadrature", "amplitude", "phase", "dB".

---

### 6. QCTrace (`qc_trace.py`)

**Purpose**: Quantum-capacitance (charge-parity / quasiparticle-tunnelling) trace on one
device. Unlike the classes above it drives no hardware of its own — it composes a `Sweep`,
several `TimeStream`s and an `Agilent33220A` gate-bias source, because the four steps and their
ordering *are* the measurement.

**Sequence** (run in order by `run()`):
1. A `Sweep` with auto-fit to locate `fr`; every later step reads out there. Raises
   `RuntimeError` if the fit yields no usable `fr` (the sweep's own file is still saved).
2. The **QC trace** — a gated sawtooth on the gate plus an externally-triggered `TimeStream`
   spanning `num_periods` whole ramp periods, folded into one period by `fold_timestream`.
3. A **bias hunt** — one constant-bias `TimeStream` per entry of `bias_voltages`, keeping the
   largest parity contrast `std(|signal|)`.
4. One `TimeStream` under a **free-running** ramp, the same length as the tries.

**Key Parameters**:
- `freq_center`: Centre frequency of the locating sweep (Hz)
- `amp`: Drive amplitude (fraction of full scale; use `power_dbm_to_amp` to convert from dBm),
  shared by the sweep and every time stream
- `output_port` / `input_port`: DAC output / ADC input port
- `ramp_vpp`, `ramp_freq_hz`, `ramp_offset_v`, `ramp_symmetry_pct`: gate sawtooth shape
- `sampling_frequency`, `num_periods`: time-stream sample rate (Hz) and periods averaged
- `n_bias_try` / `bias_voltages`: number of random constant-bias tries, or explicit voltages
  to scan (e.g. a `numpy.linspace`) instead of sampling
- `v_min` / `v_max`, `seed`: bounds and seed for the random bias draw
- `ts_duration_s`: length (s) of each constant-bias try and of the ramp stream
- `trigger_states`: which Presto digital output ports gate the QC-trace step. `None` (default)
  takes the port from the bias generator's own `trigger_port`; pass presto states (`[1]`,
  `[0, 1]`, or `True` for port 1) to override it for one measurement
- `device` (required for DB), `filter`, `notes`

**Trigger routing**: only step 2 is gated, and by default it gates whichever port the bias
generator says it is wired to — `Agilent33220A.trigger_port`, port 1 in the lab's default
setup, overridable per instrument or via `DAQ_FGEN_TRIGGER_PORT`. Rewiring the rig therefore
needs no change to the measurement, and the generator is consulted on *every* run, so fixing a
port and re-running the same object gates the new one. This matters because getting it wrong is
silent: an ungated ramp sits at its burst start level and step 2 records a static bias rather
than a swept one. The resolved states are validated before the first acquisition — a routing
that gates no port raises, and an explicit one that leaves the generator's own port unasserted
warns — printed at the start of the run, and saved with the record (`trigger_states` in HDF5
and MongoDB), so a stored measurement pins down which port was gated. `load()` restores that
routing for inspection but does not pin it: re-running a loaded measurement re-reads the
generator in hand. Files written before the parameter existed load as port 1, which is what the
old hardcoded `external_trigger=True` meant.

**Usage Example**:
```python
from daq import QCTrace, power_dbm_to_amp

qct = QCTrace(
    freq_center=2.8e9,
    amp=power_dbm_to_amp(2.8, -110 + 78),   # device power + attenuation
    output_port=1, input_port=1,
    ramp_vpp=2.0, ramp_freq_hz=500,
    sampling_frequency=5e4, num_periods=200,
    n_bias_try=20, ts_duration_s=5.0,
    device="B260416-NG-D1_dev3",
    filter="VBFZ-2575-S+, ZX60-83LN-S+ and ZX60-63GLN+ warm amp",
)
qct.run()                       # opens the 33220A itself; or run(bias=<open instrument>)
qct.analyze()                   # parity contrast vs bias, above the folded I/Q trace
```

Note `run()` takes the **bias generator** as its only positional argument, not
`presto_address`; the Presto connection parameters are keyword-only
(`run(presto_address=...)`).

**Routing the gate on a differently-wired rig** — nothing about the measurement changes, so
tell the instrument, not the measurement:

```python
from daq import Agilent33220A, QCTrace

with Agilent33220A(trigger_port=3) as bias:      # or export DAQ_FGEN_TRIGGER_PORT=3
    qct = QCTrace(freq_center=2.8e9, amp=amp, output_port=1, input_port=1, device="my_device")
    qct.run(bias=bias)      # "QC trace: gating the ramp on Presto digital output port 3"
```

The generator is re-read on every run, so a port corrected mid-session (`bias.trigger_port =
2`) takes effect on the next `qct.run(bias=bias)` — no need to rebuild the measurement. To
pin the routing to the measurement instead, pass `QCTrace(..., trigger_states=[0, 0, 1])`;
that overrides the generator, and warns if it leaves the generator's own port unasserted.

**Records**: every step saves its own HDF5 + MongoDB record via the normal path with
`attach(bias=...)` applied, so each raw acquisition stays individually loadable. `QCTrace` saves
one further `qc_trace` record with the derived products (`fr`, `time_ms`/`avg_iq`,
`parity_contrast`, `best_bias`) and the constituent file paths (`sweep_file`, `qc_file`,
`best_bias_file`, `ramp_file`), plus the sweep's fit (`fit_fr`/`fit_Qi`/…). The constituent
objects hang off read-only properties (`sweep`, `qc_stream`, `bias_streams`,
`best_bias_stream`, `ramp_stream`); `load()` restores the derived record but not those objects.

**Analysis**: parity contrast versus gate bias with the winner marked, above the folded I/Q
trace over one ramp period.

---

## Common Parameters

All measurements share these common parameters:

### Database Integration
- `device` (str, **required**): Device name for database logging
- `filter` (str, optional): Filter name in signal path
- `notes` (str, optional): User notes explaining the measurement

### Hardware Configuration
- `output_port` (int): DAC output port number
- `input_port` (int): ADC input port number
- `dither` (bool, default=True): Enable dithering

### Measurement Settings
- `num_averages` (int): Number of averages per data point
- `num_skip` (int, default=0): Number of samples to skip before averaging
- `df` (float): Frequency resolution or sample rate (Hz)

### Run Method Parameters
All `run()` methods accept:
- `presto_address` (str, optional): Presto device IP (defaults to config)
- `presto_port` (int, optional): Presto device port (defaults to config)
- `ext_ref_clk` (bool, default=False): Use external reference clock
- `save_filename` (str, optional): Custom filename (auto-generated if None)

---

## Data Storage

All measurements save data in HDF5 format with:
- **Automatic Filenaming**: `{number}-{device}-{type}.h5`
- **Metadata Storage**: All parameters stored as HDF5 attributes
- **Data Arrays**: Measurement data stored as HDF5 datasets
- **Source Code**: Original measurement script saved for reference

---

## Database Integration

All measurements log to MongoDB when the configured server is available.
Default configuration:
- **URI**: `mongodb://localhost:27017`
- **Database**: `WashU_Astroparticle_Detector`
- **Collection**: `measurement`
- **Document Fields**: All measurement parameters + metadata
- **Fit Results**: Sweep measurements include resonator fit parameters

If database is unavailable, measurements are still saved locally with 
timestamp-based numbering.

---

## Analysis Methods

Each measurement class provides an `analyze()` method for visualization:

- **Sweep**: Static plots with optional resonator fitting
- **TimeStream**: I/Q stream plots for each frequency
- **SweepPower**: 2D heatmap with fitted `fr`/`Qi` vs. drive power
- **SweepFreqAndDC**: 2D heatmap with multiple quantity options
- **TwoTonePower**: 2D heatmap with interactive linecuts
- **QCTrace**: parity contrast vs. gate bias, above the block-averaged I/Q QC trace

---

## Load/Save Operations

All measurements support loading from saved files:

```python
# Save (automatic during run())
filepath = measurement.run()

# Load
from daq import Sweep
loaded = Sweep.load(filepath)
loaded.analyze()
```

---

## Notes

- All measurements require the `device` parameter for database integration
- Database connection failures don't prevent local file saving
- Measurement type is automatically determined from class name
- Large data arrays are excluded from database (stored only in HDF5 files)
- All measurements inherit error handling and robustness from `Base` class
