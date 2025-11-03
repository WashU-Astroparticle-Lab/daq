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
- USB/LSB sideband separation

**Key Parameters**:
- `lo_freq`: LO frequency (Hz)
- `if_freqs`: Array of IF frequencies (Hz)
- `df`: Sample rate (Hz)
- `pixel_counts`: Number of samples
- `amp`: Array of amplitudes for each tone
- `output_port`: DAC output port
- `input_port`: ADC input port
- `device`: Device name (required for DB)
- `filter`: Filter name (optional)
- `notes`: Measurement notes (optional)

**Usage Example**:
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

**Analysis**: Plots I/Q streams for each frequency tone.

---

### 3. SweepPower (`sweep_power.py`)

**Purpose**: 2D sweep of drive power and frequency.

**Key Features**:
- 2D parameter sweep (frequency × power)
- Power-dependent resonator characterization
- Interactive visualization with linecuts

**Key Parameters**:
- `freq_center`: Center frequency (Hz)
- `freq_span`: Frequency span (Hz)
- `df`: Frequency resolution (Hz)
- `num_averages`: Number of averages per point
- `amp_arr`: Array of drive amplitudes
- `output_port`: DAC output port
- `input_port`: ADC input port
- `device`: Device name (required for DB)
- `filter`: Filter name (optional)
- `notes`: Measurement notes (optional)

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

**Analysis**: 2D heatmap with interactive linecuts and optional resonator 
fitting.

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
- `control_amp_arr`: Array of pump amplitudes
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

All measurements automatically log to MongoDB Atlas:
- **Database**: "WashU_Astroparticle_Detector"
- **Collection**: "measurement"
- **Document Fields**: All measurement parameters + metadata
- **Fit Results**: Sweep measurements include resonator fit parameters

If database is unavailable, measurements are still saved locally with 
timestamp-based numbering.

---

## Analysis Methods

Each measurement class provides an `analyze()` method for visualization:

- **Sweep**: Static plots with optional resonator fitting
- **TimeStream**: I/Q stream plots for each frequency
- **SweepPower**: 2D heatmap with interactive linecuts
- **SweepFreqAndDC**: 2D heatmap with multiple quantity options
- **TwoTonePower**: 2D heatmap with interactive linecuts

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

