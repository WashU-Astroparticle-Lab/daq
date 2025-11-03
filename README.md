# daq
Stable wrappers for data acquisition system based on Intermodulation 
Product Presto-8, used at WashU Astroparticle Lab.

## Features

- **Measurement Classes**: Sweep, TimeStream, SweepPower, 
  SweepFreqAndDC
- **MongoDB Integration**: Automatic logging of all measurements 
  to MongoDB Atlas (setup by Lanqing, which is beyond the scope of this package)
- **Automatic Fitting**: Sweep measurements automatically perform 
  resonator fitting (optional, enabled by default)
- **Data Management**: Organized data storage with cumulative 
  numbering system
- **Analysis Tools**: Built-in visualization and fitting 
  capabilities

## Installation

### Using pip:
```bash
pip install -e .
```

### Dependencies
- numpy
- h5py
- matplotlib
- pymongo (for database integration)
- presto
- resonator_tools

## Usage

### Basic Sweep Measurement

```python
from daq import Sweep

# Create a sweep measurement with database logging
sweep = Sweep(
    freq_center=5e9,        # 5 GHz center frequency
    freq_span=100e6,        # 100 MHz span
    df=1e3,                 # 1 kHz resolution
    num_averages=100,       # Number of averages
    amp=0.1,                # Amplitude (fraction of full scale)
    output_port=1,          # DAC output port
    input_port=1,           # ADC input port
    device="Resonator_A",   # Device name (required for DB)
    filter="LPF_10GHz",     # Optional filter name
    notes="Cooldown test",   # Optional notes
    auto_fit=True           # Automatic fitting (default: True)
)

# Run the measurement
# - Automatically performs resonator fitting (if enabled)
# - Saves fit results to database
# - Saves data file to disk
filepath = sweep.run()
print(f"Data saved to: {filepath}")

# Visualize the results (optional)
sweep.analyze()
```

**Note**: By default, Sweep measurements automatically perform resonator 
fitting after data acquisition and store fit results (frequency, quality 
factors, etc.) in the MongoDB database. Set `auto_fit=False` to disable 
automatic fitting.

### TimeStream Measurement

```python
from daq import TimeStream

# Create a timestream measurement
ts = TimeStream(
    lo_freq=5e9,            # LO frequency
    if_freqs=[10e6, 20e6],  # IF frequencies
    df=1e3,                 # Sample rate
    pixel_counts=10000,     # Number of samples
    amp=[0.05, 0.05],       # Amplitudes for each tone
    output_port=1,
    input_port=1,
    device="Detector_B",    # Device name (required for DB)
    notes="Noise measurement"
)

# Run and analyze
filepath = ts.run()
ts.analyze()
```

## MongoDB Database Integration

All measurements are automatically logged to MongoDB Atlas:
- **Database**: "WashU Astroparticle Detector"
- **Collection**: "measurement"

### Document Structure

Each measurement creates a document with:
- `utc_time`: UTC timestamp
- `number`: 8-digit cumulative measurement number (e.g., "00000001")
- `type`: "sweep" or "timestream"
- `device`: Device name (required)
- `filter`: Filter name (optional)
- `notes`: User notes (optional)
- `file`: Full path to HDF5 data file
- `output_port`, `input_port`: Port numbers
- `amp`: Readout amplitude
- All measurement-specific parameters (freq_center, lo_freq, etc.)

**For Sweep measurements with automatic fitting enabled**, the document 
also includes fit results:
- `fit_fr`, `fit_fr_err`: Resonant frequency and error (Hz)
- `fit_Qi`, `fit_Qi_err`: Internal quality factor and error
- `fit_Qc`, `fit_Qc_err`: Coupling quality factor and error
- `fit_Ql`, `fit_Ql_err`: Loaded quality factor and error
- `fit_kappa`: Coupling rate = fr / Qc (Hz)

These fit fields are only present if automatic fitting succeeds. If 
fitting is disabled (`auto_fit=False`) or fails, the document is saved 
without fit fields.

### File Naming Convention

Data files are automatically named: `{number}-{device}-{type}.h5`

Example: `00000042-Resonator_A-sweep.h5`

## Project Structure

```
daq/
├── measurements/        # Measurement classes
│   ├── sweep.py
│   ├── timestream.py
│   ├── sweep_power.py
│   └── sweep_freq_and_dc.py
├── db/                 # Database integration
│   └── database.py
├── analysis/           # Analysis tools
│   └── mattis_bardeen.py
├── _base.py            # Base class for measurements
└── utils.py            # Utility functions

data/                   # Data storage directory
```

## Configuration

Edit `daq/utils.py` to configure:
- `PRESTO_ADDRESS`: Presto device IP address
- `PRESTO_PORT`: Presto device port
- `DATA_FOLDER`: Data storage location

Database credentials are configured in `daq/db/database.py`.
