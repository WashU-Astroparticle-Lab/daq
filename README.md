# daq
Stable wrappers for data acquisition system based on Intermodulation 
Product Presto-8, used at WashU Astroparticle Lab.

## Features

- **Measurement Classes**: Sweep, TimeStream, SweepPower, 
  SweepFreqAndDC
- **MongoDB Integration**: Automatic logging of all measurements 
  to MongoDB Atlas
- **Data Management**: Organized data storage with cumulative 
  numbering system
- **Analysis Tools**: Built-in visualization and fitting 
  capabilities

## Installation

### Using pip:
```bash
pip install -e .
```

### Using poetry:
```bash
poetry install
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
    notes="Cooldown test"   # Optional notes
)

# Run the measurement (saves to DB automatically)
filepath = sweep.run()
print(f"Data saved to: {filepath}")

# Analyze the results
sweep.analyze()
```

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
├── _base.py            # Base class for measurements
├── utils.py            # Utility functions
└── analysis.py         # Analysis tools

data/                   # Data storage directory
```

## Configuration

Edit `daq/utils.py` to configure:
- `PRESTO_ADDRESS`: Presto device IP address
- `PRESTO_PORT`: Presto device port
- `DATA_FOLDER`: Data storage location

Database credentials are configured in `daq/db/database.py`.
