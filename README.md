# daq
Stable wrappers for data acquisition system based on Intermodulation 
Product Presto-8, used at WashU Astroparticle Lab.

## Features

- **Measurement Classes**: Sweep, TimeStream, SweepPower, 
  SweepFreqAndDC, TwoTonePower
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
- pandas (for database querying)
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

### TwoTonePower Measurement

```python
from daq import TwoTonePower

# Create a two-tone power measurement
# 2D sweep of pump power and frequency with fixed probe
tt = TwoTonePower(
    readout_freq=6e9,          # Fixed probe frequency
    control_freq_center=5e9,    # Pump center frequency
    control_freq_span=100e6,    # Pump frequency span
    df=1e3,                     # Frequency resolution
    readout_amp=0.1,           # Probe amplitude
    control_amp_arr=[0.01, 0.05, 0.1],  # Pump amplitudes
    readout_port=1,            # Probe output port
    control_port=2,            # Pump output port
    input_port=1,              # Input port
    num_averages=100,          # Number of averages
    device="Device_C",         # Device name (required for DB)
    notes="Two-tone spectroscopy"
)

# Run the measurement
filepath = tt.run()

# Analyze results
tt.analyze(quantity="quadrature", linecut=True)
```

## MongoDB Database Integration

All measurements are automatically logged to MongoDB Atlas:
- **Database**: "WashU_Astroparticle_Detector"
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

### Querying the Database

Use the `select_runs` function to query measurement runs from the database:

```python
from daq.db import select_runs
import pandas as pd
```

#### Basic Queries

**Query by device:**
```python
# Find all measurements for a specific device
df = select_runs(device="Resonator_A")
print(df[['number', 'file', 'utc_time', 'type']])
```

**Query by measurement type:**
```python
# Find all sweep measurements
df = select_runs(measurement_type="sweep")
```

**Query with multiple filters:**
```python
# Find sweeps for a specific device
df = select_runs(
    device="Resonator_A",
    measurement_type="sweep"
)
```

#### Time Range Queries

**Using ISO format strings:**
```python
# Find measurements in a date range
df = select_runs(
    start_time="2024-01-01T00:00:00",
    end_time="2024-12-31T23:59:59"
)
```

**Using datetime objects:**
```python
from datetime import datetime

# Find measurements from the last week
start = datetime(2024, 1, 1)
end = datetime(2024, 1, 8)
df = select_runs(start_time=start, end_time=end)
```

**Combine time range with other filters:**
```python
# Find recent sweeps for a specific device
df = select_runs(
    device="Resonator_A",
    measurement_type="sweep",
    start_time="2024-01-01T00:00:00",
    end_time="2024-12-31T23:59:59"
)
```

#### String Matching Modes

**Exact matching (default):**
```python
# Exact match for device name
df = select_runs(device="Resonator_A", string_match="exact")
```

**Partial/regex matching:**
```python
# Find all devices starting with "Resonator"
df = select_runs(
    device="Resonator",
    string_match="regex"
)

# Find notes containing "cooldown" (case-insensitive)
df = select_runs(
    notes="cooldown",
    string_match="regex"
)
```

#### Advanced Filtering

**Filter by additional measurement parameters:**
```python
# Find sweeps with specific amplitude
df = select_runs(
    measurement_type="sweep",
    amp=0.1
)

# Find measurements with specific port configuration
df = select_runs(
    output_port=1,
    input_port=1
)

# Combine multiple parameter filters
df = select_runs(
    device="Resonator_A",
    measurement_type="sweep",
    freq_center=5e9,
    amp=0.1
)
```

**Filter by fit results (for Sweep measurements):**
```python
# Find measurements with specific resonant frequency
# Note: Fit fields may not exist for all measurements
df = select_runs(measurement_type="sweep")
df_with_fit = df[df['fit_fr'].notna()]
df_filtered = df_with_fit[
    (df_with_fit['fit_fr'] > 4.9e9) & 
    (df_with_fit['fit_fr'] < 5.1e9)
]
```

#### Working with Results

The function returns a pandas DataFrame with all document fields:

```python
# Get all results
df = select_runs(device="Resonator_A")

# Display basic info
print(f"Found {len(df)} measurements")
print(df.columns.tolist())

# Access specific fields
for idx, row in df.iterrows():
    print(f"Run {row['number']}: {row['file']}")
    print(f"  Device: {row['device']}")
    print(f"  Type: {row['type']}")
    print(f"  Time: {row['utc_time']}")

# Export to CSV
df.to_csv('measurements.csv', index=False)

# Filter DataFrame further
recent_sweeps = df[
    (df['type'] == 'sweep') & 
    (df['utc_time'] > '2024-01-01')
]
```

**Empty results:**
```python
# Returns empty DataFrame if no matches found
df = select_runs(device="NonExistentDevice")
print(df.empty)  # True
print(len(df))   # 0
```

### Listing Devices

Use the `list_devices` function to get all unique device names recorded in 
the database:

```python
from daq.db import list_devices

# Get all devices with their measurement counts
devices_df = list_devices()
print(devices_df)
```

**Example output:**
```
         device  count
0  Resonator_A     42
1  Resonator_B     31
2   Detector_C     18
```

The results are sorted by count in descending order (most measurements first).

**Working with device list:**
```python
# Get device names as a list
devices_df = list_devices()
device_names = devices_df['device'].tolist()
print(f"Found {len(device_names)} devices: {device_names}")

# Get device with most measurements
top_device = devices_df.iloc[0]['device']
print(f"Most measured device: {top_device}")

# Filter devices with at least 10 measurements
active_devices = devices_df[devices_df['count'] >= 10]
print(active_devices)
```

**Empty database:**
```python
# Returns empty DataFrame with correct columns if no devices found
devices_df = list_devices()
print(devices_df.empty)  # True
print(devices_df.columns.tolist())  # ['device', 'count']
```

## Project Structure

```
daq/
├── measurements/        # Measurement classes
│   ├── sweep.py
│   ├── timestream.py
│   ├── sweep_power.py
│   ├── sweep_freq_and_dc.py
│   └── two_tone_power.py
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
