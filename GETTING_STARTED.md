# Getting Started with MongoDB Integration

## Quick Start Guide

### 1. Install Dependencies

First, install the required `pymongo` package:

```bash
# Using pip
pip install pymongo

# Or install all project dependencies
pip install -e .

# Using poetry
poetry add pymongo
# or
poetry install
```

### 2. Your First Measurement with Database Logging

Here's a complete example:

```python
from daq import Sweep

# Create a sweep with database logging
# Only 'device' is required for DB integration
sweep = Sweep(
    freq_center=5.5e9,           # 5.5 GHz
    freq_span=100e6,             # 100 MHz
    df=1e3,                      # 1 kHz resolution
    num_averages=100,
    amp=0.1,                     # 10% of full scale
    output_port=1,
    input_port=1,
    device="Resonator_A",        # REQUIRED for DB
    filter="LPF_10GHz",          # Optional
    notes="First test measurement"  # Optional but recommended
)

# Run the measurement
filepath = sweep.run()

# Expected output:
# Data saved to: /path/to/data/00000001-Resonator_A-sweep.h5
# Document inserted to MongoDB with ID: 507f1f77bcf86cd799439011

# Analyze the results
sweep.analyze()
```

### 3. TimeStream Example

```python
from daq import TimeStream

ts = TimeStream(
    lo_freq=6e9,
    if_freqs=[10e6, 20e6, 30e6],  # Three tones
    df=1e3,
    pixel_counts=10000,
    amp=[0.05, 0.05, 0.05],       # Amplitude for each tone
    output_port=1,
    input_port=1,
    device="MKID_Array",          # REQUIRED
    notes="Noise measurement at 50mK"
)

filepath = ts.run()
ts.analyze()
```

## What Happens Automatically

When you call `.run()`, the system:

1. ✅ Gets next available measurement number from database
2. ✅ Generates filename: `{number}-{device}-{type}.h5`
3. ✅ Saves HDF5 file to `data/` folder
4. ✅ Creates MongoDB document with all metadata
5. ✅ Inserts document to database
6. ✅ Returns filepath

## MongoDB Document Example

Your measurement creates a document like this:

```json
{
    "_id": "507f1f77bcf86cd799439011",
    "utc_time": "2025-11-03T15:30:45.123456",
    "number": "00000001",
    "type": "sweep",
    "device": "Resonator_A",
    "filter": "LPF_10GHz",
    "notes": "First test measurement",
    "file": "/Users/you/daq/data/00000001-Resonator_A-sweep.h5",
    "output_port": 1,
    "input_port": 1,
    "amp": 0.1,
    "freq_center": 5500000000.0,
    "freq_span": 100000000.0,
    "df": 1000.0,
    "num_averages": 100,
    "dither": true,
    "num_skip": 0
}
```

## Important Parameters

### Required
- `device`: Name of the device being measured
  - **Must be provided for database logging**
  - Used in filename generation
  - Example: "Resonator_A", "MKID_Array", "Detector_B"

### Highly Recommended
- `notes`: Explanation of what you're measuring and why
  - Helps track experiments over time
  - Example: "Testing after cooldown", "Sweeping to find resonance"

### Optional
- `filter`: Name of filter in signal path
  - Default: `None`
  - Example: "LPF_10GHz", "BPF_4-8GHz"

## File Organization

```
your_project/
├── data/                    # Auto-created
│   ├── 00000001-Resonator_A-sweep.h5
│   ├── 00000002-MKID_Array-timestream.h5
│   ├── 00000003-Detector_B-sweep.h5
│   └── ...
└── your_script.py
```

## Querying the Database (Optional)

You can also query the database directly:

```python
from daq.db.database import _get_collection

collection = _get_collection()

# Get the latest measurement
latest = collection.find_one(sort=[("number", -1)])
print(f"Latest: {latest['number']} - {latest['device']}")

# Find all measurements for a specific device
for doc in collection.find({"device": "Resonator_A"}):
    print(f"{doc['number']}: {doc['notes']}")

# Count total measurements
count = collection.count_documents({})
print(f"Total measurements: {count}")

# Find measurements by type
sweeps = collection.find({"type": "sweep"}).count()
streams = collection.find({"type": "timestream"}).count()
print(f"Sweeps: {sweeps}, TimeStreams: {streams}")
```

## Error Handling

The system is designed to never lose your data:

```python
# Even if database is unavailable...
sweep = Sweep(..., device="TestDevice")
filepath = sweep.run()

# Output might be:
# Data saved to: /path/to/data/00000X-TestDevice-sweep.h5
# WARN: Failed to insert to MongoDB: ConnectionError(...)

# Your data is STILL SAVED in the HDF5 file!
# You can re-insert to database later if needed
```

## Configuration

Edit `daq/utils.py` to change:

```python
PRESTO_ADDRESS = "172.23.20.29"  # Your Presto IP
PRESTO_PORT = None               # Or specific port
DATA_FOLDER = "/path/to/data"    # Data storage location
```

Database settings are in `daq/db/database.py`.

## Migrating Existing Code

If you have existing scripts, just add the `device` parameter:

**Before:**
```python
sweep = Sweep(
    freq_center=5e9,
    freq_span=100e6,
    df=1e3,
    num_averages=100,
    amp=0.1,
    output_port=1,
    input_port=1
)
```

**After:**
```python
sweep = Sweep(
    freq_center=5e9,
    freq_span=100e6,
    df=1e3,
    num_averages=100,
    amp=0.1,
    output_port=1,
    input_port=1,
    device="Resonator_A",        # Add this
    notes="Your description"      # And optionally this
)
```

That's it! Everything else works the same way.

## Troubleshooting

### "ModuleNotFoundError: No module named 'pymongo'"
```bash
pip install pymongo
```

### "ValueError: device parameter is required for database logging"
Add the `device` parameter to your measurement:
```python
sweep = Sweep(..., device="YourDeviceName")
```

### "Failed to insert to MongoDB: ConnectionError"
- Check internet connection
- Verify MongoDB Atlas credentials in `daq/db/database.py`
- Your data is still saved locally in HDF5 file

### Files not appearing in `data/` folder
- The folder is auto-created on first run
- Default location is `./data/` relative to project root
- Change location in `daq/utils.py` if needed

## Next Steps

1. ✅ Install pymongo: `pip install pymongo`
2. ✅ Try the Quick Start example above
3. ✅ Check your MongoDB database for the new document
4. ✅ Update your existing scripts to include `device` parameter
5. ✅ Read `MONGODB_INTEGRATION.md` for detailed information

## Need Help?

- See `README.md` for general DAQ usage
- See `MONGODB_INTEGRATION.md` for technical details
- See `IMPLEMENTATION_CHECKLIST.md` for what was implemented
- Check MongoDB Atlas dashboard to view your measurements

Happy measuring! 🎉

