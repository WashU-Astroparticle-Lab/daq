# MongoDB Integration - Implementation Summary

## Overview

This document describes the MongoDB Atlas integration that has been 
implemented for the DAQ system. All measurements taken with `Sweep` 
and `TimeStream` classes are now automatically logged to a MongoDB 
Atlas database.

## Changes Made

### 1. Code Restructuring

The codebase has been reorganized into a more maintainable structure:

```
daq/
├── measurements/           # All measurement classes
│   ├── __init__.py
│   ├── sweep.py           # Modified for DB integration
│   ├── timestream.py      # Modified for DB integration
│   ├── sweep_power.py     # Imports updated
│   └── sweep_freq_and_dc.py  # Imports updated
├── db/                    # Database integration (NEW)
│   ├── __init__.py
│   └── database.py        # MongoDB operations
├── _base.py               # Modified for DB integration
├── utils.py               # Added DATA_FOLDER configuration
├── analysis.py            # Unchanged
└── __init__.py            # Updated imports
```

### 2. Database Module (`daq/db/database.py`)

New module providing MongoDB Atlas integration:

**Functions:**
- `get_next_number()`: Retrieves the next cumulative measurement 
  number
- `generate_filename(number, device, type)`: Creates standardized 
  filenames
- `insert_measurement(document)`: Inserts measurement metadata 
  to database

**Configuration:**
- Database: "WashU Astroparticle Detector"
- Collection: "measurement"
- URI: Hardcoded in `database.py`

### 3. Base Class Modifications (`daq/_base.py`)

The `Base._save()` method now:
1. Gets next measurement number from database
2. Generates filename: `{number}-{device}-{type}.h5`
3. Saves HDF5 file to `data/` folder
4. Builds MongoDB document with all metadata
5. Inserts document to database
6. Returns filepath

New method `_build_document()` constructs MongoDB documents:
- Converts numpy arrays to lists
- Handles all data types properly
- Excludes large data arrays from database
- Includes all `__init__` parameters

### 4. Measurement Class Updates

**Sweep** (`daq/measurements/sweep.py`):
- Added parameters: `device`, `filter`, `notes`
- `device` is required for database logging
- All parameters stored as instance variables

**TimeStream** (`daq/measurements/timestream.py`):
- Added parameters: `device`, `filter`, `notes`
- `device` is required for database logging
- All parameters stored as instance variables

### 5. Configuration (`daq/utils.py`)

Added:
- `DATA_FOLDER`: Path to data storage directory
- `get_data_folder()`: Function to retrieve data folder path

### 6. Dependencies

Updated `pyproject.toml` and created `requirements.txt`:
- Added `pymongo` as a dependency
- Required for MongoDB Atlas connectivity

## MongoDB Document Structure

Each measurement creates a document with the following fields:

### Required Fields
- `utc_time`: ISO format UTC timestamp (auto-generated)
- `number`: 8-digit zero-padded string (e.g., "00000001")
- `type`: Measurement type ("sweep" or "timestream")
- `device`: Device name (user-provided)
- `file`: Full path to HDF5 data file
- `output_port`: Presto output port number
- `input_port`: Presto input port number
- `amp`: Readout amplitude(s)

### Optional Fields
- `filter`: Filter name (default: None)
- `notes`: User notes explaining the measurement (default: None)

### Measurement-Specific Fields

**Sweep measurements include:**
- `freq_center`: Center frequency (Hz)
- `freq_span`: Frequency span (Hz)
- `df`: Frequency resolution (Hz)
- `num_averages`: Number of averages
- `dither`: Dither setting (boolean)
- `num_skip`: Number of samples to skip

**TimeStream measurements include:**
- `lo_freq`: LO frequency (Hz)
- `if_freqs`: List of IF frequencies (Hz)
- `df`: Sample rate (Hz)
- `pixel_counts`: Number of samples
- `dither`: Dither setting (boolean)
- `phases_i`: I phase values (list)
- `phases_q`: Q phase values (list)

## Usage Examples

### Sweep Measurement with Database Logging

```python
from daq import Sweep

sweep = Sweep(
    freq_center=5.5e9,
    freq_span=100e6,
    df=1e3,
    num_averages=100,
    amp=0.1,
    output_port=1,
    input_port=1,
    dither=True,
    device="Resonator_A",        # Required
    filter="LPF_10GHz",          # Optional
    notes="Testing new resonator"  # Optional
)

# Run measurement - automatically saves to DB
filepath = sweep.run()
# Output: Data saved to: /path/to/data/00000001-Resonator_A-sweep.h5
# Output: Document inserted to MongoDB with ID: <mongodb_id>
```

### TimeStream Measurement with Database Logging

```python
from daq import TimeStream

ts = TimeStream(
    lo_freq=6e9,
    if_freqs=[10e6, 20e6, 30e6],
    df=1e3,
    pixel_counts=50000,
    amp=[0.05, 0.05, 0.05],
    output_port=1,
    input_port=1,
    device="MKID_Array",         # Required
    filter=None,                 # Optional
    notes="Noise characterization run #3"
)

filepath = ts.run()
# Output: Data saved to: /path/to/data/00000002-MKID_Array-timestream.h5
# Output: Document inserted to MongoDB with ID: <mongodb_id>
```

## File Naming Convention

All data files follow the pattern:
```
{number}-{device}-{type}.h5
```

Examples:
- `00000001-Resonator_A-sweep.h5`
- `00000042-Detector_B-timestream.h5`
- `00001337-MKID_Array-sweep.h5`

The number is cumulative across all measurements in the database, 
ensuring unique filenames.

## Error Handling

The system includes robust error handling:

1. **Missing Device Name**: Raises `ValueError` if `device` is not 
   provided
2. **Database Connection Issues**: Prints warning but continues with 
   file save
3. **Database Insert Failures**: Prints warning, file is still saved 
   locally

This ensures that measurements are never lost due to database issues.

## Installation Requirements

Before using the database integration, install dependencies:

```bash
# Using pip
pip install pymongo

# Or install all dependencies
pip install -e .

# Using poetry
poetry install
```

## Testing the Integration

To verify the database integration is working:

```python
# Test imports
from daq import Sweep, TimeStream
from daq.db import get_next_number, generate_filename

# Test database connection
next_num = get_next_number()
print(f"Next measurement number: {next_num}")

# Test filename generation
filename = generate_filename("00000001", "TestDevice", "sweep")
print(f"Generated filename: {filename}")
```

## Notes for Users

1. **Device Name is Required**: Always provide a `device` parameter 
   when creating measurements
2. **Notes are Highly Recommended**: Add descriptive notes to help 
   track experiments
3. **Filter Information**: Record filter settings for traceability
4. **Cumulative Numbering**: Numbers increment automatically across 
   all measurement types
5. **Data Persistence**: Files are saved locally even if database 
   fails

## Future Enhancements

Potential improvements for future versions:

1. Query interface to search and retrieve measurements from database
2. Web dashboard for viewing measurement history
3. Automatic data backup and synchronization
4. Measurement comparison tools
5. Integration with analysis pipelines
6. User authentication and access control

## Technical Details

### Database Connection

The MongoDB client uses:
- Server API version 1
- Automatic retry on transient failures
- Connection pooling for efficiency

### Document Size Limits

MongoDB documents are limited to 16 MB. The implementation excludes:
- Raw data arrays (`freq_arr`, `resp_arr`, etc.)
- Pixel data arrays (`pixel_i`, `pixel_q`, etc.)
- Sideband data (`lsb`, `usb`, etc.)

These are stored only in HDF5 files. The database contains metadata 
and configuration parameters only.

### Data Type Conversion

The system automatically converts:
- NumPy arrays → Python lists
- NumPy scalars → Python native types
- Other types → String representation

This ensures compatibility with MongoDB's BSON format.

## Support

For issues or questions:
1. Check that `pymongo` is installed
2. Verify MongoDB Atlas credentials in `daq/db/database.py`
3. Ensure network connectivity to MongoDB Atlas
4. Review error messages in console output

## Summary

The MongoDB integration provides:
✓ Automatic metadata logging for all measurements
✓ Cumulative numbering system for data organization
✓ Standardized file naming convention
✓ Robust error handling
✓ Easy-to-use interface (no changes to workflow)
✓ Complete traceability of experimental conditions

All existing scripts will continue to work - just add the `device` 
parameter to enable database logging!

