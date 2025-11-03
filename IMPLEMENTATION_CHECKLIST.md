# MongoDB Integration - Implementation Checklist

## ✅ Completed Tasks

### 1. ✅ Code Restructuring
- [x] Created `daq/measurements/` folder for measurement classes
- [x] Created `daq/db/` folder for database code
- [x] Moved `sweep.py` to `daq/measurements/`
- [x] Moved `timestream.py` to `daq/measurements/`
- [x] Moved `sweep_power.py` to `daq/measurements/`
- [x] Moved `sweep_freq_and_dc.py` to `daq/measurements/`
- [x] Created `daq/measurements/__init__.py`
- [x] Updated all imports in moved files to use relative imports
- [x] Updated `daq/__init__.py` to re-export from new locations
- [x] Added `DATA_FOLDER` to `daq/utils.py`
- [x] Added `get_data_folder()` function to `daq/utils.py`

### 2. ✅ Database Module
- [x] Created `daq/db/__init__.py`
- [x] Created `daq/db/database.py`
- [x] Implemented `get_next_number()` function
  - Queries MongoDB for max number
  - Returns 8-digit zero-padded string
  - Handles empty collection (starts at 1)
- [x] Implemented `generate_filename()` function
  - Format: `{number}-{device}-{type}.h5`
- [x] Implemented `insert_measurement()` function
  - Inserts document to MongoDB
  - Auto-adds UTC timestamp
  - Returns document ID
- [x] Configured MongoDB Atlas connection
  - Database: "WashU Astroparticle Detector"
  - Collection: "measurement"
  - URI hardcoded in module

### 3. ✅ Base Class Modifications
- [x] Updated imports in `_base.py`
- [x] Modified `_save()` method to:
  - Get next number from database
  - Generate standardized filename
  - Create data folder if needed
  - Save HDF5 file first
  - Build MongoDB document
  - Insert document to database
  - Handle errors gracefully
- [x] Created `_build_document()` method to:
  - Extract all instance attributes
  - Convert numpy types to Python types
  - Skip private attributes
  - Skip large data arrays
  - Include all metadata fields

### 4. ✅ Sweep Class Updates
- [x] Updated imports to use relative paths
- [x] Added `device` parameter to `__init__`
- [x] Added `filter` parameter to `__init__`
- [x] Added `notes` parameter to `__init__`
- [x] Stored all three as instance variables
- [x] All parameters automatically saved to database via Base class

### 5. ✅ TimeStream Class Updates
- [x] Updated imports to use relative paths
- [x] Added `device` parameter to `__init__`
- [x] Added `filter` parameter to `__init__`
- [x] Added `notes` parameter to `__init__`
- [x] Stored all three as instance variables
- [x] All parameters automatically saved to database via Base class

### 6. ✅ Other Measurement Classes
- [x] Updated imports in `sweep_power.py`
- [x] Updated imports in `sweep_freq_and_dc.py`
- [x] These classes NOT modified for DB integration (per user request)

### 7. ✅ Dependencies
- [x] Added `pymongo` to `pyproject.toml`
- [x] Created `requirements.txt` with all dependencies

### 8. ✅ Documentation
- [x] Updated `README.md` with:
  - Feature list
  - Installation instructions
  - Usage examples for Sweep and TimeStream
  - MongoDB integration details
  - Document structure explanation
  - File naming convention
  - Project structure overview
  - Configuration instructions
- [x] Created `MONGODB_INTEGRATION.md` with:
  - Complete implementation overview
  - Detailed changes made
  - MongoDB document structure
  - Usage examples
  - Error handling details
  - Installation requirements
  - Testing instructions
  - Technical details
  - Support information
- [x] Created this checklist

### 9. ✅ Testing Files
- [x] Created `test_db_integration.py` for verification
- [x] Created `data/` directory for data storage

## 📋 MongoDB Document Fields

### Standard Fields (All Measurements)
- ✅ `utc_time`: ISO format UTC timestamp
- ✅ `number`: 8-digit zero-padded measurement number
- ✅ `type`: "sweep" or "timestream"
- ✅ `device`: Device name (required)
- ✅ `filter`: Filter name (optional, default None)
- ✅ `notes`: User notes (optional, default None)
- ✅ `file`: Full path to HDF5 file
- ✅ `output_port`: Presto output port
- ✅ `input_port`: Presto input port
- ✅ `amp`: Readout amplitude(s)

### Sweep-Specific Fields
- ✅ `freq_center`: Center frequency
- ✅ `freq_span`: Frequency span
- ✅ `df`: Frequency resolution
- ✅ `num_averages`: Number of averages
- ✅ `dither`: Dither setting
- ✅ `num_skip`: Samples to skip

### TimeStream-Specific Fields
- ✅ `lo_freq`: LO frequency
- ✅ `if_freqs`: IF frequencies (list)
- ✅ `df`: Sample rate
- ✅ `pixel_counts`: Number of samples
- ✅ `dither`: Dither setting
- ✅ `phases_i`: I phases (list)
- ✅ `phases_q`: Q phases (list)

## 🎯 Design Decisions

### ✅ Implemented As Specified
1. ✅ Only Sweep and TimeStream classes get DB integration
2. ✅ Cumulative numbering via database query (not local counter)
3. ✅ device, filter, notes as optional `__init__` parameters
4. ✅ Database credentials hardcoded in database.py
5. ✅ DATA_FOLDER configured in utils.py
6. ✅ Code reorganized into measurements/ and db/ subfolders

### ✅ Additional Features
1. ✅ Automatic data type conversion (numpy → Python)
2. ✅ Robust error handling (measurement saved even if DB fails)
3. ✅ Automatic data folder creation
4. ✅ Comprehensive documentation
5. ✅ Backward compatible (existing code works with new param)

## 🔍 Verification Steps

To verify the implementation works correctly:

### Step 1: Install Dependencies
```bash
pip install pymongo
# or
poetry add pymongo
```

### Step 2: Test Imports
```python
from daq import Sweep, TimeStream
from daq.db import get_next_number, generate_filename
print("✓ Imports successful")
```

### Step 3: Test Database Connection
```python
from daq.db.database import _get_collection
collection = _get_collection()
print(f"✓ Connected to: {collection.database.name}.{collection.name}")
```

### Step 4: Test Number Generation
```python
from daq.db import get_next_number
next_num = get_next_number()
print(f"✓ Next number: {next_num}")
```

### Step 5: Run Test Measurement (if Presto available)
```python
from daq import Sweep

sweep = Sweep(
    freq_center=5e9,
    freq_span=100e6,
    df=1e3,
    num_averages=10,
    amp=0.1,
    output_port=1,
    input_port=1,
    device="TestDevice",
    notes="Integration test"
)

# This will save file and insert to DB
filepath = sweep.run()
print(f"✓ Data saved: {filepath}")
```

### Step 6: Verify Database Entry
```python
from daq.db.database import _get_collection
collection = _get_collection()
latest = collection.find_one(sort=[("number", -1)])
print(f"✓ Latest measurement: {latest['number']} - {latest['device']}")
```

## 📊 File Structure Summary

```
daq/
├── measurements/
│   ├── __init__.py          ✅ Created
│   ├── sweep.py             ✅ Modified (+ device, filter, notes)
│   ├── timestream.py        ✅ Modified (+ device, filter, notes)
│   ├── sweep_power.py       ✅ Modified (imports only)
│   └── sweep_freq_and_dc.py ✅ Modified (imports only)
├── db/
│   ├── __init__.py          ✅ Created
│   └── database.py          ✅ Created
├── _base.py                 ✅ Modified (DB integration)
├── utils.py                 ✅ Modified (+ DATA_FOLDER)
├── analysis.py              ✅ Unchanged
└── __init__.py              ✅ Modified (new imports)

Project Root:
├── data/                    ✅ Created
├── pyproject.toml           ✅ Modified (+ pymongo)
├── requirements.txt         ✅ Created
├── README.md                ✅ Enhanced
├── MONGODB_INTEGRATION.md   ✅ Created
├── IMPLEMENTATION_CHECKLIST.md ✅ This file
└── test_db_integration.py   ✅ Created
```

## ✨ Summary

All planned features have been successfully implemented:

1. ✅ **Restructured** codebase into logical folders
2. ✅ **Created** MongoDB database integration module
3. ✅ **Modified** Base class for automatic DB logging
4. ✅ **Updated** Sweep and TimeStream classes
5. ✅ **Configured** data folder management
6. ✅ **Added** pymongo dependency
7. ✅ **Documented** everything comprehensively

The system is ready to use! Users just need to:
1. Install pymongo
2. Add `device` parameter when creating measurements
3. Optionally add `filter` and `notes`
4. Run measurements as usual

Everything else happens automatically! 🎉

