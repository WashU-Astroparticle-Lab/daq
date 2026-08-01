# Database

Every measurement writes an HDF5 file to `DAQ_DATA_FOLDER` and inserts one metadata document
into MongoDB. The document is what makes a run findable later: the HDF5 file holds the data,
the document holds everything you would otherwise have to remember.

Logging is best-effort — if the configured server is unreachable, the acquisition still runs
and the file is still saved (`get_next_number()` falls back to a timestamp).

Defaults (override via the environment, see the [top-level README](../../README.md#configuration)):

| | |
|---|---|
| URI | `mongodb://localhost:27017` |
| Database | `WashU_Astroparticle_Detector` |
| Collection | `measurement` |

## File naming

Data files are named `{number}-{device}-{type}.h5`, e.g. `00000042-Resonator_A-sweep.h5`,
where `number` is the 8-digit cumulative measurement number also stored in the document.

## Document structure

Common to every measurement:

| Field | Meaning |
|---|---|
| `utc_time` | UTC timestamp (ISO string) |
| `number` | 8-digit cumulative measurement number |
| `type` | `"sweep"`, `"timestream"`, `"qc_trace"`, `"bias_hunt"`, … |
| `device` | Device name (required) |
| `filter` | Filter name (optional) |
| `notes` | Free-text notes (optional) |
| `file` | Full path to the HDF5 file |
| `output_port`, `input_port`, `amp`, … | All measurement-specific parameters |

Calibrated power is added automatically:

| Field | Measurement |
|---|---|
| `power_dbm` | `Sweep`, `SweepFreqAndDC`, `QCTrace`, `BiasHunt` (scalar); `TimeStream` (per-tone list) |
| `power_dbm_arr` | `SweepPower` (drive power array) |
| `readout_power_dbm`, `control_power_dbm_arr` | `TwoTonePower` |

Fit results are added when auto-fitting succeeds (`Sweep`; `QCTrace` and `BiasHunt` carry none
of their own, since neither fits a resonance — they read out where you tell them to):
`fit_fr`, `fit_Qi`, `fit_Qc`, `fit_Ql` with their `_err` counterparts, plus `fit_kappa = fr/Qc`.
These keys are simply absent when `auto_fit=False` or the fit fails — filter on them with
`df['fit_fr'].notna()` rather than assuming they exist.

Auxiliary instrument state lands in the document too, if you call
[`Base.attach()`](../instruments/README.md) before `run()`: each instrument's `settings()` is
flattened to `<prefix>_<key>` scalars (`bias_offset_v`, `led_pwm_count`, …), which makes bias
voltages and LED parameters queryable instead of buried in a `notes` string.

## Querying

```python
from daq.db import select_runs, list_devices
```

`select_runs(**kwargs)` returns a `pandas.DataFrame` of matching documents (empty if none
match), with `_id` stripped:

```python
# By device, by type, or both
df = select_runs(device="Resonator_A")
df = select_runs(measurement_type="sweep")
df = select_runs(device="Resonator_A", measurement_type="sweep")

# Any other document field works as a filter
df = select_runs(measurement_type="sweep", freq_center=5e9, amp=0.1)
```

Time ranges take ISO strings or `datetime` objects:

```python
from datetime import datetime

df = select_runs(start_time="2026-01-01T00:00:00", end_time="2026-06-30T23:59:59")
df = select_runs(device="Resonator_A", start_time=datetime(2026, 1, 1))
```

String fields (`device`, `filter_name`, `notes`, `measurement_type`, and any string `kwargs`)
match exactly by default; `string_match="regex"` switches to case-insensitive regex:

```python
df = select_runs(device="Resonator", string_match="regex")   # every Resonator_*
df = select_runs(notes="cooldown", string_match="regex")
```

Note the keyword is **`filter_name`**, not `filter` — the document field is `filter`, but the
Python name would shadow the builtin.

Numeric ranges are not expressible in the query; filter the DataFrame afterwards:

```python
df = select_runs(measurement_type="sweep")
near_5ghz = df[df["fit_fr"].notna() & df["fit_fr"].between(4.9e9, 5.1e9)]
```

`list_devices()` returns a `('device', 'count')` DataFrame sorted by count, descending — the
quickest way to see what has been measured:

```python
list_devices()
#         device  count
# 0  Resonator_A     42
# 1  Resonator_B     31
```

## Caveats when querying older runs

Schema changes are not retroactive, so a filter on a field only matches records of the right
vintage. The known case: `external_trigger` was a bool before it became a per-port list, so
`select_runs(external_trigger=True)` matches only pre-change records.
