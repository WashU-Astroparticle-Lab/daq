# daq

Stable wrappers for data acquisition on the Intermodulation Products **Presto-8**, used at the
WashU Astroparticle Lab.

A measurement class configures the Presto, runs an acquisition, optionally fits the result,
writes an HDF5 file and logs a metadata document to MongoDB — so a run is reproducible and
findable months later. Around that sit drivers for the benchtop instruments, a power
calibration, and the analysis layer (resonator fits, noise PSDs, basis transforms).

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
    device="Resonator_A",   # required: names the device in the database
    notes="Cooldown test",
)
filepath = sweep.run()      # acquires, fits, saves HDF5, logs to MongoDB
sweep.analyze()             # plot
```

## Where to look

| If you want to… | Read |
|---|---|
| Run a measurement — parameters, `run()`/`analyze()`/`load()` | [`daq/measurements/README.md`](daq/measurements/README.md) |
| Analyse data — resonator fits, PSDs, parity, folding, plots | [`daq/analysis/README.md`](daq/analysis/README.md) |
| Drive the function generator or LED driver over VISA | [`daq/instruments/README.md`](daq/instruments/README.md) |
| Find past runs, or know what a document contains | [`daq/db/README.md`](daq/db/README.md) |
| Know what the Presto hardware can and cannot do | [`CLAUDE.md`](CLAUDE.md) — spec-sheet and API facts, with provenance |

The two authoritative sources for anything about the hardware itself are the
[presto API manual](https://www.intermod.pro/manuals/presto/index.html) and the
[Presto spec sheet](https://intermod.pro/res/docs/presto_spec_sheet.pdf). `CLAUDE.md` records
what has been established from them, and marks what is still unverified.

## Install

```bash
pip install -e .
```

Core requirements: `numpy`, `scipy`, `h5py`, `matplotlib`, `pandas`, `pymongo`, `presto`.
Two optional extras, both imported lazily so `import daq` works without them:

```bash
pip install -e ".[analysis]"      # resonator_tools -- resonator circle fitting
pip install -e ".[instruments]"   # pyvisa -- benchtop instruments (needs a VISA runtime)
```

Use the **upstream** `resonator_tools` from PyPI. The old WashU fork
(`FaroutYLq/resonator_tools`) is no longer required: `daq.analysis.resonator` recovers the
environmental term through the package's public `do_calibration()` API, bit-for-bit identically.

The lab machine runs everything inside the `presto` conda environment:

```bash
conda activate presto
```

## Configuration

Everything is read from the environment (see [`daq/config.py`](daq/config.py)); settings are
cached on first use, `reload_settings()` picks up changes.

| Variable | Default |
|---|---|
| `DAQ_PRESTO_ADDRESS` | `172.23.20.29` |
| `DAQ_PRESTO_PORT` | Presto default |
| `DAQ_DATA_FOLDER` | `<repo>/data` |
| `DAQ_MONGODB_URI` | `mongodb://localhost:27017` |
| `DAQ_MONGODB_DB_NAME` | `WashU_Astroparticle_Detector` |
| `DAQ_MONGODB_COLLECTION_NAME` | `measurement` |
| `DAQ_FGEN_RESOURCE`, `DAQ_LED_RESOURCE` | autodiscover |
| `DAQ_FGEN_TRIGGER_PORT`, `DAQ_LED_TRIGGER_PORT` | driver defaults (1, 2) |
| `DAQ_VISA_BACKEND` | system NI-VISA |

## Layout

```
daq/
├── measurements/   Sweep, TimeStream, SweepPower, SweepFreqAndDC, TwoTonePower, QCTrace, BiasHunt
├── analysis/       resonator fits, noise PSDs, folding, plotting, Mattis-Bardeen
├── instruments/    VISA drivers: Agilent33220A (gate bias), DC2200 (LED)
├── db/             MongoDB logging and querying
├── calibrations/   amp <-> dBm power calibration
├── triggers.py     which Presto digital port gates which instrument
├── config.py       runtime configuration
└── _base.py        Base: HDF5 + MongoDB save, attach() for auxiliary instrument state
```

`daq/utils.py` is a backward-compatible facade; new code should use `daq.config` and
`daq.time_utils`.

Power calibration is exported at top level — `amp_to_power_dbm(freq_ghz, amp)` and
`power_dbm_to_amp(freq_ghz, power_dbm)` — so a measurement can be set up in dBm
(`amp=power_dbm_to_amp(5.0, -20.0)`). Here `amp` is a full-scale voltage
fraction, so its power dependence is `20 log10(amp)`; `SweepPower` and
`TwoTonePower` label their axes with the calibration. Any `0 < |amp| <= 1` converts, but
below `min_verified_amp(freq_ghz)` — the amplitude where the spectrum analyzer's own floor
started hiding the tone during calibration — the conversion is an extrapolation of that law
and raises a `CalibrationWarning` to say so.

The spectrum-analyzer sweeps the asset was built from are committed under
`daq/calibrations/source_data/`. Regenerate the packaged asset—and optionally its diagnostic
figure—with:

```bash
python scripts/build_power_calibration.py --diagnostic-plot docs/power_calibration_diagnostic.png
```

Two more figures document the calibration and can be regenerated the same way:
`docs/power_calibration_interpolation.png` (`scripts/plot_power_calibration_interpolation.py`)
shows how the conversions behave at frequencies with no calibration data — the interpolated
full-scale power and verified floor across the band, every DAC-configuration switch, and nine
off-grid frequencies drawn between their calibrated neighbours — and is the plot to check after
any rebuild. `docs/power_calibration_before_after.png`
(`scripts/plot_power_calibration_before_after.py`) compares it with the four-corner grid it
replaced.

## Tests

There is no CI. `tests/` is an offline verification suite: no hardware, no VISA runtime, no
MongoDB. Each script prints one PASS/FAIL line per check and exits non-zero on failure.

```bash
python tests/test_calibrations.py && python tests/test_instruments.py && python tests/test_timestream_run.py && python tests/test_timestream_analyze.py && python tests/test_resonator.py && python tests/test_qc_trace.py && python tests/test_bias_hunt.py && python tests/test_plotting.py
```

They are standalone scripts, not pytest suites — run them as scripts. Extend them when
touching the instruments layer, `TimeStream.run()` or `analyze()`'s bias dispatch, `QCTrace`'s
trigger routing, `BiasHunt`'s ranking, the resonator fitting adapter, or the plotting helpers.

## Conventions

Sphinx-style docstrings and complete type annotations on public APIs (`Optional[T]` where
`None` is allowed), Black at 100 columns:

```bash
black --line-length 100 daq/
```

When you add a measurement class, analysis tool or instrument driver, update the matching
README above and the Architecture section of `CLAUDE.md`.
