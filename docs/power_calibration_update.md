# Unfiltered power calibration v2 — 12 October 2026

This proposed update replaces the filtered low-band portion of v1 with direct unfiltered
measurements, fills the 2.1–2.5 GHz gap and keeps the unfiltered 2.5–9 GHz measurements.
Both conversion function names and their GHz inputs are unchanged. It has not been installed
on the acquisition computer. Hardware spot measurements have been
compared offline with this model; they are distinct from software integration tests.

## Data and model

251 nodes cover 0.1–9 GHz: 100 low nodes below 2.1 GHz (20 MHz spacing), 20 gap nodes from
2.1 to 2.48 GHz (20 MHz), and 131 high nodes from 2.5 to 9 GHz (50 MHz). 14,200 observations
are selected. Overlap observations remain in the source files; 2.1 GHz uses the dense gap
sweep and 2.5 GHz retains the original high sweep. Endpoints from different days are not averaged.

The law is `P(A) = P1(f) + 20*log10(abs(A))`, with inverse
`A = 10**((P - P1(f))/20)`. Negative amplitude has the same predicted power.
At dense gap/high nodes, P1 is the median high-amplitude (A>=0.1), unflagged offset from this
law. At low nodes P1 is the direct A=0.1 reading plus 20 dB. There, P1 is inferred, not a
full-scale measurement. **98 low nodes only have A=0.1 measured.** Sparse independent checks
at 0.1 and 1 GHz support A=0.01–1 without refitting those anchors. The 2.1 GHz check validates
the gap model. There are 151 dense-sweep nodes in total.

The old Oct 2 low-band campaign used two cascaded VLFX-2500+ low-pass filters. Those powers
remain historical and are not used in v2. Direct unfiltered measurements avoid applying an
assumed filter correction. Sparse same-day with/without tests establish loss only at their
five measured frequencies; differences between campaigns can also include drift or setup changes.
Suppressed spurs cannot be reconstructed reliably by adding a main-tone correction.

## What the checks establish

15 independent low-band amplitude observations (five each at 0.1, 1 and 2.1 GHz) have maximum
absolute model discrepancy 0.1261 dB. Nine independent high-band checks (2.5, 5.2 and 9 GHz,
targeting −40, −30 and −20 dBm) have maximum discrepancy 0.1660 dB. The earlier v1 commands
generated the high check amplitudes; their high-band model is unchanged in v2. These points
were not used to fit P1. Low checks at two frequencies do inform supported amplitude ranges.

Dense-sweep amplitude support starts above every acquisition flag or absolute model residual
>0.5 dB. This fit-consistency rule is not an uncertainty bound, and repeated readings measure
repeatability rather than absolute accuracy. Sparse five-point validation does not directly
measure every intermediate amplitude. Full-band accuracy is not established by 24 spot points.

The reproducible JSON diagnostic also reports withheld-frequency errors. Their maxima are
0.669, 0.296 and 0.885 dB for low, gap and high bands, respectively, after interpolation
exclusions. Withholding a node tests a wider bracket than normal interpolation; these values
are diagnostics, not guaranteed errors. Do not claim a universal ±0.17 dB calibration accuracy.

## Calling the functions

```python
from daq import amp_to_power_dbm, power_dbm_to_amp, verified_amp_range

power_dbm = amp_to_power_dbm(5.2, 0.1)
amplitude = power_dbm_to_amp(5.2, -20.0)
lower, upper = verified_amp_range(5.2)
```

Use GHz unless calling the explicitly named amp_to_power_dbm_hz convenience function.
Amplitude is a dimensionless voltage fraction, not amperes. The supported amplitude range
at a single-anchor node is (0.1, 0.1). Both functions warn but return model estimates outside
the empirical range. min_verified_amp remains compatible but returns only its lower bound.
Off-grid ranges use the intersection of the two endpoint ranges, still an estimate.
The inverse rejects power above inferred full-scale power; it does not silently clip.
Zero/nonfinite amplitude and amplitudes above magnitude 1 are rejected.

## Frequency interpolation and issue #69

The functions reject interpolation across intervals touching the previously reported coarse
DAC-switch uncertainty windows, as well as adjacent fitted levels differing by >1 dB.
The latter is a conservative review threshold, not a physical boundary identification.
All 251 exact nodes work. Allowed intervals use linear interpolation of P1 in GHz; calibration
does not extend to 10 GHz. The 20 excluded open intervals are below (GHz; endpoints usable):

```
(0.94, 0.96)
(1.02, 1.04)
(2, 2.02)
(2.38, 2.4)
(2.4, 2.42)
(2.95, 3)
(3, 3.05)
(4.25, 4.3)
(4.85, 4.9)
(4.9, 4.95)
(5.55, 5.6)
(5.6, 5.65)
(6, 6.05)
(6.55, 6.6)
(6.95, 7)
(7, 7.05)
(7.05, 7.1)
(7.1, 7.15)
(7.95, 8)
(8, 8.05)
```

This prevents silent cross-step averaging but does not solve arbitrary DAC-configuration,
shared-tile or multiport dependence. The source notebook assumes port 1, automatic Mixed
mode, 40.5 mA DAC current and inverse-sinc disabled. Executed API version and per-point DAC
configuration were not frozen in the logs. Treat these as setup assumptions.

## Reference plane and acquisition limits

Power is at the Signal Hound SM200C input through the actual unfiltered cable path, without
cable-loss correction. It is not de-embedded power at the Presto connector, sample or cryostat.
Changing that path can change the answer. Powers are flat-top trace maxima within ±100 kHz
of the programmed tone, with 100 kHz RBW/VBW and average detection, summarized within a hold.
They are not integrated channel power. Source logs and analyzer acquisitions were aligned
after the run with guards; there was no hardware handshake. The new low anchors and sparse
checks used 2 s holds; the gap and original high sweep used 0.5 s holds. Acquisition flags
and original observations remain available rather than deleting inconvenient values.

## Reproduction and adoption

Run from the repository root with NumPy installed:

```text
python scripts/build_power_calibration.py
python tests/test_calibrations.py
python scripts/validate_power_calibration_update.py
```

The optional matplotlib script plot_power_calibration_update.py regenerates the diagnostic.
Source tables and policy are byte-hashed in the schema-3 NPZ; newline conversion is disabled
for those files. The runtime rejects older assets. Install the runtime and NPZ together.
The supplied standalone calibration.py plus adjacent NPZ exposes the same functions without
importing the complete DAQ package. It changes no hardware settings. The patch is prepared
against commit 9deebeb57803887097aef8146a93d6328d58b86b and should be reviewed before deployment.
