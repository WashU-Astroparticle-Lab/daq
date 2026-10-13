# Unfiltered calibration sources, 12 October 2026

All current model inputs refer to the analyzer input with no external VLFX-2500+ filters.
Amplitude is a dimensionless voltage setting. No cable-loss correction is applied.

| File | Origin and use |
| --- | --- |
| low_anchor.csv | Oct 12, 101 frequencies 0.1–2.1 GHz at amplitude 0.1, 2 s holds. From exports/Unfiltered_Lowband_2026-10-12/unfiltered_main_tone_power.csv in the laboratory collection. The builder selects 100 nodes below 2.1 GHz. |
| gap_power.csv | Oct 12, 21 frequencies 2.1–2.5 GHz, 50 logarithmic amplitudes per frequency, 0.5 s holds. Derived from measurements/speed_test_20261012T221108_430733Z/alignment_analysis/main_tone_power.csv. Select 20 nodes below 2.5 GHz. |
| high_power.csv | Unchanged copy of the Oct 2 unfiltered 2.5–9 GHz table (131 frequencies, 100 amplitudes each), also preserved in ../2026-10-02/. Select all nodes. |
| low_amplitude_checks.csv | Oct 12 independent five-amplitude sweeps at 0.1, 1.0 and 2.1 GHz, 2 s holds. Powers do not fit the model; checks at 0.1 and 1 GHz establish sparse amplitude support. |
| high_validation.csv | Oct 12 nine independent checks at 2.5, 5.2 and 9 GHz, targets −40, −30 and −20 dBm. Original commands used v1, whose high-band model is retained in v2. |
| interpolation_policy.json | Conservative exclusions and metadata, including a 1 dB adjacent-level jump review threshold. |

The selected dataset has 251 nodes and 14,200 observations. Overlap endpoints remain in the
tables for audit: 2.1 GHz uses the dense gap sweep; 2.5 GHz uses the original high-band sweep.
Separate campaigns are not averaged at the endpoints.

All source observations remain present. Gap review flags combine repeat range >0.5 dB,
±50 ms alignment sensitivity >0.5 dB or unavailable sensitivity, and local contrast <20 dB;
27 of its 1,050 rows are flagged. High-band acquisition flags are preserved. The dense-sweep
builder fits unflagged amplitudes >=0.1 and sets a supported lower amplitude above every
flagged point or absolute fit residual >0.5 dB. This is a consistency rule, not an uncertainty
bound. A single reading cannot establish repeatability.

Raw spectra and source XLS logs remain in the laboratory measurement collection. XLS files
are source timing/amplitude logs, not analyzer powers. Matching uses post-run clock alignment
and interior sweep guards, not a hardware handshake. See docs/power_calibration_update.md.

The NPZ embeds SHA-256 hashes of all six source/policy files. Git newline conversion is
disabled for CSV and JSON so these hashes survive transfer. Historical filtered low-band
powers are excluded from this model.
