"""Offline verification of ``daq.analysis.led_response`` on a synthetic flash train.

The flashes are software-started, so their phase in a record is unknown until read off a
marker tone. These checks inject a train at a known phase into a KID-like trace and verify::

    python tests/test_led_response.py

- ``find_pulse_comb`` recovers the onset phase to a sample from the *folded* profile, even
  when a single flash is below the noise, and reports the SNRs;
- ``average_pulse`` recovers the injected pulse shape;
- ``fold_events`` keeps complete cycles only and drops events before the first flash;
- ``folded_event_rate`` returns the injected baseline and excess with Poisson errors, and
  ``folded_fraction`` the suppressed fraction.

Loads the module by path; needs numpy only.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "led_response", ROOT / "daq" / "analysis" / "led_response.py"
)
lr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lr  # dataclasses resolve annotations through sys.modules
spec.loader.exec_module(lr)

results = []


def check(label, condition, detail=""):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


FS, DUR, PERIOD, PHASE = 100e3, 5.0, 50e-3, 45.3e-3
TAU, AMP, NOISE = 200e-6, -3e-4, 1e-4  # single-flash SNR 3, folded ~30 over 100 flashes
rng = np.random.default_rng(1)
n = int(DUR * FS)
t = np.arange(n) / FS
template_t = np.arange(int(2e-3 * FS)) / FS
template = AMP * np.exp(-template_t / TAU)
x = NOISE * rng.standard_normal(n)
onsets_true = PHASE + PERIOD * np.arange(int(np.floor((DUR - PHASE) / PERIOD)) + 1)
onsets_true = onsets_true[onsets_true < DUR]
for t0 in onsets_true:
    i = int(round(t0 * FS))
    m = min(template.size, n - i)
    x[i : i + m] += template[:m]

# ---------------------------------------------------------------- comb

comb = lr.find_pulse_comb(x, FS, PERIOD)
check(
    "phase recovered to within a sample",
    abs(comb.phase_s - PHASE) <= 1.5 / FS,
    f"{comb.phase_s * 1e3:.3f} ms",
)
check(
    "all flashes enumerated",
    comb.times.size == onsets_true.size and np.allclose(comb.times, onsets_true, atol=1.5 / FS),
    f"{comb.times.size} vs {onsets_true.size}",
)
check("n_flashes counts complete periods", comb.n_flashes == int(n // round(PERIOD * FS)))
check("single-flash SNR is about 3", 2.0 < comb.snr_single < 4.5, f"{comb.snr_single:.2f}")
check(
    "folded SNR is sqrt(n) higher",
    abs(comb.snr_folded / comb.snr_single - np.sqrt(comb.n_flashes)) < 1e-9,
)
check("detected", comb.detected)
check("peak is negative (the injected sign)", comb.peak < 0)
check(
    "profile has one period of samples",
    comb.profile.size == round(PERIOD * FS) and comb.profile_t[-1] < PERIOD,
)

quiet = lr.find_pulse_comb(NOISE * rng.standard_normal(n), FS, PERIOD)
check("pure noise is not detected", not quiet.detected, f"folded SNR {quiet.snr_folded:.1f}")

shifted = lr.find_pulse_comb(x[100:], FS, PERIOD, record_start_s=100 / FS)
check("record_start_s offsets the returned times", abs(shifted.phase_s - PHASE) <= 1.5 / FS)

# ---------------------------------------------------------------- gate ripple
# A marker beside a ramped array carries the gate pickup. The flash period is a whole number
# of gate periods, so the ripple folds coherently at every gate-period phase and a weak flash
# loses to it. remove_period_s subtracts the gate fold first.
GATE = 1 / 5e3
ripple = 2.5e-4 * np.sin(2 * np.pi * t / GATE) + 1e-4 * np.sin(4 * np.pi * t / GATE + 0.3)
x_weak = NOISE * rng.standard_normal(n) + ripple
for t0 in onsets_true:
    i = int(round(t0 * FS))
    m = min(template.size, n - i)
    x_weak[i : i + m] += 0.25 * template[:m]  # single-flash SNR 0.75, folded ~7.5
# Noise plus ripple, no flash: without removal the ripple alone "detects" a train (the real
# failure mode -- a dark file scoring a folded SNR far above the acceptance threshold); with
# the gate fold removed it does not.
x_ripple_only = NOISE * rng.standard_normal(n) + ripple
fooled = lr.find_pulse_comb(x_ripple_only, FS, PERIOD)
cleaned_quiet = lr.find_pulse_comb(x_ripple_only, FS, PERIOD, remove_period_s=GATE)
check(
    "ripple alone is 'detected' without removal and not with it",
    fooled.snr_folded >= 5.0 and not cleaned_quiet.detected,
    f"SNR {fooled.snr_folded:.1f} -> {cleaned_quiet.snr_folded:.1f}",
)
cleaned = lr.find_pulse_comb(x_weak, FS, PERIOD, remove_period_s=GATE)
check(
    "with the gate fold removed the weak flash is found",
    abs(cleaned.phase_s - PHASE) <= 2 / FS,
    f"phase {cleaned.phase_s * 1e3:.3f} ms, SNR {cleaned.snr_folded:.1f}",
)
strong = lr.find_pulse_comb(x + ripple, FS, PERIOD, remove_period_s=GATE)
check(
    "removal leaves a strong flash's phase and SNR intact",
    abs(strong.phase_s - PHASE) <= 1.5 / FS and abs(strong.snr_folded / comb.snr_folded - 1) < 0.3,
    f"SNR {strong.snr_folded:.1f} vs {comb.snr_folded:.1f}",
)
for bad in (PERIOD, -1.0, 0.0):
    try:
        lr.find_pulse_comb(x, FS, PERIOD, remove_period_s=bad)
        check(f"remove_period_s={bad} is refused", False)
    except ValueError:
        check(f"remove_period_s={bad} is refused", True)

# ---------------------------------------------------------------- average pulse

tt, mean, std, n_win = lr.average_pulse(x, FS, comb.times, pre_s=0.5e-3, post_s=2e-3)
check(
    "windows fit: all but any that run off the end",
    n_win == comb.times.size - int(comb.times[-1] + 2e-3 > DUR),
)
k0 = int(round(0.5e-3 * FS))
check("time axis is zero at onset", tt[k0] == 0.0 and tt[0] == -k0 / FS)
resid = mean[k0 : k0 + template.size] - template
check(
    "mean pulse matches the template within the averaged noise",
    np.abs(resid).max() < 5 * NOISE / np.sqrt(n_win),
    f"max resid {np.abs(resid).max():.2e}",
)
check("pre-onset baseline is flat", abs(mean[:k0].mean()) < 1e-6)
check(
    "std is per-sample across windows",
    std.shape == tt.shape and 0.5 * NOISE < np.median(std) < 1.5 * NOISE,
)

# ---------------------------------------------------------------- folding events

GATE = 1 / 5e3
base_rate = 40.0  # events per second per record, Poisson
n_base = rng.poisson(base_rate * DUR)
events = np.sort(rng.uniform(0, DUR, n_base))
# Excess: 3 extra events within 2 ms after every flash.
excess = np.concatenate([o + rng.uniform(0, 2e-3, 3) for o in onsets_true])
events = np.sort(np.concatenate([events, excess]))
folded, n_cycles = lr.fold_events(events, comb.phase_s, PERIOD, DUR)
check(
    "complete cycles only", n_cycles == int(np.floor((DUR - comb.phase_s) / PERIOD)), str(n_cycles)
)
check("folded times live in [0, period)", folded.min() >= 0 and folded.max() < PERIOD)
n_kept_expected = np.sum((events >= comb.phase_s) & (events < comb.phase_s + n_cycles * PERIOD))
check(
    "events before the first flash and in the incomplete tail are dropped",
    folded.size == n_kept_expected,
)

rate, err, edges = lr.folded_event_rate(folded, n_cycles, PERIOD, bins=250)
check("one gate period per bin", np.allclose(np.diff(edges), GATE))
late = rate[50:]  # > 10 ms after the flash: baseline only
check(
    "baseline rate recovered",
    abs(late.mean() - base_rate)
    < 4 * base_rate / np.sqrt(late.size * n_cycles * GATE * base_rate + 1) + 3,
    f"{late.mean():.1f} vs {base_rate}",
)
early = rate[:10].mean()  # first 2 ms: 3 events per cycle over 10 bins -> +3/(10 * GATE) Hz per bin
check(
    "excess after the flash is where it was injected",
    early > base_rate + 0.5 * (3 / (10 * GATE)),
    f"{early:.0f} Hz early vs {late.mean():.0f} Hz late",
)
check("Poisson errors", np.allclose(err * n_cycles * GATE, np.sqrt(rate * n_cycles * GATE)))

frac, ferr, fedges = lr.folded_fraction(
    np.tile((np.arange(250) + 0.5) * GATE, n_cycles), n_cycles, PERIOD, GATE, bins=250
)
check(
    "every period suppressed gives fraction 1 with zero error",
    np.allclose(frac, 1.0) and np.allclose(ferr, 0.0, atol=1e-6),
)
try:
    lr.folded_event_rate(folded, 0, PERIOD)
    check("zero cycles is refused", False)
except ValueError:
    check("zero cycles is refused", True)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
