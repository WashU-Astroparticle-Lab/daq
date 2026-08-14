"""Offline verification of ``TimeStream.analyze()``'s bias-mode dispatch and reconstructions.

Checks that a stream reconstructs itself according to how the gate was biased while it was
recorded -- folded into one ramp period for a sawtooth bias, two-level parity reconstruction
plus spectrum for a constant one, and the plain time-stream plot when nothing says::

    python tests/test_timestream_analyze.py

Prints one PASS/FAIL line per check and exits non-zero if any check fails. Needs no hardware,
no VISA runtime and no MongoDB; ``presto`` is stubbed when it is absent, since nothing under
test here touches it.

The dispatch reads the generator settings ``Base.attach`` flattened onto the measurement, so
these have to hold together and each is checked separately:

- **detection**: ``bias_function`` of ``"RAMP"`` means sawtooth, ``"DC"`` means constant, and
  anything else -- including no attachment at all -- falls back to the raw plot rather than
  guessing;
- **round-trip**: ``load()`` restores those attributes, or a reloaded file analyses itself as
  ``unknown`` and the whole feature evaporates the moment the data comes off disk;
- **reconstruction**: the fold really uses the attached ramp period at the *tuned* ``df``, the
  parity spectrum really is of the mean-subtracted projection, and the flip count recovers the
  switching rate the spectral fit reports without using its model;
- **multi-tone**: every tone is reconstructed and spectrated, not just tone 0.

Two silent failures get their own checks. A ramp gated on a port the acquisition never
asserted records a static bias, folds happily, and produces a flat trace that looks like a dead
device -- exactly what ``QCTrace`` refuses outright, so here it must at least warn. And a
record with no two-level structure must be reported as unresolved rather than thresholded:
2-means splits pure Gaussian noise into "levels" 2.4 noise widths apart, so a detector that
trusted its own output would decorate structureless noise with thousands of switching events.
"""

import os
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np

# The round-trip check (section 6) goes through the real ``save()``, which asks MongoDB for a
# measurement number and falls back to a timestamp when it cannot. Point it somewhere that
# refuses immediately: the fallback is the path a no-database machine takes anyway, and the
# default URI's 30 s server-selection timeout would otherwise dominate the suite's runtime.
os.environ.setdefault("DAQ_MONGODB_URI", "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50")

# Test the checkout this file lives in, not whatever ``pip install -e`` happens to point at.
# Running as a script puts ``tests/`` on the path, not the repository root, so an editable
# install of another checkout (a git worktree, say) would otherwise be what gets exercised --
# and the suite would pass or fail describing code nobody is editing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # pragma: no cover - depends on the machine, not on the code under test
    import presto  # noqa: F401
except ImportError:
    # analyze() and load() never reach the hardware layer, so a stub keeps this suite
    # runnable on an analysis machine with no presto install.
    class _Enum:
        Mixed = "Mixed"

    _presto = types.ModuleType("presto")
    _lockin = types.ModuleType("presto.lockin")
    _lockin.Lockin = object
    _utils = types.ModuleType("presto.utils")
    _utils.untwist_downconversion = lambda i, q: (i, q)
    _utils.get_sourcecode = lambda path: [""]
    _utils.recommended_dac_config = lambda freq: {}
    _utils.asarray = lambda x: x
    _utils.rotate_opt = lambda x: x

    class _ProgressBar:
        def __init__(self, *a, **k):
            pass

        def increment(self):
            pass

        def done(self):
            pass

    _utils.ProgressBar = _ProgressBar
    _hardware = types.ModuleType("presto.hardware")
    _hardware.AdcMode = _Enum
    _hardware.DacMode = _Enum
    for _name, _module in (("lockin", _lockin), ("utils", _utils), ("hardware", _hardware)):
        setattr(_presto, _name, _module)
        sys.modules[f"presto.{_name}"] = _module
    sys.modules["presto"] = _presto
    print("INFO: presto is not installed; using a stub (analyze() does not touch it)")

import matplotlib

matplotlib.use("Agg")  # never open a window
import matplotlib.pyplot as plt

from daq.measurements.timestream import TimeStream

results = []


def check(label, condition, detail=""):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ------------------------------------------------- synthetic streams

FS = 5e4  # tuned sample rate, Hz
RAMP_HZ = 500.0  # ramp rate, a whole divisor of FS (100 samples per period)
N_PERIODS = 40
GAMMA_P = 200.0  # parity switching rate, Hz


def make_stream(signal, *, if_freqs=None, is_usb=None, **kwargs):
    """Build a TimeStream carrying *signal* (1-D, or one column per tone) as its data."""
    signal = np.asarray(signal, dtype=np.complex128)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_tones = signal.shape[1]
    if_freqs = [0.0] * n_tones if if_freqs is None else if_freqs
    ts = TimeStream(
        lo_freq=2.8e9,
        if_freqs=if_freqs,
        is_usb=is_usb,
        df=FS,
        pixel_counts=signal.shape[0],
        amp=0.01,
        output_port=1,
        input_port=1,
        discard_start_ms=0.0,
        device="TestDevice",
        **kwargs,
    )
    ts.df = FS  # what run() would leave behind after tuning
    ts.signal = signal
    ts.freqs_usb = ts.lo_freq + ts.if_freqs
    ts.freqs_lsb = ts.lo_freq - ts.if_freqs
    ts.signal_freqs = np.where(ts.is_usb, ts.freqs_usb, ts.freqs_lsb)
    ts.freq_arr = np.zeros(n_tones)
    ts.pixel_i = signal.copy()
    ts.pixel_q = signal.copy()
    ts.usb = signal.copy()
    ts.lsb = signal.copy()
    return ts


def ramp_signal(seed=0):
    """A sawtooth-modulated response: one sharp bump per ramp period, buried in noise."""
    rng = np.random.default_rng(seed)
    per_period = int(round(FS / RAMP_HZ))
    phase = np.arange(per_period) / per_period
    # A narrow feature, so a fold on the wrong period washes it out rather than shifting it.
    period_shape = np.exp(-(((phase - 0.4) / 0.03) ** 2))
    signal = np.tile(period_shape, N_PERIODS).astype(np.complex128)
    signal += 0.5 * (rng.standard_normal(signal.size) + 1j * rng.standard_normal(signal.size))
    return signal


def telegraph_signal(seed=1, n=200_000):
    """A random-telegraph magnitude at GAMMA_P, on a constant operating point."""
    rng = np.random.default_rng(seed)
    # Poisson switching: each sample flips with probability gamma_p / fs.
    flips = rng.random(n) < (GAMMA_P / FS)
    state = np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0)
    magnitude = 0.5 + 0.02 * state + 0.002 * rng.standard_normal(n)
    return magnitude.astype(np.complex128)


RAMP_SETTINGS = {
    "resource": "USB::FAKE",
    "idn": "Agilent,33220A",
    "trigger_port": 1,
    "function": "RAMP",
    "offset_v": 1.0,
    "load": "INF",
    "output": True,
    "vpp": 2.0,
    "freq_hz": RAMP_HZ,
    "symmetry_pct": 100.0,
    "burst": True,
    "burst_mode": "GAT",
    "burst_phase_deg": 180.0,
    "gate_polarity": "NORM",
}

DC_SETTINGS = {
    "resource": "USB::FAKE",
    "idn": "Agilent,33220A",
    "trigger_port": 1,
    "function": "DC",
    "offset_v": 0.7,
    "load": "INF",
    "output": True,
}

LED_SETTINGS = {
    "resource": "USB::LED",
    "idn": "Thorlabs,DC2200",
    "trigger_port": 2,
    "mode": "TTL",
    "current_limit_a": 0.1,
    "output": True,
    "ttl_current_a": 0.02,
}


# ------------------------------------------------- 1. detection

sawtooth = make_stream(ramp_signal(), external_trigger=True)
sawtooth.attach(bias=RAMP_SETTINGS)
check("1a  a ramp-biased stream reports bias_mode 'sawtooth'", sawtooth.bias_mode == "sawtooth")
check(
    "1b  its bias period is the attached ramp's",
    np.isclose(sawtooth.bias_period_s, 1.0 / RAMP_HZ),
    f"{sawtooth.bias_period_s}",
)

constant = make_stream(telegraph_signal())
constant.attach(bias=DC_SETTINGS)
check("1c  a DC-biased stream reports bias_mode 'constant'", constant.bias_mode == "constant")
check("1d  a constant bias has no ramp period", constant.bias_period_s is None)

bare = make_stream(telegraph_signal())
check("1e  an unattached stream reports bias_mode 'unknown'", bare.bias_mode == "unknown")

# An attached LED must not be read as a gate bias: the DC2200 reports `mode`, not `function`.
led_only = make_stream(telegraph_signal())
led_only.attach(led=LED_SETTINGS)
check("1f  an attached LED alone is not a bias", led_only.bias_mode == "unknown")

both = make_stream(ramp_signal(), external_trigger=True)
both.attach(bias=RAMP_SETTINGS, led=LED_SETTINGS)
check("1g  a bias plus an LED still reads the bias", both.bias_mode == "sawtooth")

# A waveform that is neither must not be guessed at.
sine = make_stream(telegraph_signal())
sine.attach(bias={**DC_SETTINGS, "function": "SIN"})
check("1h  an unrecognised carrier reports 'unknown'", sine.bias_mode == "unknown")


# ------------------------------------------------- 2. dispatch

modes = []


def record_mode(name):
    """Wrap a private reconstruction so the dispatch can be observed without drawing."""

    def wrapper(self, **kwargs):
        modes.append(name)
        return None

    return wrapper


originals = {
    name: getattr(TimeStream, f"_analyze_{name}") for name in ("raw", "sawtooth", "constant")
}
for name in originals:
    setattr(TimeStream, f"_analyze_{name}", record_mode(name))

sawtooth.analyze()
constant.analyze()
bare.analyze()
check("2a  a ramp bias dispatches to the fold", modes[0] == "sawtooth", modes[0])
check("2b  a DC bias dispatches to the spectrum", modes[1] == "constant", modes[1])
check("2c  no bias falls back to the raw plot", modes[2] == "raw", modes[2])

modes.clear()
sawtooth.analyze(mode="raw")
constant.analyze(mode="sawtooth")
check("2d  mode='raw' overrides a detected sawtooth", modes[0] == "raw", modes[0])
check("2e  mode= forces a reconstruction the bias does not imply", modes[1] == "sawtooth", modes[1])

try:
    bare.analyze(mode="folded")
    bad_mode = False
except ValueError:
    bad_mode = True
check("2f  an unknown mode raises ValueError", bad_mode)

# **fit_kwargs absorbs anything, so a fit argument on a plot that has no fit must not pass
# silently -- that is how a misspelling looks like a setting that did nothing.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sawtooth.analyze(fit_onef=True)
check("2g  fit arguments on a non-spectrum reconstruction warn", len(caught) == 1)

for name, original in originals.items():
    setattr(TimeStream, f"_analyze_{name}", original)

empty = make_stream(telegraph_signal())
empty.signal = None
try:
    empty.analyze()
    no_data = False
except RuntimeError:
    no_data = True
check("2h  analysing an unrun measurement raises RuntimeError", no_data)


# ------------------------------------------------- 3. the sawtooth reconstruction

time_ms, avg_iq = sawtooth.fold()
per_period = int(round(FS / RAMP_HZ))
check(
    "3a  fold() uses the attached ramp period", avg_iq.shape == (2, per_period), str(avg_iq.shape)
)
check(
    "3b  the folded time axis spans one period",
    np.isclose(time_ms[-1] + (time_ms[1] - time_ms[0]), 1e3 / RAMP_HZ),
)

# The point of folding: the feature survives while the noise falls as sqrt(N).
folded_i = avg_iq[0]
raw_i = np.real(np.asarray(sawtooth.signal)[:per_period, 0])
peak_sample = int(round(0.4 * per_period))
baseline = np.concatenate([folded_i[: peak_sample - 10], folded_i[peak_sample + 10 :]])
snr_folded = (folded_i[peak_sample] - baseline.mean()) / baseline.std()
raw_baseline = np.concatenate([raw_i[: peak_sample - 10], raw_i[peak_sample + 10 :]])
snr_raw = (raw_i[peak_sample] - raw_baseline.mean()) / raw_baseline.std()
check(
    "3c  folding lifts the feature out of the noise",
    snr_folded > 4 * snr_raw,
    f"SNR {snr_raw:.2f} -> {snr_folded:.2f} over {N_PERIODS} periods",
)

# Folding on the *tuned* df, not the requested rate, is what keeps blocks from smearing.
shifted = make_stream(ramp_signal(), external_trigger=True)
shifted.attach(bias=RAMP_SETTINGS)
shifted.df = FS * 1.5  # a tuned rate far from the requested one
_, avg_shifted = shifted.fold()
check(
    "3d  fold() uses the tuned df, not sampling_frequency",
    avg_shifted.shape[1] == int(round(shifted.df / RAMP_HZ)),
    str(avg_shifted.shape),
)

# Without an attached ramp there is no period to fold on, and inventing one would silently
# average the record into a meaningless shape.
try:
    bare.fold()
    folded_bare = True
except RuntimeError:
    folded_bare = False
check("3e  folding an unattached stream raises rather than guessing a period", not folded_bare)
_, avg_explicit = bare.fold(period_s=1.0 / RAMP_HZ)
check("3f  ... but an explicit period_s folds it", avg_explicit.shape == (2, per_period))

# A gated ramp on a port nothing asserted records a static bias: it must warn, not pass.
ungated = make_stream(ramp_signal(), external_trigger=False)
ungated.attach(bias=RAMP_SETTINGS)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ungated._warn_if_ramp_ungated()
check(
    "3g  a gated ramp with no port asserted warns",
    len(caught) == 1,
    str([str(w.message)[:60] for w in caught]),
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sawtooth._warn_if_ramp_ungated()
check("3h  ... and a correctly gated one does not", len(caught) == 0)

# Port 2 gated while the generator waits on port 1 is the same failure, one step subtler.
misrouted = make_stream(ramp_signal(), external_trigger=[0, 1])
misrouted.attach(bias=RAMP_SETTINGS)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    misrouted._warn_if_ramp_ungated()
check("3i  a ramp gated on the wrong port warns", len(caught) == 1)


# ------------------------------------------------- 4. the constant-bias reconstruction

f, psd = constant.parity_psd()
check("4a  parity_psd's axis runs to the Nyquist of the tuned df", np.isclose(f[-1], FS / 2))
check(
    "4b  the spectrum is of the fluctuation, not the operating point",
    psd[0] < psd[1:].max() * 1e-6,
    f"DC bin {psd[0]:.3e} against peak {psd[1:].max():.3e}",
)

series = constant._parity_series()
check("4c  the parity series is mean-subtracted", abs(series.mean()) < 1e-12 * np.abs(series).max())

try:
    constant.parity_psd(quantity="magnitude")
    bad_quantity = False
except ValueError:
    bad_quantity = True
check("4d  an unknown quantity raises ValueError", bad_quantity)

fit = constant.fit_parity()
check(
    "4e  fit_parity recovers the telegraph rate",
    fit["success"] and abs(fit["gamma_p"] - GAMMA_P) < 0.35 * GAMMA_P,
    f"gamma_p = {fit['gamma_p']:.1f} Hz against {GAMMA_P:.0f} Hz",
)
check("4f  the fit holds f_bw at the tuned df", np.isclose(fit["f_bw"], FS))
check("4g  the fit lands on fit_results", constant.fit_results is fit)


# ------------------------------------------------- 5. the time-domain reconstruction

# The flip count is an estimate of the same rate the spectral fit reports, arrived at without
# the model. Agreeing with the truth *and* with the fit is what makes either believable.
rec = constant.reconstruct_parity()
check(
    "5a  the flip rate recovers the telegraph rate",
    abs(rec["gamma_p_flips"] - GAMMA_P) < 0.2 * GAMMA_P,
    f"gamma_p_flips = {rec['gamma_p_flips']:.1f} Hz against {GAMMA_P:.0f} Hz",
)
check("5b  the two levels are resolved", rec["separated"], f"snr = {rec['snr']:.1f}")
check(
    "5c  the state assignment spans the record",
    rec["state"].shape[0] == np.asarray(constant.signal).shape[0],
)
check(
    "5d  flip times are inside the record",
    rec["flip_times_s"].size == rec["n_flips"] and (rec["flip_times_s"] <= rec["duration_s"]).all(),
)
check("5e  a quiet record reports no bursts", rec["bursts"] == [], str(rec["bursts"]))

# A record with no telegraph at all must not be thresholded into thousands of "flips" and
# presented as a reconstruction -- 2-means splits pure noise happily, and the resulting picket
# fence looks exactly like a fast switcher.
flat = make_stream(0.5 + 0.002 * np.random.default_rng(11).standard_normal(200_000))
flat.attach(bias=DC_SETTINGS)
flat_rec = flat.reconstruct_parity()
check(
    "5f  structureless noise is reported as unresolved, not reconstructed",
    not flat_rec["separated"],
    f"snr = {flat_rec['snr']:.2f}",
)
check("5g  ... and no bursts are hunted through it", flat_rec["bursts"] == [])

# An injected burst: 10x the switching rate for 200 ms in the middle of the record.
burst_seed = np.random.default_rng(12)
flips = burst_seed.random(200_000) < (GAMMA_P / FS)
lo, hi = int(2.0 * FS), int(2.2 * FS)
flips[lo:hi] = burst_seed.random(hi - lo) < (10 * GAMMA_P / FS)
state = np.cumsum(flips) % 2 == 0
bursty = make_stream(
    0.5 + 0.02 * np.where(state, 1.0, -1.0) + 0.002 * burst_seed.standard_normal(200_000)
)
bursty.attach(bias=DC_SETTINGS)
bursty_rec = bursty.reconstruct_parity()
found = bursty_rec["bursts"]
check("5h  an injected rapid-switching burst is found", len(found) == 1, f"{len(found)} found")
check(
    "5i  ... at the time it was injected",
    bool(found) and found[0]["start_s"] < 2.1 < found[0]["end_s"],
    str([(round(b["start_s"], 2), round(b["end_s"], 2)) for b in found]),
)
check(
    "5j  ... with an elevated rate inside it",
    bool(found) and found[0]["rate_hz"] > 3 * bursty_rec["gamma_p_flips"],
    (
        f"{found[0]['rate_hz']:.0f} Hz inside against {bursty_rec['gamma_p_flips']:.0f} Hz overall"
        if found
        else ""
    ),
)


# ------------------------------------------------- 6. multi-tone

# The whole complaint that prompted this: a two-tone acquisition reconstructed only tone 0.
two_tone_signal = np.stack(
    [telegraph_signal(seed=1), 0.5 + 0.002 * np.random.default_rng(13).standard_normal(200_000)],
    axis=1,
)
two_tone = make_stream(two_tone_signal, if_freqs=[0.0, 10e6], is_usb=[True, False])
two_tone.attach(bias=DC_SETTINGS)

f2, psd2 = two_tone.parity_psd()
check("6a  parity_psd returns one spectrum per tone", psd2.shape == (2, f2.size), str(psd2.shape))
check(
    "6b  a named tone still returns one spectrum",
    two_tone.parity_psd(tone=1)[1].ndim == 1,
)
fits2 = two_tone.fit_parity()
check("6c  fit_parity returns one fit per tone", isinstance(fits2, list) and len(fits2) == 2)
recs2 = two_tone.reconstruct_parity()
check(
    "6d  reconstruct_parity returns one result per tone",
    isinstance(recs2, list) and len(recs2) == 2,
)
check(
    "6e  ... and each knows which tone it is",
    [r["tone"] for r in recs2] == [0, 1],
)
check(
    "6f  ... the telegraph tone resolves and the reference does not",
    recs2[0]["separated"] and not recs2[1]["separated"],
    f"snr {recs2[0]['snr']:.1f} vs {recs2[1]['snr']:.1f}",
)
try:
    two_tone.parity_psd(tone=5)
    bad_tone = False
except IndexError:
    bad_tone = True
check("6g  an out-of-range tone raises IndexError", bad_tone)


# ------------------------------------------------- 7. the figures themselves

plt.close("all")
fig = sawtooth.analyze()
check("7a  the sawtooth reconstruction draws a row of I/Q per tone", len(fig.axes) == 2)
check(
    "7b  ... over one ramp period", "QC trace" in fig._suptitle.get_text(), fig._suptitle.get_text()
)

fig = constant.analyze()
check(
    "7c  the constant reconstruction keeps the two-column grid, plus a spectrum", len(fig.axes) == 3
)
check(
    "7d  ... streams on linear time above a log-log spectrum",
    fig.axes[0].get_xscale() == "linear" and fig.axes[-1].get_xscale() == "log",
)

fig = two_tone.analyze()
check("7e  a two-tone constant stream draws 2x2 panels and one shared spectrum", len(fig.axes) == 5)
fig = two_tone.analyze(tone=0)
check("7f  ... restricted to one tone on request", len(fig.axes) == 3)

fig = bare.analyze()
check("7g  the raw plot is unchanged: one row of I/Q per tone", len(fig.axes) == 2)
plt.close("all")


# ------------------------------------------------- 8. the HDF5 round trip

# Without this the whole feature stops at the end of the session that took the data.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "roundtrip.h5")
    sawtooth.save(save_filename=path)
    reloaded = TimeStream.load(path)

check("8a  a reloaded stream still knows it was ramp-biased", reloaded.bias_mode == "sawtooth")
check(
    "8b  ... at the same ramp frequency",
    np.isclose(reloaded.bias_period_s, 1.0 / RAMP_HZ),
    f"{reloaded.bias_period_s}",
)
check("8c  ... and remembers its device", reloaded.device == "TestDevice")
check(
    "8d  ... and the generator's trigger port, so the ungated check still runs",
    int(reloaded.bias_settings["trigger_port"]) == 1,
)
check(
    "8e  the reloaded record folds to the same trace",
    np.allclose(reloaded.fold()[1], avg_iq),
)


print()
print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
