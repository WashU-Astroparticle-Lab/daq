"""Offline verification of ``TimeStream.analyze()``'s bias-mode dispatch.

Checks that a stream reconstructs itself according to how the gate was biased while it was
recorded -- folded into one ramp period for a sawtooth bias, spectrum-and-parity-fit for a
constant one, and the plain time-stream plot when nothing says::

    python tests/test_timestream_analyze.py

Prints one PASS/FAIL line per check and exits non-zero if any check fails. Needs no hardware,
no VISA runtime and no MongoDB; ``presto`` is stubbed when it is absent, since nothing under
test here touches it.

The dispatch reads the generator settings ``Base.attach`` flattened onto the measurement, so
three things have to hold together and each is checked separately:

- **detection**: ``bias_function`` of ``"RAMP"`` means sawtooth, ``"DC"`` means constant, and
  anything else -- including no attachment at all -- falls back to the raw plot rather than
  guessing;
- **round-trip**: ``load()`` restores those attributes, or a reloaded file analyses itself as
  ``unknown`` and the whole feature evaporates the moment the data comes off disk;
- **reconstruction**: the fold really uses the attached ramp period at the *tuned* ``df``, and
  the parity spectrum really is of the mean-subtracted projection.

The gated-but-untriggered case gets its own check. A ramp gated on a port the acquisition
never asserted records a static bias, folds happily, and produces a flat trace that looks like
a dead device -- exactly the silent failure ``QCTrace`` refuses outright, so here it must at
least warn.
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


def make_stream(signal, **kwargs):
    """Build a single-tone TimeStream carrying *signal* as its acquired data."""
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1, 1)
    ts = TimeStream(
        lo_freq=2.8e9,
        if_freqs=[0.0],
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
    ts.signal_freqs = np.array([2.8e9])
    ts.freq_arr = np.array([0.0])
    ts.pixel_i = signal.copy()
    ts.pixel_q = signal.copy()
    ts.usb = signal.copy()
    ts.lsb = signal.copy()
    ts.freqs_usb = np.array([2.8e9])
    ts.freqs_lsb = np.array([2.8e9])
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


# ------------------------------------------------- 5. the figures themselves

plt.close("all")
fig = sawtooth.analyze()
check("5a  the sawtooth reconstruction draws two panels", len(fig.axes) == 2)
check(
    "5b  ... over one ramp period", "QC trace" in fig._suptitle.get_text(), fig._suptitle.get_text()
)

fig = constant.analyze()
check("5c  the constant reconstruction draws two panels", len(fig.axes) == 2)
check(
    "5d  ... time series above spectrum",
    fig.axes[1].get_xscale() == "log" and fig.axes[0].get_xscale() == "linear",
)

fig = bare.analyze()
check("5e  the raw plot is unchanged: one row of I/Q per tone", len(fig.axes) == 2)
plt.close("all")


# ------------------------------------------------- 6. the HDF5 round trip

# Without this the whole feature stops at the end of the session that took the data.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "roundtrip.h5")
    sawtooth.save(save_filename=path)
    reloaded = TimeStream.load(path)

check("6a  a reloaded stream still knows it was ramp-biased", reloaded.bias_mode == "sawtooth")
check(
    "6b  ... at the same ramp frequency",
    np.isclose(reloaded.bias_period_s, 1.0 / RAMP_HZ),
    f"{reloaded.bias_period_s}",
)
check("6c  ... and remembers its device", reloaded.device == "TestDevice")
check(
    "6d  ... and the generator's trigger port, so the ungated check still runs",
    int(reloaded.bias_settings["trigger_port"]) == 1,
)
check(
    "6e  the reloaded record folds to the same trace",
    np.allclose(reloaded.fold()[1], avg_iq),
)


print()
print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
