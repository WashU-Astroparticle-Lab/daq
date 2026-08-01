"""Offline verification of ``BiasHunt``.

The bias hunt parks the gate at a series of constant voltages and ranks them by parity
contrast. Its failure modes are the mirror image of ``QCTrace``'s:

- every acquisition must be **ungated**. The bias is a DC level already written over SCPI, so
  a trigger asserted here would gate whatever else is wired to that port for the whole hunt.
- the winner must be the argmax of ``std(|signal|)``, and ``best_bias`` / ``best_bias_file`` /
  ``best_bias_stream`` must all name the *same* try -- three ways of saying it is three ways
  to disagree.
- the voltages are drawn in ``__init__``, so the object pins down what was measured before any
  hardware is touched, and a seed reproduces the draw.
- there is no default gate range: a missing ``v_min``/``v_max`` must raise rather than put an
  invented voltage on somebody's device.
- the contrast curve, the winner and the constituent file paths survive the HDF5 round trip.

The averaged spectrum and its parity fit are checked against a *known* answer rather than
against themselves: the stand-in stream is a genuine random-telegraph process switching at
``GAMMA_P``, built from exponential dwell times rather than hand-shaped to match the model, so
``fit_psd`` has to recover a number it was never told. The averaging, the mean subtraction, the
use of the tuned rate and the invalidation of a stale fit are each pinned separately.

Requires ``presto`` to be importable (``BiasHunt`` imports it transitively); no hardware and no
network. The database calls are stubbed so the round-trip check does not sit through MongoDB's
server-selection timeout. Run from the repository root::

    python tests/test_bias_hunt.py
"""

import os
import sys
import tempfile

try:
    import presto  # noqa: F401
except ImportError:
    print("SKIP: presto is not installed; BiasHunt cannot be imported without it")
    sys.exit(0)

import matplotlib

matplotlib.use("Agg")  # the analyze() checks below must not need a display
import matplotlib.pyplot  # noqa: E402
import numpy as np  # noqa: E402

import daq._base as base_mod  # noqa: E402
import daq.measurements._gate_bias as gate_bias_mod  # noqa: E402
from daq.analysis.noise import compute_psd  # noqa: E402
from daq.measurements.bias_hunt import BiasHunt  # noqa: E402

base_mod.get_next_number = lambda: "00000001"
base_mod.insert_measurement = lambda document: "offline"

results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- stand-ins

FS = 5e4
#: Parity-switching rate of the synthetic response, in hertz. The spectrum checks below fit
#: for this number, so the fit is pinned against a known answer rather than against itself.
GAMMA_P = 120.0


class FakeTimeStream:
    """A genuine random-telegraph response whose contrast depends on the bias it sits at.

    The signal switches between two magnitudes at :data:`GAMMA_P`, so its PSD really is the
    Lorentzian the parity model describes, and ``std(|signal|)`` really does measure the
    switching amplitude. The amplitude peaks at ``PEAK_V``, giving the hunt an unambiguous
    winner. Both the ranking and the spectral fit are therefore checked against known answers.
    """

    instances = []
    PEAK_V = 1.25

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.df = kwargs["df"]
        n = int(kwargs["pixel_counts"])
        # The note carries the bias; parse it rather than threading extra state through.
        voltage = float(kwargs["notes"].split("Constant bias ")[1].split(" V")[0])
        spread = np.exp(-(((voltage - self.PEAK_V) / 0.3) ** 2))
        # Seed from the voltage, NOT from hash(notes): Python randomises string hashing per
        # process, so a hash-derived seed silently regenerates different data on every run.
        # The spectral checks below then fluctuate by tens of hertz between runs and pass only
        # on the width of their tolerance -- which is a coin flip dressed as a test.
        rng = np.random.default_rng(int(round(abs(voltage) * 1e6)) % 2**32)
        # Switch with probability Gamma/fs per sample -> exponential dwell times, i.e. a real
        # telegraph process rather than a spectrum hand-built to match the model.
        telegraph = np.where(np.cumsum(rng.random(n) < GAMMA_P / self.df) % 2 == 0, 1.0, -1.0)
        level = 1.0 + 0.02 * spread * telegraph
        noise = 0.002 * rng.standard_normal(n)  # readout noise -> the white floor
        self.signal = ((level + noise) + 0j).reshape(n, 1)
        FakeTimeStream.instances.append(self)

    def attach(self, **instruments):
        pass

    def run(self, **kwargs):
        return f"/tmp/ts{len(FakeTimeStream.instances)}.h5"


class FakeBias:
    """A 33220A stand-in that records the voltages it was told to hold."""

    def __init__(self, trigger_port=1):
        self.trigger_port = trigger_port
        self.output = False
        self.calls = []

    def constant(self, voltage):
        self.calls.append(("constant", voltage))

    def sawtooth(self, **kwargs):
        self.calls.append(("sawtooth", kwargs.get("gated")))


gate_bias_mod.TimeStream = FakeTimeStream

# Stub the save at *class* level, not per instance: an instance attribute would itself land in
# the saved record, since Base._save walks __dict__. The real one is kept for the round trip.
save_for_real = BiasHunt.save
BiasHunt.save = lambda self, save_filename=None: "/dev/null"  # type: ignore[assignment]


def make(**kwargs):
    params = dict(
        readout_freq=2.8e9,
        amp=0.01,
        output_port=1,
        input_port=1,
        v_min=0.0,
        v_max=2.0,
        n_bias_try=8,
        # Long enough that the telegraph switches ~GAMMA_P * ts_duration_s = 120 times, so
        # std(|signal|) measures the switching amplitude rather than the luck of a record that
        # happened to contain one transition.
        ts_duration_s=1.0,
        sampling_frequency=FS,
        discard_start_ms=0.0,
        seed=7,
        device="offline-test",
    )
    params.update(kwargs)
    return BiasHunt(**params)


def run_hunt(hunt, bias=None):
    FakeTimeStream.instances = []
    hunt.run(bias=bias or FakeBias())
    return FakeTimeStream.instances


# ---------------------------------------------------------------- the draw

hunt = make()
check("the draw happens in __init__", hunt.bias_voltages is not None and hunt.n_bias_try == 8)
check(
    "the draw stays inside the bounds",
    np.all(hunt.bias_voltages >= 0.0) and np.all(hunt.bias_voltages <= 2.0),
    f"[{hunt.bias_voltages.min():.3f}, {hunt.bias_voltages.max():.3f}]",
)
check("a seed reproduces the draw", np.allclose(make().bias_voltages, hunt.bias_voltages))
check(
    "a different seed draws differently",
    not np.allclose(make(seed=8).bias_voltages, hunt.bias_voltages),
)

explicit = make(bias_voltages=np.linspace(0.0, 2.0, 5), v_min=None, v_max=None)
check(
    "explicit voltages are used verbatim", np.allclose(explicit.bias_voltages, np.linspace(0, 2, 5))
)
check("n_bias_try follows the explicit list", explicit.n_bias_try == 5)
check("v_min/v_max report the explicit span", (explicit.v_min, explicit.v_max) == (0.0, 2.0))

# No default gate range: guessing one would put an arbitrary voltage on the device.
for missing in ({"v_min": None}, {"v_max": None}, {"v_min": None, "v_max": None}):
    try:
        make(**missing)
        check(f"missing {sorted(missing)} rejected", False, "constructed")
    except ValueError as exc:
        check(f"missing {sorted(missing)} rejected", "no default gate range" in str(exc))

for bad in ({"v_min": 2.0, "v_max": 1.0}, {"n_bias_try": 0}, {"amp": 1.5}, {"ts_duration_s": 0}):
    try:
        make(**bad)
        check(f"invalid {bad} rejected", False, "constructed")
    except ValueError:
        check(f"invalid {bad} rejected", True)

try:
    make(bias_voltages=[], v_min=None, v_max=None)
    check("an empty bias_voltages is rejected", False, "constructed")
except ValueError:
    check("an empty bias_voltages is rejected", True)

# ---------------------------------------------------------------- acquisition

hunt = make()
bias = FakeBias()
streams = run_hunt(hunt, bias)

check("one acquisition per bias", len(streams) == hunt.n_bias_try, str(len(streams)))
check(
    "every acquisition is ungated",
    all(ts.kwargs["external_trigger"] is False for ts in streams),
    str([ts.kwargs["external_trigger"] for ts in streams]),
)
check(
    "the generator is only ever set to a constant",
    all(call[0] == "constant" for call in bias.calls) and len(bias.calls) == hunt.n_bias_try,
    str(bias.calls[:2]),
)
check(
    "it holds the drawn voltages in order",
    np.allclose([call[1] for call in bias.calls], hunt.bias_voltages),
)
check(
    "the tone sits at readout_freq, at zero IF",
    all(
        ts.kwargs["lo_freq"] == 2.8e9 and np.allclose(ts.kwargs["if_freqs"], [0.0])
        for ts in streams
    ),
)

# The winner is the argmax of std(|signal|) -- checked against the known peak of the stand-in
# response, not just against the measurement's own bookkeeping.
best = int(np.argmax(hunt.parity_contrast))
check(
    "the winner is the argmax of the contrast",
    hunt.best_bias == hunt.bias_voltages[best] and hunt.best_contrast == hunt.parity_contrast[best],
)
check(
    "the winner is the bias nearest the true peak",
    best == int(np.argmin(np.abs(hunt.bias_voltages - FakeTimeStream.PEAK_V))),
    f"won at {hunt.best_bias:.3f} V, peak at {FakeTimeStream.PEAK_V} V",
)
check("best_bias_file names the winning try", hunt.best_bias_file == hunt.bias_files[best])
check("best_bias_stream is the winning stream", hunt.best_bias_stream is streams[best])
check("every try's path is kept", len(hunt.bias_files) == hunt.n_bias_try)
check(
    "the contrast really is std(|signal|)",
    np.allclose(hunt.parity_contrast, [np.std(np.abs(ts.signal[:, 0])) for ts in streams]),
)

# ---------------------------------------------------------------- bias safety

# The gate is never left energised. A bias left on an unattended device is the one failure here
# that outlives the session.
safe_bias = FakeBias()
safe_bias.output = True
run_hunt(make(), safe_bias)
check("a caller-supplied generator is de-energised on success", safe_bias.output is False)

failing_bias = FakeBias()
failing_bias.output = True
failing_bias.constant = lambda v: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    run_hunt(make(), failing_bias)
    check("an exception mid-hunt still de-energises the gate", False, "no exception raised")
except RuntimeError:
    check("an exception mid-hunt still de-energises the gate", failing_bias.output is False)

# ---------------------------------------------------------------- stale results

# A hunt that fails part-way through a RE-RUN must not leave the previous run's results beside
# this run's shorter stream list: best_bias_stream indexes one by the argmax of the other, so a
# stale parity_contrast either raises IndexError or reports a new stream under an old voltage.
rerun = make()
run_hunt(rerun, FakeBias())
first_contrast = np.array(rerun.parity_contrast)


def fail_after(bias, n):
    """Let *n* tries through, then raise."""
    seen = []

    def constant(voltage):
        seen.append(voltage)
        if len(seen) > n:
            raise RuntimeError("boom")
        bias.calls.append(("constant", voltage))

    bias.constant = constant
    return bias


try:
    run_hunt(rerun, fail_after(FakeBias(), 2))
    check("a failed re-run raises", False, "no exception raised")
except RuntimeError:
    check("a failed re-run raises", True)

check(
    "a failed re-run leaves no stale contrast curve",
    rerun.parity_contrast is None,
    f"still holds {None if rerun.parity_contrast is None else len(rerun.parity_contrast)} entries",
)
check(
    "and no stale winner",
    rerun.best_bias is None and rerun.best_bias_file is None and rerun.bias_files is None,
)
check(
    "so best_bias_stream is silent rather than wrong",
    rerun.best_bias_stream is None,
    f"streams={len(rerun.bias_streams)}, previous contrast had {len(first_contrast)}",
)

# ---------------------------------------------------------------- averaged spectrum

# A record long enough to resolve the corner: 2 s at 50 kHz gives 0.5 Hz resolution against a
# GAMMA_P/pi ~ 38 Hz corner, so there is real spectrum on both sides of it.
spec = make(ts_duration_s=2.0, n_bias_try=6)
spec_streams = run_hunt(spec)

check("no spectrum before average_psd()", spec.psd_avg is None and spec.fit_results is None)

f, psd = spec.average_psd()
n_samples = spec_streams[0].signal.shape[0]
check(
    "the frequency axis is the rfft axis of one try", f.shape == (n_samples // 2 + 1,), str(f.shape)
)
check("one spectrum per try was averaged", spec.psd_n_averaged == 6, str(spec.psd_n_averaged))
check("the tuned rate is used, not the requested one", spec.psd_fs == spec_streams[0].df)
check("the quantity is recorded", spec.psd_quantity == "abs")
check(
    "the average really is the mean of the per-try spectra",
    np.allclose(
        psd,
        np.mean(
            [
                compute_psd(np.abs(s.signal[:, 0]) - np.abs(s.signal[:, 0]).mean(), s.df)[1]
                for s in spec_streams
            ],
            axis=0,
        ),
    ),
)
check(
    "averaging is not just the first try",
    not np.allclose(
        psd,
        compute_psd(
            np.abs(spec_streams[0].signal[:, 0]) - np.abs(spec_streams[0].signal[:, 0]).mean(),
            spec_streams[0].df,
        )[1],
    ),
)

# The point of the whole exercise: does the fit return the rate that was put in?
#
# Tolerance is 30 %, which is wider than it looks. Two known effects sit between the number
# fed in and the number that comes back, both measured rather than guessed:
#
#   * the discrete-time generator's own rate is ~2 % below GAMMA_P (its autocorrelation decays
#     as (1 - 2p)^k, not exp(-2pk));
#   * fit_parity_psd reads high on short records -- ~10 % over 20 independent realisations of
#     6 x 2 s, shrinking to ~5 % at 4 M samples and to 0.4 % when handed the exact analytic
#     model. It is a log-periodogram small-sample effect, not a defect in the fitter, and it
#     is why a real measurement should not over-trust Gamma_p from a couple of seconds.
#
# The check that actually has teeth is the shape one below: nothing about a wrong Lorentzian
# reproduces the corner AND the residual.
res = spec.fit_psd()
check("the fit converges", res["success"], str(res.get("success")))
check(
    "the fit recovers the true parity rate",
    abs(res["gamma_p"] - GAMMA_P) < 0.3 * GAMMA_P,
    f"fitted {res['gamma_p']:.1f} +/- {res['gamma_p_err']:.1f} Hz vs true {GAMMA_P} Hz",
)
check(
    "and the corner that follows from it",
    np.isclose(res["f_corner"], res["gamma_p"] / np.pi, rtol=1e-6),
    f"{res['f_corner']:.2f} vs {res['gamma_p'] / np.pi:.2f} Hz",
)
check(
    "the model describes the data",
    res["resid_dex_rms"] < 0.3,
    f"resid_dex_rms = {res['resid_dex_rms']:.3f} dex",
)
check("f_bw is held at the tuned rate", res["f_bw"] == spec.psd_fs)

# Averaging fewer streams must give a noisier answer, not a different one -- and the two must
# agree, since they are the same process at different statistics.
single = spec.average_psd([spec.best_bias_stream])[1]
check("re-averaging invalidates the stale fit", spec.fit_results is None)
res_one = spec.fit_psd()
check(
    "a single try recovers the same rate, less precisely",
    abs(res_one["gamma_p"] - GAMMA_P) < 5 * res_one["gamma_p_err"]
    and res_one["gamma_p_err"] > res["gamma_p_err"],
    f"{res_one['gamma_p']:.1f} +/- {res_one['gamma_p_err']:.1f} vs "
    f"{res['gamma_p']:.1f} +/- {res['gamma_p_err']:.1f} Hz",
)

# The spectrum is of the FLUCTUATION: the operating point it sits on is not parity signal,
# and leaving it in would put the whole DC level into the lowest bins.
series = BiasHunt._parity_series(spec_streams[0], "abs")
check(
    "the mean is removed before the spectrum",
    abs(series.mean()) < 1e-12 * np.abs(spec_streams[0].signal).mean(),
    f"residual mean {series.mean():.3e}",
)
for quantity in ("real", "imag"):
    spec.average_psd(quantity=quantity)
    check(f"quantity={quantity!r} is accepted", spec.psd_quantity == quantity)
try:
    spec.average_psd(quantity="magnitude")
    check("an unknown quantity is rejected", False, "accepted")
except ValueError:
    check("an unknown quantity is rejected", True)

# Failure modes that must not silently produce a wrong spectrum.
try:
    make().average_psd()
    check("no streams raises", False, "returned a spectrum")
except RuntimeError as exc:
    check("no streams raises", "No time streams" in str(exc))

short = make(ts_duration_s=0.02)
run_hunt(short)
try:
    spec.average_psd(list(spec_streams) + list(short.bias_streams))
    check("mismatched record lengths raise", False, "averaged anyway")
except ValueError as exc:
    check("mismatched record lengths raise", "different numbers of samples" in str(exc))

# ---------------------------------------------------------------- analyze

matplotlib.pyplot.close("all")
fig = spec.analyze()
check("analyze() draws contrast and spectrum", len(fig.axes) == 2, f"{len(fig.axes)} axes")
fig = spec.analyze(psd=False)
check("analyze(psd=False) draws the contrast alone", len(fig.axes) == 1, f"{len(fig.axes)} axes")

# A loaded measurement has no streams. The contrast curve is still worth seeing, so the
# spectrum panel is skipped rather than the whole plot raising.
stripped = make()
stripped.parity_contrast = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
stripped.best_bias = float(stripped.bias_voltages[-1])
fig = stripped.analyze()
check("analyze() on a stream-less measurement degrades to one panel", len(fig.axes) == 1)
matplotlib.pyplot.close("all")

# ---------------------------------------------------------------- persistence

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "00000001-offline-test-bias_hunt.h5")
    save_for_real(hunt, save_filename=path)
    reloaded = BiasHunt.load(path)

    check("bias voltages round trip", np.allclose(reloaded.bias_voltages, hunt.bias_voltages))
    check(
        "the contrast curve round trips",
        np.allclose(reloaded.parity_contrast, hunt.parity_contrast),
    )
    check(
        "the winner round trips",
        reloaded.best_bias == hunt.best_bias
        and reloaded.best_contrast == hunt.best_contrast
        and reloaded.best_bias_file == hunt.best_bias_file,
    )
    check("the constituent paths round trip", reloaded.bias_files == hunt.bias_files)
    check("readout_freq round trips", reloaded.readout_freq == hunt.readout_freq)
    check(
        "the draw bounds round trip, not the extremes",
        (reloaded.v_min, reloaded.v_max) == (hunt.v_min, hunt.v_max)
        and (reloaded.v_min, reloaded.v_max)
        != (hunt.bias_voltages.min(), hunt.bias_voltages.max()),
        f"({reloaded.v_min}, {reloaded.v_max})",
    )
    check("the raw streams are not restored", reloaded.bias_streams == [])

# ------------------------------------------------------------------------------ summary
failed = [label for label, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label in failed:
        print("  FAILED:", label)
    sys.exit(1)
