"""Offline verification of ``QCTrace``: trigger routing (#53) and the folding step.

``QCTrace`` is one gated acquisition, and getting the gate routing wrong is *silent*: an
ungated 33220A holds its burst start level, the acquisition succeeds, and the folded trace is
flat. These checks pin the routing down without hardware by swapping ``TimeStream`` in the
shared readout module for a recorder and driving ``run()`` with a stand-in bias generator, then
asserting which per-port states the acquisition actually asked for:

- the acquisition follows the bias generator's own ``trigger_port``, so a rewired rig gates the
  right port with no change to the measurement;
- an explicit ``trigger_states`` overrides it, and is validated in ``__init__``;
- a routing that gates *nothing* raises instead of recording a static bias;
- the resolved states survive the HDF5 round trip.

Plus the fold: ``run()`` folds on the ramp's own period at the *tuned* sample rate, and
``fold()`` is re-callable on a stream loaded back from ``qc_file``.

Note the ``TimeStream`` swap targets ``daq.measurements._gate_bias``, not this measurement's own
module: the readout builder both ``QCTrace`` and ``BiasHunt`` acquire through lives there.

Requires ``presto`` to be importable (``QCTrace`` imports it transitively); no hardware and no
network. The database calls are stubbed rather than left to fail, so the round-trip check does
not sit through MongoDB's server-selection timeout on a machine with no database. Run from the
repository root::

    python tests/test_qc_trace.py
"""

import os
import sys
import tempfile
import warnings

try:
    import presto  # noqa: F401
except ImportError:
    print("SKIP: presto is not installed; QCTrace cannot be imported without it")
    sys.exit(0)

import numpy as np

import daq._base as base_mod
import daq.measurements._gate_bias as gate_bias_mod
from daq.measurements.qc_trace import QCTrace

# The save path is exercised for the round trip below; keep it off the network.
base_mod.get_next_number = lambda: "00000001"
base_mod.insert_measurement = lambda document: "offline"

results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- stand-ins

RAMP_HZ = 500.0
FS = 5e4
N_PERIODS = 10
#: Noise on the synthetic response, well under the per-sample step of the ramp buried in it
#: (1 / 100), so a correct fold leaves that ramp monotonic and an incorrect one does not.
NOISE = 1e-3


class FakeTimeStream:
    """Records the ``external_trigger`` the step asked for; produces plausible data."""

    instances = []
    #: Sample rate the "hardware" tunes to, which run() must fold on rather than the
    #: requested one -- a window off by a sample smears the average across blocks.
    tuned_df = FS

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.df = FakeTimeStream.tuned_df
        n = int(kwargs["pixel_counts"])
        # A ramp-shaped response repeating once per period, plus noise: folding must recover
        # the ramp. Built on the tuned rate, as a real acquisition would be.
        per = int(round(self.df / RAMP_HZ))
        rng = np.random.default_rng(0)
        ramp = np.tile(np.arange(per) / per, n // per + 1)[:n]
        noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        self.signal = (ramp + 1j * ramp + NOISE * noise).reshape(n, 1)
        FakeTimeStream.instances.append(self)

    def attach(self, **instruments):
        pass

    def run(self, **kwargs):
        return f"/tmp/ts{len(FakeTimeStream.instances)}.h5"


class FakeBias:
    """A 33220A stand-in: knows which port it is wired to, records what it was told."""

    def __init__(self, trigger_port=1):
        self.trigger_port = trigger_port
        self.output = False
        self.calls = []

    def sawtooth(self, **kwargs):
        self.calls.append(("sawtooth", kwargs.get("gated")))

    def constant(self, voltage):
        self.calls.append(("constant", voltage))

    def samples_for_periods(self, n_periods, sample_rate, *, freq_hz=None, discard_ms=25.0):
        return int(round(sample_rate / freq_hz)) * int(n_periods)


gate_bias_mod.TimeStream = FakeTimeStream

# Stub the save at *class* level, not per instance: an instance attribute would itself land in
# the saved record, since Base._save walks __dict__. The real one is kept for the round trip.
save_for_real = QCTrace.save
QCTrace.save = lambda self, save_filename=None: "/dev/null"  # type: ignore[assignment]


def make(**kwargs):
    params = dict(
        readout_freq=2.8e9,
        amp=0.01,
        output_port=1,
        input_port=1,
        ramp_freq_hz=RAMP_HZ,
        sampling_frequency=FS,
        num_periods=N_PERIODS,
        discard_start_ms=0.0,
        device="offline-test",
    )
    params.update(kwargs)
    return QCTrace(**params)


def run_with(qct, bias):
    """Drive ``run()`` past the save step and return the per-step trigger states."""
    FakeTimeStream.instances = []
    qct.run(bias=bias)
    return [ts.kwargs["external_trigger"] for ts in FakeTimeStream.instances]


def states_of(value):
    return np.asarray(value).tolist()


# ---------------------------------------------------------------- routing

# 1. The default follows the generator's own wiring, and the acquisition is the only step.
qct = make()
check("routing is deferred to run()", qct.trigger_states is None)
steps = run_with(qct, FakeBias(trigger_port=1))
check("exactly one acquisition", len(steps) == 1, str(len(steps)))
check("it gates the generator's port", states_of(steps[0]) == [1], str(states_of(steps[0])))
check("resolved states land on the measurement", states_of(qct.trigger_states) == [1])

# 2. Rewire the generator and the same measurement follows it -- the point of #53.
qct = make()
steps = run_with(qct, FakeBias(trigger_port=2))
check("a generator on port 2 is gated on port 2", states_of(steps[0]) == [0, 1], str(steps[0]))

# 3. An explicit routing overrides the instrument, and is checked in __init__.
qct = make(trigger_states=[0, 0, 1])
check("explicit states resolved at construction", states_of(qct.trigger_states) == [0, 0, 1])
steps = run_with(qct, FakeBias(trigger_port=1))
check("explicit states beat the instrument", states_of(steps[0]) == [0, 0, 1], str(steps[0]))

check("True still means port 1", states_of(make(trigger_states=True).trigger_states) == [1])

# 3b. The default re-reads the generator on EVERY run. Caching the first run's answer would
# mean that rewiring the rig and re-running the same object silently gates the old port --
# the failure this class exists to prevent, reintroduced through the back door.
qct = make()
first = run_with(qct, FakeBias(trigger_port=1))
second = run_with(qct, FakeBias(trigger_port=2))
check(
    "a second run follows a rewired generator",
    states_of(first[0]) == [1] and states_of(second[0]) == [0, 1],
    f"{states_of(first[0])} then {states_of(second[0])}",
)

qct = make(trigger_states=[0, 1])
again = run_with(qct, FakeBias(trigger_port=2))
check("an explicit routing survives repeated runs", states_of(again[0]) == [0, 1])

# 3c. An explicit routing that leaves the generator's own port unasserted still gates
# *something*, so it passes the no-port check -- but the ramp waits on a port nothing
# asserts. Warn rather than raise: the generator's declared port may itself be the wrong one.
qct = make(trigger_states=[0, 1])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    run_with(qct, FakeBias(trigger_port=1))
check(
    "a routing that skips the generator's port warns",
    any("trigger_port=1" in str(w.message) for w in caught),
    str([str(w.message)[:60] for w in caught]),
)
with warnings.catch_warnings(record=True) as quiet:
    warnings.simplefilter("always")
    run_with(make(), FakeBias(trigger_port=2))
check("a correct routing does not warn", not quiet, str([str(w.message)[:40] for w in quiet]))

# 4. A routing that gates nothing must raise -- it would record a static bias.
for bad in (False, [0], [0, 0]):
    try:
        make(trigger_states=bad)
        check(f"trigger_states={bad!r} rejected", False, "constructed")
    except ValueError as exc:
        check(f"trigger_states={bad!r} rejected", "gates no digital output port" in str(exc))

for bad in ([3], [1, 1, 1, 1, 1], [0.5]):
    try:
        make(trigger_states=bad)
        check(f"invalid states {bad!r} rejected", False, "constructed")
    except ValueError:
        check(f"invalid states {bad!r} rejected", True)

# 5. A generator that does not say where it is wired fails before any acquisition.
qct = make()
FakeTimeStream.instances = []
try:
    qct.run(bias=FakeBias(trigger_port=None))
    check("undeclared trigger_port raises", False, "ran anyway")
except ValueError as exc:
    check("undeclared trigger_port raises", "trigger_port is None" in str(exc), str(exc)[:60])
check("nothing was acquired before the routing failed", not FakeTimeStream.instances)

# 6. The ramp is gated, since a free-running one would not line up with the blocks.
bias = FakeBias(trigger_port=1)
run_with(make(), bias)
check("the sawtooth is gated", bias.calls == [("sawtooth", True)], str(bias.calls))

# ---------------------------------------------------------------- folding

# run() folds on the ramp's own period at the TUNED sample rate. Tune df away from what was
# asked for: dividing the record by num_periods would then land a sample short of the physical
# period and smear the average across blocks.
FakeTimeStream.tuned_df = FS + 100.0
qct = make()
run_with(qct, FakeBias(trigger_port=1))
FakeTimeStream.tuned_df = FS
expected = int(round(qct.qc_stream.df / RAMP_HZ))
check(
    "run() folds one period at the tuned rate",
    qct.avg_iq is not None and qct.avg_iq.shape == (2, expected),
    f"{None if qct.avg_iq is None else qct.avg_iq.shape} vs (2, {expected})",
)
check(
    "the folded trace recovers the ramp",
    # The injected response is a rising ramp per period, in both quadratures; averaging must
    # leave it monotonic despite the noise it was buried in.
    np.all(np.diff(qct.avg_iq[0]) > 0)
    and np.allclose(qct.avg_iq[0], qct.avg_iq[1], atol=10 * NOISE),
    f"min slope {np.min(np.diff(qct.avg_iq[0])):.2e}",
)
check(
    "time_ms spans one period",
    np.isclose(qct.time_ms[-1] + qct.time_ms[1], 1e3 / RAMP_HZ, rtol=1e-2),
    f"{qct.time_ms[-1]:.4f} ms vs {1e3 / RAMP_HZ:.4f} ms",
)

# fold() is re-callable, on this measurement's stream or on one loaded back from qc_file.
qct.fold(n_periods=N_PERIODS)
check(
    "fold(n_periods=) re-folds in place",
    qct.avg_iq.shape[1] == qct.qc_stream.signal.shape[0] // N_PERIODS,
)
qct.fold()
check("fold() returns to the ramp period", qct.avg_iq.shape == (2, expected))
t_ms, avg = qct.fold(qct.qc_stream)
check("fold() accepts an explicit stream", avg.shape == (2, expected) and t_ms.shape[0] == expected)

for kwargs in ({"period_s": 1e-3, "n_periods": 5},):
    try:
        qct.fold(**kwargs)
        check(f"fold({kwargs}) rejected", False, "accepted")
    except ValueError:
        check(f"fold({kwargs}) rejected", True)

fresh = make()
try:
    fresh.fold()
    check("fold() before run() raises", False, "folded")
except RuntimeError as exc:
    check("fold() before run() raises", "No time stream to fold" in str(exc))

# ---------------------------------------------------------------- persistence

qct = make(trigger_states=[0, 1])
run_with(qct, FakeBias(trigger_port=2))
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "00000001-offline-test-qc_trace.h5")
    save_for_real(qct, save_filename=path)
    reloaded = QCTrace.load(path)
    check(
        "trigger states survive the HDF5 round trip",
        states_of(reloaded.trigger_states) == [0, 1],
        str(reloaded.trigger_states),
    )
    check(
        "the folded trace survives the round trip",
        np.allclose(reloaded.avg_iq, qct.avg_iq) and np.allclose(reloaded.time_ms, qct.time_ms),
    )
    check("readout_freq round trips", reloaded.readout_freq == qct.readout_freq)
    check("the raw acquisition's path round trips", reloaded.qc_file == qct.qc_file)
    check("the raw stream is not restored", reloaded.qc_stream is None)

    # The stored routing describes the run that made the file, not the bench in front of you:
    # re-running a loaded measurement must read the generator it is handed.
    steps = run_with(reloaded, FakeBias(trigger_port=1))
    check(
        "re-running a loaded measurement re-reads the generator",
        states_of(steps[0]) == [1],
        str(states_of(steps[0])),
    )

# ------------------------------------------------------------------------------ summary
failed = [label for label, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label in failed:
        print("  FAILED:", label)
    sys.exit(1)
