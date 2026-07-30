"""Offline verification of ``QCTrace``'s trigger routing (#53).

``QCTrace`` gates exactly one of its four steps -- the QC trace itself -- and getting the
routing wrong is *silent*: an ungated 33220A holds its burst start level, the acquisition
succeeds, and the folded trace is flat. These checks pin the routing down without hardware by
swapping ``Sweep`` and ``TimeStream`` in the module for recorders and driving ``run()`` with a
stand-in bias generator, then asserting which per-port states each step actually asked for:

- the QC step follows the bias generator's own ``trigger_port``, so a rewired rig gates the
  right port with no change to the measurement;
- an explicit ``trigger_states`` overrides it, and is validated in ``__init__``;
- a routing that gates *nothing* raises instead of recording a static bias;
- the bias hunt and the free-running ramp stay ungated;
- the resolved states survive the HDF5 round trip, and files written before the parameter
  existed load as port 1 (what the old hardcoded ``external_trigger=True`` meant).

Requires ``presto`` to be importable (``QCTrace`` imports it transitively); no hardware and no
network. The database calls are stubbed rather than left to fail, so the round-trip check does
not sit through MongoDB's server-selection timeout on a machine with no database. Run from the
repository root::

    python tests/test_qc_trace.py
"""

import os
import sys
import tempfile

try:
    import presto  # noqa: F401
except ImportError:
    print("SKIP: presto is not installed; QCTrace cannot be imported without it")
    sys.exit(0)

import h5py
import numpy as np

import daq._base as base_mod
import daq.measurements.qc_trace as qcm
from daq.measurements.qc_trace import QCTrace

# The save path is exercised for the round trip below; keep it off the network.
base_mod.get_next_number = lambda: "00000001"
base_mod.insert_measurement = lambda document: "offline"

results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- stand-ins


class FakeSweep:
    """A locating sweep that always finds a resonance."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_results = {"fr": 2.8e9, "fr_err": 10.0}

    def run(self, **kwargs):
        return "/tmp/sweep.h5"


class FakeTimeStream:
    """Records the ``external_trigger`` each step asked for; produces plausible data."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.df = kwargs["df"]
        rng = np.random.default_rng(0)
        n = int(kwargs["pixel_counts"])
        self.signal = (rng.standard_normal((n, 1)) + 1j * rng.standard_normal((n, 1))) * 1e-3
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
        return 1000


qcm.Sweep = FakeSweep
qcm.TimeStream = FakeTimeStream


def make(**kwargs):
    params = dict(
        freq_center=2.8e9,
        amp=0.01,
        output_port=1,
        input_port=1,
        ramp_freq_hz=500.0,
        sampling_frequency=5e4,
        num_periods=10,
        n_bias_try=2,
        ts_duration_s=0.01,
        discard_start_ms=0.0,
        device="offline-test",
    )
    params.update(kwargs)
    return QCTrace(**params)


def run_with(qct, bias):
    """Drive ``run()`` past the save step and return the per-step trigger states."""
    FakeTimeStream.instances = []
    qct.save = lambda save_filename=None: "/dev/null"  # type: ignore[assignment]
    qct.run(bias=bias)
    return [ts.kwargs["external_trigger"] for ts in FakeTimeStream.instances]


def states_of(value):
    return np.asarray(value).tolist()


# ---------------------------------------------------------------- routing

# 1. The default follows the generator's own wiring, and only the QC step is gated.
qct = make()
check("routing is deferred to run()", qct.trigger_states is None)
steps = run_with(qct, FakeBias(trigger_port=1))
check("QC step gates the generator's port", states_of(steps[0]) == [1], str(states_of(steps[0])))
check(
    "bias hunt and free-running ramp stay ungated",
    all(s is False for s in steps[1:]),
    str(steps[1:]),
)
check("resolved states land on the measurement", states_of(qct.trigger_states) == [1])

# 2. Rewire the generator and the same measurement follows it -- the point of #53.
qct = make()
steps = run_with(qct, FakeBias(trigger_port=2))
check("a generator on port 2 is gated on port 2", states_of(steps[0]) == [0, 1], str(steps[0]))
check("the other steps are unaffected", all(s is False for s in steps[1:]))

# 3. An explicit routing overrides the instrument, and is checked in __init__.
qct = make(trigger_states=[0, 0, 1])
check("explicit states resolved at construction", states_of(qct.trigger_states) == [0, 0, 1])
steps = run_with(qct, FakeBias(trigger_port=1))
check("explicit states beat the instrument", states_of(steps[0]) == [0, 0, 1], str(steps[0]))

check("True still means port 1", states_of(make(trigger_states=True).trigger_states) == [1])

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
qct.save = lambda save_filename=None: "/dev/null"  # type: ignore[assignment]
try:
    qct.run(bias=FakeBias(trigger_port=None))
    check("undeclared trigger_port raises", False, "ran anyway")
except ValueError as exc:
    check("undeclared trigger_port raises", "trigger_port is None" in str(exc), str(exc)[:60])
check("nothing was acquired before the routing failed", not FakeTimeStream.instances)

# ---------------------------------------------------------------- persistence

qct = make(trigger_states=[0, 1])
qct.fr = 2.8e9
qct.fr_err = 10.0
qct.time_ms = np.linspace(0, 2, 100)
qct.avg_iq = np.zeros((2, 100))
qct.parity_contrast = np.array([1.0, 2.0])
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "00000001-offline-test-qc_trace.h5")
    qct.save(save_filename=path)
    reloaded = QCTrace.load(path)
    check(
        "trigger states survive the HDF5 round trip",
        states_of(reloaded.trigger_states) == [0, 1],
        str(reloaded.trigger_states),
    )

    # Files written before the parameter existed carry no dataset; they were all port 1.
    legacy = os.path.join(tmp, "00000002-offline-test-qc_trace.h5")
    with h5py.File(path, "r") as src, h5py.File(legacy, "w") as dst:
        for key in src.attrs:
            dst.attrs[key] = src.attrs[key]
        for key in src:
            if key != "trigger_states":
                src.copy(key, dst)
    check(
        "a pre-#53 file loads as port 1",
        states_of(QCTrace.load(legacy).trigger_states) == [1],
    )

# ------------------------------------------------------------------------------ summary
failed = [label for label, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label in failed:
        print("  FAILED:", label)
    sys.exit(1)
