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

import numpy as np

import daq._base as base_mod
import daq.measurements._gate_bias as gate_bias_mod
from daq.measurements.bias_hunt import BiasHunt

base_mod.get_next_number = lambda: "00000001"
base_mod.insert_measurement = lambda document: "offline"

results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- stand-ins

FS = 5e4


class FakeTimeStream:
    """Records what each try asked for, with a contrast set by the bias it was taken at.

    The magnitude spread peaks at ``PEAK_V``, so the hunt has an unambiguous winner and the
    ranking can be checked against a known answer rather than against itself.
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
        rng = np.random.default_rng(abs(hash(kwargs["notes"])) % 2**32)
        self.signal = ((1.0 + spread * rng.standard_normal(n)) + 0j).reshape(n, 1)
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
        ts_duration_s=0.01,
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
