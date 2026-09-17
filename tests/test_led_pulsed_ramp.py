"""Offline verification of ``LEDPulsedRamp``: LED sequencing, trigger routing and the record.

The measurement composes the shared gated ramp with a DC2200 pulse train that is started by a
software write from inside ``TimeStream.run``'s ``on_acquire`` hook. Two things about it are
silent when wrong, and each gets a check here:

- the LED must be enabled *inside* the hook (after the Presto is configured, before the
  pixels are requested), and disabled on every exit path -- an LED left on an unattended
  device outlives the session;
- the routing must gate the bias generator's port and **never** the LED's: in pulse mode the
  DC2200's modulation SMA is an output, so asserting its port drives an output into an output.

Plus the rest of the contract: a dark file (``led_current_a=0``) runs the identical
acquisition with the LED never enabled; the current limit and protection flags are checked
before the ramp is armed; the multitone stream is planned from the tone list with the storage
settings forwarded; the record round-trips through HDF5.

The ``TimeStream`` swap targets ``daq.measurements._gate_bias`` (the shared readout builder),
as the other gate-bias suites do. Requires ``presto`` to be importable; no hardware, no
network. Run from the repository root::

    python tests/test_led_pulsed_ramp.py
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

try:
    import presto  # noqa: F401
except ImportError:
    print("SKIP: presto is not installed; LEDPulsedRamp cannot be imported without it")
    sys.exit(0)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import daq._base as base_mod  # noqa: E402
import daq.measurements._gate_bias as gate_bias_mod  # noqa: E402
from daq.measurements.led_pulsed_ramp import LEDPulsedRamp  # noqa: E402

base_mod.get_next_number = lambda: "00000001"
base_mod.insert_measurement = lambda document: "offline"

results = []


def check(label, condition, detail=""):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- stand-ins

FS = 1e5
RAMP_HZ = 5e3
EVENTS = []  # the global order of instrument writes and stream milestones


class FakeTimeStream:
    """Records its construction, attaches, and calls on_acquire like the real run() does."""

    instances = []
    tuned_df = FS
    fail_in_run = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.df = FakeTimeStream.tuned_df
        self.attached = {}
        n = int(kwargs["pixel_counts"])
        n_tones = np.atleast_1d(kwargs["if_freqs"]).size
        rng = np.random.default_rng(0)
        self.signal = (
            rng.standard_normal((n, n_tones)) + 1j * rng.standard_normal((n, n_tones))
        ) * 1e-3
        FakeTimeStream.instances.append(self)

    def attach(self, **instruments):
        for prefix, inst in instruments.items():
            self.attached[prefix] = (
                dict(inst.settings()) if hasattr(inst, "settings") else dict(inst)
            )

    def run(self, **kwargs):
        EVENTS.append("stream:configured")
        hook = kwargs.get("on_acquire")
        if hook is not None:
            hook()
        EVENTS.append("stream:get_pixels")
        if FakeTimeStream.fail_in_run:
            raise RuntimeError("simulated acquisition failure")
        EVENTS.append("stream:saved")
        return f"/tmp/led_ts{len(FakeTimeStream.instances)}.h5"


class FakeBias:
    def __init__(self, trigger_port=1):
        self.trigger_port = trigger_port
        self._output = False
        self.calls = []

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = bool(value)
        EVENTS.append(f"bias:output={'on' if value else 'off'}")

    def sawtooth(self, **kwargs):
        self.calls.append(("sawtooth", kwargs))
        EVENTS.append("bias:sawtooth")

    def settings(self):
        return {"function": "RAMP", "freq_hz": RAMP_HZ, "trigger_port": self.trigger_port}


class FakeLED:
    def __init__(self, trigger_port=2, limit=0.2, terminal=2, tripped=None):
        self.trigger_port = trigger_port
        self.current_limit = limit
        self._terminal = terminal
        self._output = False
        self.tripped = tripped or {}
        self.pulse = None
        self.mode = "CC"

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = bool(value)
        EVENTS.append(f"led:output={'on' if value else 'off'}")

    @property
    def terminal(self):
        return self._terminal

    @terminal.setter
    def terminal(self, value):
        self._terminal = int(value)
        EVENTS.append(f"led:terminal={value}")

    def protection_status(self):
        base = {
            "current_limit": False,
            "interlock": False,
            "driver_over_temp": False,
            "head_over_temp": False,
        }
        base.update(self.tripped)
        return base

    def configure_pulse(self, **kwargs):
        if kwargs.get("current_a") is not None and not 0 < kwargs["current_a"] < self.current_limit:
            raise ValueError("current out of range")
        self.pulse = kwargs
        self.mode = "PULS"
        EVENTS.append(f"led:configure_pulse(output={kwargs.get('output')})")
        self.output = kwargs.get("output", False)

    def settings(self):
        state = {
            "mode": self.mode,
            "current_limit_a": self.current_limit,
            "output": self.output,
            "trigger_port": self.trigger_port,
        }
        if self.pulse:
            state["pulse_on_time_s"] = self.pulse["on_time_s"]
            state["pulse_off_time_s"] = self.pulse["off_time_s"]
        return state


gate_bias_mod.TimeStream = FakeTimeStream
save_for_real = LEDPulsedRamp.save
LEDPulsedRamp.save = lambda self, save_filename=None: "/dev/null"  # type: ignore[assignment]

FREQS = [2.719e9, 2.786e9, 2.848e9]
LABELS = ["dev0", "dev1", "dev9 (KID)"]


def make(**kwargs):
    params = dict(
        readout_freqs=FREQS,
        amp=[0.01, 0.02, 0.2],
        output_port=1,
        input_port=1,
        duration_s=0.02,
        led_on_s=100e-6,
        led_period_s=5e-3,  # 25 gate periods
        led_current_a=0.099,
        expected_led_limit_a=0.2,
        led_terminal=2,
        ramp_vpp=0.4,
        ramp_freq_hz=RAMP_HZ,
        sampling_frequency=FS,
        tone_labels=LABELS,
        device="offline-test",
    )
    params.update(kwargs)
    return LEDPulsedRamp(**params)


def run(meas, bias=None, led=None):
    EVENTS.clear()
    FakeTimeStream.instances = []
    bias = bias or FakeBias()
    led = led or FakeLED()
    meas.run(bias=bias, led=led)
    return bias, led


# ---------------------------------------------------------------- 1. the LED sequence

m = make()
bias, led = run(m)
stream = FakeTimeStream.instances[0]
check("exactly one acquisition", len(FakeTimeStream.instances) == 1)
i_cfg = EVENTS.index("led:configure_pulse(output=False)")
i_ramp = EVENTS.index("bias:sawtooth")
i_conf = EVENTS.index("stream:configured")
i_on = EVENTS.index("led:output=on")
i_pix = EVENTS.index("stream:get_pixels")
check(
    "pulse train configured with the output off, before the ramp is armed",
    i_cfg < i_ramp,
    str(EVENTS),
)
check(
    "LED enabled inside on_acquire: after configure, before get_pixels",
    i_conf < i_on < i_pix,
    str(EVENTS),
)
check("LED output is off after the run", led.output is False)
check("gate output is off after the run", bias.output is False)
check(
    "the pulse train is what was asked for",
    led.pulse["on_time_s"] == 100e-6
    and abs(led.pulse["off_time_s"] - 4.9e-3) < 1e-12
    and led.pulse["current_a"] == 0.099
    and led.pulse["count"] == 0,
    str(led.pulse),
)
check("the ramp is gated", bias.calls[0][1]["gated"] is True and bias.calls[0][1]["vpp"] == 0.4)
check(
    "host stamps bracket the enabling write",
    "host" in stream.attached
    and stream.attached["host"]["led_enable_after_unix"]
    >= stream.attached["host"]["led_enable_before_unix"]
    >= stream.attached["host"]["acquire_hook_unix"],
)
check(
    "LED settings and plan are attached to the stream",
    stream.attached.get("led", {}).get("mode") == "PULS"
    and stream.attached.get("led_plan", {}).get("period_s") == 5e-3
    and stream.attached["led_plan"]["scheme"].startswith("DC2200 internal pulse engine"),
)
check(
    "the record carries the tuned rate, sample count and path",
    m.df_tuned == FS and m.n_samples == stream.signal.shape[0] and m.raw_file.endswith(".h5"),
)
check("expected flash count", abs(m.led_flashes_expected - 4.0) < 1e-9, str(m.led_flashes_expected))
check(
    "the terminal was selected and read back",
    m.led_terminal_read == 2 and "led:terminal=2" in EVENTS,
)

# LED off on the exception path too.
FakeTimeStream.fail_in_run = True
try:
    run(make())
    check("an acquisition failure propagates", False)
except RuntimeError:
    check("an acquisition failure propagates", True)
finally:
    FakeTimeStream.fail_in_run = False
check(
    "LED output is off after a failed run",
    EVENTS[-1] in ("led:output=off", "bias:output=off")
    and "led:output=off" in EVENTS[EVENTS.index("led:output=on") :],
    str(EVENTS[-4:]),
)

# ---------------------------------------------------------------- 2. dark files

m = make(led_current_a=0.0)
check("zero current means dark", m.dark is True)
bias, led = run(m)
check(
    "a dark file never configures or enables the LED",
    led.pulse is None and "led:output=on" not in EVENTS,
    str(EVENTS),
)
check("...but still runs the gated ramp", bias.calls and bias.calls[0][1]["gated"] is True)
check("dark plan is recorded", FakeTimeStream.instances[0].attached["led_plan"]["dark"] is True)
check("dark expects no flashes", m.led_flashes_expected == 0.0)

# ---------------------------------------------------------------- 3. routing

m = make()
bias, led = run(m, FakeBias(trigger_port=1), FakeLED(trigger_port=2))
states = np.asarray(FakeTimeStream.instances[0].kwargs["external_trigger"]).tolist()
check("routing gates the generator's port only", states == [1], str(states))

# The LED's port must never be asserted -- explicitly or via a generator that claims it.
for bad_states, bias_port in (([1, 1], 1), ([0, 1], 2)):
    m = make(trigger_states=bad_states)
    EVENTS.clear()
    FakeTimeStream.instances = []
    try:
        m.run(bias=FakeBias(trigger_port=bias_port), led=FakeLED(trigger_port=2))
        check(f"routing {bad_states} asserting the LED port is refused", False, "ran")
    except ValueError as exc:
        check(f"routing {bad_states} asserting the LED port is refused", "OUTPUT" in str(exc))
    check(
        "...before any acquisition or LED enable",
        not FakeTimeStream.instances and "led:output=on" not in EVENTS,
    )

for bad in (False, [0]):
    try:
        make(trigger_states=bad)
        check(f"trigger_states={bad!r} (gates nothing) is refused", False)
    except ValueError:
        check(f"trigger_states={bad!r} (gates nothing) is refused", True)

# ---------------------------------------------------------------- 4. LED checks before arming

for label, led_kwargs, exc_type, needle in (
    ("wrong current limit", dict(limit=0.1), RuntimeError, "current limit"),
    ("tripped protection", dict(tripped={"interlock": True}), RuntimeError, "protection"),
):
    EVENTS.clear()
    FakeTimeStream.instances = []
    try:
        make().run(bias=FakeBias(), led=FakeLED(**led_kwargs))
        check(f"{label} aborts", False, "ran")
    except exc_type as exc:
        check(f"{label} aborts", needle in str(exc), str(exc)[:60])
    check(
        f"{label}: nothing acquired, ramp not armed",
        not FakeTimeStream.instances and "bias:sawtooth" not in EVENTS,
    )

try:
    make(led_current_a=0.25)
    check("a current at or above the expected limit is refused in __init__", False)
except ValueError:
    check("a current at or above the expected limit is refused in __init__", True)

# ---------------------------------------------------------------- 5. the stream plan

m = make(save_arrays="pixels", save_dtype="complex128")
run(m)
kw = FakeTimeStream.instances[0].kwargs
lo = (min(FREQS) + max(FREQS)) / 2
check("LO defaults to the midpoint", m.lo_freq == lo and kw["lo_freq"] == lo)
check(
    "IFs are |f - LO| with the sideband from the sign",
    np.allclose(kw["if_freqs"], np.abs(np.array(FREQS) - lo))
    and np.asarray(kw["is_usb"]).tolist() == [False, True, True],
)
check("per-tone amplitudes forwarded", np.asarray(kw["amp"]).tolist() == [0.01, 0.02, 0.2])
check(
    "storage settings forwarded", kw["save_arrays"] == "pixels" and kw["save_dtype"] == "complex128"
)
m_default = make()
run(m_default)
check(
    "storage defaults are resolved and forwarded explicitly (the record must store them)",
    FakeTimeStream.instances[0].kwargs["save_arrays"] == "signal"
    and FakeTimeStream.instances[0].kwargs["save_dtype"] == "complex64"
    and (m_default.save_arrays, m_default.save_dtype) == ("signal", "complex64"),
)
check("sample count covers the duration", kw["pixel_counts"] == int(round(0.02 * FS)))
check("discard defaults to zero", kw["discard_start_ms"] == 0.0)

single = make(readout_freqs=2.8e9, amp=0.05, tone_labels=["only"])
run(single)
kw = FakeTimeStream.instances[0].kwargs
check(
    "a single tone sits at zero IF on its own LO",
    kw["lo_freq"] == 2.8e9 and np.allclose(kw["if_freqs"], [0.0]),
)

# ---------------------------------------------------------------- 6. warnings and validation

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    make(led_period_s=5.05e-3)  # 25.25 gate periods
check(
    "a flash period that is not whole gate periods warns",
    any("gate periods" in str(w.message) for w in caught),
)
with warnings.catch_warnings(record=True) as quiet:
    warnings.simplefilter("always")
    make()
check("a commensurate period does not warn", not quiet, str([str(w.message)[:50] for w in quiet]))

for bad in (
    dict(led_period_s=50e-6),
    dict(led_on_s=0),
    dict(duration_s=0),
    dict(led_current_a=-1),
    dict(amp=[0.5, 0.5, 0.5]),
    dict(readout_freqs=[2.7e9, 2.7e9, 2.8e9]),
    dict(tone_labels=["a", "b"]),
    dict(led_terminal=3),
    dict(lo_freq=1e9),
):
    try:
        make(**bad)
        check(f"{bad} is refused", False)
    except ValueError:
        check(f"{bad} is refused", True)

# ---------------------------------------------------------------- 7. round trip

LEDPulsedRamp.save = save_for_real  # type: ignore[assignment]
m = make()
with tempfile.TemporaryDirectory() as tmp:
    run(m)
    path = os.path.join(tmp, "led.h5")
    m.save(save_filename=path)
    loaded = LEDPulsedRamp.load(path)
check(
    "round trip: tones, LO and labels",
    np.array_equal(loaded.readout_freqs, m.readout_freqs)
    and loaded.lo_freq == m.lo_freq
    and loaded.tone_labels == LABELS,
)
check(
    "round trip: LED plan",
    (loaded.led_on_s, loaded.led_period_s, loaded.led_current_a, loaded.dark)
    == (m.led_on_s, m.led_period_s, m.led_current_a, m.dark),
)
check(
    "round trip: routing and raw path",
    np.asarray(loaded.trigger_states).tolist() == [1] and loaded.raw_file == m.raw_file,
)
check(
    "round trip: tuned rate and sample count",
    loaded.df_tuned == m.df_tuned and loaded.n_samples == m.n_samples,
)
check(
    "round trip: attached LED readback",
    getattr(loaded, "led_mode", None) == "PULS"
    and getattr(loaded, "led_pulse_on_time_s", None) == 100e-6,
)
check("the stream is not restored", loaded.led_stream is None)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
