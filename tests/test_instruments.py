"""Offline verification of ``daq.instruments`` against a simulated VISA backend.

Runs with no hardware, no VISA runtime and no ``presto`` install: a fake ``pyvisa`` module is
injected into ``sys.modules`` *before* ``daq.instruments`` is imported, so every check
exercises the real driver code against simulated instruments (including ones that reject
commands, answer slowly, or hold the wrong model).

Run from the repository root::

    python tests/test_instruments.py

Prints one PASS/FAIL line per check and exits non-zero if any check fails. Import order
matters -- the fake pyvisa must be installed before ``daq.instruments`` -- so keep this a
standalone script rather than converting it to pytest collection without care.

This suite has caught real bugs before they reached hardware: ``close()`` marking the
connection closed before ``safe_state()`` could run, ``attach()`` leaving stale keys after an
instrument mode change, a trailing space making ``SOURCE1:MODE PULS`` invalid, and PWM
``count=0`` being rejected when the instrument defines it as "infinite".
"""

import sys
import types

import numpy as np

# ---------------------------------------------------------------- fake pyvisa


class FakeResource:
    """Minimal 33220A / DC2200 simulator with a SCPI error queue."""

    def __init__(self, name, idn, state):
        self.name = name
        self.idn = idn
        self.state = state
        self.errors = []
        self.writes = []
        self.timeout = 0
        self.read_termination = None
        self.write_termination = None
        self.closed = False

    def write(self, cmd):
        self.writes.append(cmd)
        head = cmd.split(" ")[0].upper()
        arg = cmd.split(" ", 1)[1] if " " in cmd else ""
        # The 33220A rejects a DC carrier while burst is on -- the real trap.
        if head == "FUNC" and arg.upper() == "DC" and self.state.get("BURS:STAT") == "ON":
            self.errors.append('-221,"Settings conflict"')
            return
        if head == "BOGUS":
            self.errors.append('-113,"Undefined header"')
            return
        self.state[head] = arg

    def query(self, cmd):
        c = cmd.strip().upper()
        if c == "*IDN?":
            return self.idn
        if c == "SYST:ERR?":
            return self.errors.pop(0) if self.errors else '+0,"No error"'
        key = c[:-1]
        return str(self.state.get(key, "0"))

    def close(self):
        self.closed = True


FGEN_IDN = "Agilent Technologies,33220A,MY44000531,2.01"
LED_IDN = "Thorlabs,DC2200,M01271962,1.0"


class FakeRM:
    def __init__(self, resources):
        self.resources = resources
        self.opened = 0

    def list_resources(self):
        return tuple(self.resources)

    def open_resource(self, name):
        self.opened += 1
        return self.resources[name]


def install_fake_pyvisa(resources):
    rm = FakeRM(resources)
    mod = types.ModuleType("pyvisa")
    mod.ResourceManager = lambda backend="": rm
    sys.modules["pyvisa"] = mod
    return rm


def fgen_state():
    return {"OUTP": "0", "FUNC": "DC", "VOLT:OFFS": "0", "OUTP:LOAD": "INF", "BURS:STAT": "OFF"}


def led_state():
    return {"OUTPUT1:STATE": "0", "SOURCE1:MODE": "CC", "SOURCE1:CURRENT:LIMIT": "1.2"}


results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- tests

fgen_res = FakeResource("USB::33220A", FGEN_IDN, fgen_state())
led_res = FakeResource("USB0::0x1313::0x80C8::M01271962::INSTR", LED_IDN, led_state())
install_fake_pyvisa({"USB::33220A": fgen_res, "USB0::0x1313::0x80C8::M01271962::INSTR": led_res})

from daq.instruments import Agilent33220A, DC2200, InstrumentError  # noqa: E402
from daq.analysis import fold_timestream  # noqa: E402

# 1. Discovery picks the right box out of two, by IDN -- not list_resources()[0].
fg = Agilent33220A()
check("discovery selects 33220A among 2 resources", fg.resource == "USB::33220A", fg.resource)
led = DC2200()
check("discovery selects DC2200", led.resource.endswith("INSTR"), led.resource)

# 2. Ambiguity and absence both raise rather than guessing.
second = FakeResource("USB::33220A#2", FGEN_IDN, fgen_state())
install_fake_pyvisa({"USB::33220A": fgen_res, "USB::33220A#2": second})
try:
    Agilent33220A()
    check("two matches raises", False)
except InstrumentError as e:
    check("two matches raises instead of guessing", "2 instruments" in str(e))

install_fake_pyvisa({"USB::other": FakeResource("USB::other", "Keithley,2400,x,1", {})})
try:
    Agilent33220A()
    check("no match raises", False)
except InstrumentError as e:
    msg = str(e)
    check(
        "no-match error lists each resource and its IDN",
        "USB::other" in msg and "Keithley,2400,x,1" in msg,
        msg.splitlines()[1:2],
    )
    check("no-match error names the keywords it looked for", "33220A" in msg)


class DeafResource(FakeResource):
    """A resource that is visible but never answers *IDN? -- busy, or too slow."""

    def query(self, cmd):
        if cmd.strip().upper() == "*IDN?":
            raise TimeoutError("VI_ERROR_TMO: Timeout expired before operation completed")
        return super().query(cmd)


install_fake_pyvisa({"GPIB0::30::INSTR": DeafResource("GPIB0::30::INSTR", "x", {})})
try:
    Agilent33220A()
    check("silent resource raises", False)
except InstrumentError as e:
    msg = str(e)
    check(
        "error distinguishes 'no *IDN? response' from a model mismatch",
        "no *IDN? response" in msg and "VI_ERROR_TMO" in msg,
    )
    check("error points at PROBE_TIMEOUT_MS for a slow instrument", "PROBE_TIMEOUT_MS" in msg)

check(
    "probe timeout is not stingier than the connection timeout",
    Agilent33220A.PROBE_TIMEOUT_MS >= 5000,
    Agilent33220A.PROBE_TIMEOUT_MS,
)

# 2b. A realistic DAQ computer: both instruments plus unrelated devices, one of them silent.
# Irrelevant devices must be ignored, and the VID/PID hint should keep them from being probed.
probe_log = []


class LoggingResource(FakeResource):
    def query(self, cmd):
        if cmd.strip().upper() == "*IDN?":
            probe_log.append(self.name)
        return super().query(cmd)


class SilentResource(FakeResource):
    def query(self, cmd):
        if cmd.strip().upper() == "*IDN?":
            probe_log.append(self.name)
            raise TimeoutError("VI_ERROR_TMO")
        return super().query(cmd)


mixed = {
    "USB0::0x0957::0x0407::MY44000531::INSTR": LoggingResource(
        "USB0::0x0957::0x0407::MY44000531::INSTR", FGEN_IDN, fgen_state()
    ),
    "USB0::0x1313::0x80C8::M01271962::INSTR": LoggingResource(
        "USB0::0x1313::0x80C8::M01271962::INSTR", LED_IDN, led_state()
    ),
    "USB0::0x0699::0x0368::C012345::INSTR": LoggingResource(
        "USB0::0x0699::0x0368::C012345::INSTR", "TEKTRONIX,TDS2024B,C012345,CF:91.1CT", {}
    ),
    "ASRL3::INSTR": SilentResource("ASRL3::INSTR", "?", {}),
}
install_fake_pyvisa(mixed)
for cls, want in ((Agilent33220A, "0x0957::0x0407"), (DC2200, "0x1313::0x80C8")):
    probe_log.clear()
    inst = cls()
    check(
        f"{cls.__name__} finds its own device among unrelated ones",
        want in inst.resource,
        inst.resource,
    )
    check(
        f"{cls.__name__} does not probe unrelated devices",
        not any("0x0699" in r or "ASRL" in r for r in probe_log),
        str(probe_log),
    )
    inst.close()

# The hint is an optimisation only: with the unit on GPIB it must still be found.
install_fake_pyvisa(
    {
        "GPIB0::10::INSTR": FakeResource("GPIB0::10::INSTR", FGEN_IDN, fgen_state()),
        "USB0::0x0699::0x0368::C1::INSTR": FakeResource(
            "USB0::0x0699::0x0368::C1::INSTR", "TEKTRONIX,TDS2024B,C1,1.0", {}
        ),
    }
)
gpib_fgen = Agilent33220A()
check(
    "hint falls back to probing everything when nothing matches it",
    gpib_fgen.resource == "GPIB0::10::INSTR",
    gpib_fgen.resource,
)
gpib_fgen.close()

install_fake_pyvisa({"USB::33220A": fgen_res, "USB0::0x1313::0x80C8::M01271962::INSTR": led_res})

# 3. Rejected SCPI raises instead of being swallowed.
install_fake_pyvisa({"USB::33220A": fgen_res, "USB0::0x1313::0x80C8::M01271962::INSTR": led_res})
fg = Agilent33220A()
try:
    fg.write("BOGUS 1")
    check("rejected command raises", False)
except InstrumentError as e:
    check("rejected command raises InstrumentError", "Undefined header" in str(e))

# 4. The hermetic-setter bug: constant() after sawtooth() must not hit Settings conflict.
fg.sawtooth(vpp=2.0, freq_hz=500)
check(
    "sawtooth sets gated burst",
    fgen_res.state.get("BURS:STAT") == "ON",
    fgen_res.state.get("BURS:STAT"),
)
check(
    "sawtooth default offset is vpp/2",
    fgen_res.state.get("VOLT:OFFS") == "1.0",
    fgen_res.state.get("VOLT:OFFS"),
)
try:
    fg.constant(0.5)
    check("constant() after sawtooth() succeeds (burst cleared first)", True)
except InstrumentError as e:
    check("constant() after sawtooth() succeeds", False, str(e))
check("burst is off after constant()", fgen_res.state.get("BURS:STAT") == "OFF")
idx_burst = fg._instr.writes.index("BURS:STAT OFF")
idx_func = fg._instr.writes.index("FUNC DC")
check("BURS:STAT OFF is written before FUNC DC", idx_burst < idx_func, f"{idx_burst} < {idx_func}")

# 5. samples_for_periods reproduces the bench formula exactly.
fg.sawtooth(vpp=2.0, freq_hz=500)
got = fg.samples_for_periods(200, 5e4, discard_ms=25.0)
expected = int(25 * 5e4 / 1000) + int(1.0 / 500 * 5e4) * 200
check(
    "samples_for_periods matches qct_measurement formula", got == expected, f"{got} == {expected}"
)

# 6. Amplitude validation against the high-Z minimum.
try:
    fg.sawtooth(vpp=0.005, freq_hz=500)
    check("sub-minimum vpp raises", False)
except ValueError as e:
    check("sub-minimum vpp raises ValueError", "minimum" in str(e))

# 7. Context manager forces the output off, even on exception.
fgen_res.state["OUTP"] = "1"
try:
    with Agilent33220A() as ctx:
        ctx.output = True
        raise RuntimeError("boom")
except RuntimeError:
    pass
check(
    "context exit turns output off after exception",
    fgen_res.state.get("OUTP") == "OFF",
    fgen_res.state.get("OUTP"),
)

# 8. close() is idempotent and never raises.
fg2 = Agilent33220A()
fg2.close()
fg2.close()
check("close() is idempotent", True)

# 9. DC2200 validation against the queried current limit.
led = DC2200()
check("current_limit read from instrument", led.current_limit == 1.2, led.current_limit)
for bad, why in [(0.0, "zero"), (-1, "negative"), (5.0, "above limit")]:
    try:
        led.configure_pwm(bad, 100, 50, 10)
        check(f"PWM rejects {why} current", False)
    except ValueError:
        check(f"PWM rejects {why} current", True)
for kwargs, why in [
    (dict(current_a=0.1, freq_hz=0, duty_pct=50, count=10), "zero freq"),
    (dict(current_a=0.1, freq_hz=100, duty_pct=100, count=10), "duty=100"),
    (dict(current_a=0.1, freq_hz=100, duty_pct=50, count=-1), "negative count"),
    (dict(current_a=0.1, freq_hz=100, duty_pct=50, count=1.5), "fractional count"),
]:
    try:
        led.configure_pwm(**kwargs)
        check(f"PWM rejects {why}", False)
    except ValueError:
        check(f"PWM rejects {why}", True)

led.configure_pwm(0.1, 100, 50.0, 10)
check(
    "PWM uses FREQ not FREQUENCY", any(w.startswith("SOURCE1:PWM:FREQ ") for w in led._instr.writes)
)
check(
    "CC uses the CCURENT spelling",
    (led.configure_cc(0.1) or True)
    and any("SOURCE1:CCURENT:CURRENT" in w for w in led._instr.writes),
)

# 9b. TTL mode uses the documented header directly -- no probing, no CCURENT fallback.
led.configure_ttl(0.01)
check("TTL mode selected", led_res.state.get("SOURCE1:MODE") == "TTL")
check(
    "TTL current written to the documented header",
    float(led_res.state.get("SOURCE1:TTL:CURRENT")) == 0.01,
    led_res.state.get("SOURCE1:TTL:CURRENT"),
)
check(
    "TTL does not fall back to the constant-current register",
    "SOURCE1:CCURENT:CURRENT" not in [w.split(" ")[0] for w in led._instr.writes[-3:]],
)
check(
    "configure_ttl does NOT arm the LED by default",
    led_res.state.get("OUTPUT1:STATE") == "OFF",
    led_res.state.get("OUTPUT1:STATE"),
)
led.output = True
check("arming is explicit and separate", led_res.state.get("OUTPUT1:STATE") == "ON")
led.output = False
led.configure_ttl(0.01, output=True)
check("configure_ttl(output=True) arms in one step", led_res.state.get("OUTPUT1:STATE") == "ON")
led.output = False
check(
    "all four DC2200 mode setters default to not energising the LED",
    all(
        __import__("inspect").signature(getattr(DC2200, name)).parameters["output"].default is False
        for name in ("configure_cc", "configure_ttl", "configure_pwm", "configure_pulse")
    ),
)
try:
    led.configure_ttl(99.0)
    check("TTL validates against the current limit", False)
except ValueError:
    check("TTL validates against the current limit", True)


# 9c. Pulse mode: documented headers, ON/OFF times, brightness-based amplitude.
class PulseResource(FakeResource):
    """Accepts only the headers documented in the DC2200 manual v1.8 section 4.3.2."""

    ACCEPTED = {
        "SOURCE1:MODE",
        "SOURCE1:PULSE:BRIGHTNESS:LEVEL:AMPLITUDE",
        "SOURCE1:PULSE:ONTIME",
        "SOURCE1:PULSE:OFFTIME",
        "SOURCE1:PULSE:COUNT",
        "SOURCE1:TTL:CURRENT",
        "SOURCE1:CCURENT:CURRENT",
        "SOURCE1:PWM:CURRENT",
        "SOURCE1:PWM:FREQ",
        "SOURCE1:PWM:DCYCLE",
        "SOURCE1:PWM:COUNT",
        "OUTPUT1:STATE",
        "OUTPUT1:TERMINAL",
        "SOURCE1:CURRENT:LIMIT",
    }

    def write(self, cmd):
        head = cmd.split(" ")[0].upper()
        arg = cmd.split(" ", 1)[1] if " " in cmd else ""
        if head not in self.ACCEPTED:
            self.errors.append('-113,"Undefined header"')
            return
        self.writes.append(cmd)
        self.state[head] = arg


pulse_res = PulseResource("USB0::0x1313::0x80C8::PULSE::INSTR", LED_IDN, led_state())
install_fake_pyvisa({"USB0::0x1313::0x80C8::PULSE::INSTR": pulse_res})
pled = DC2200()
pulse_res.writes.clear()
# 10 us at 2 Hz, 1 mA against a 1.2 A limit, infinite count.
pled.configure_pulse(on_time_s=10e-6, freq_hz=2.0, current_a=1.0e-3, count=0)

check(
    "pulse mode selected",
    pulse_res.state.get("SOURCE1:MODE") == "PULS",
    pulse_res.state.get("SOURCE1:MODE"),
)
check(
    "pulse uses ONTime, not a width/duty command",
    float(pulse_res.state["SOURCE1:PULSE:ONTIME"]) == 10e-6,
    pulse_res.state.get("SOURCE1:PULSE:ONTIME"),
)
check(
    "OFF time derived as period - on_time",
    abs(float(pulse_res.state["SOURCE1:PULSE:OFFTIME"]) - (0.5 - 10e-6)) < 1e-12,
    pulse_res.state.get("SOURCE1:PULSE:OFFTIME"),
)
check(
    "current converted to brightness percent of the limit",
    abs(float(pulse_res.state["SOURCE1:PULSE:BRIGHTNESS:LEVEL:AMPLITUDE"]) - (1.0e-3 / 1.2 * 100))
    < 1e-9,
    pulse_res.state.get("SOURCE1:PULSE:BRIGHTNESS:LEVEL:AMPLITUDE"),
)
check("count=0 accepted as infinite", pulse_res.state.get("SOURCE1:PULSE:COUNT") == "0")
check(
    "configure_pulse does NOT arm the LED by default",
    pulse_res.state.get("OUTPUT1:STATE") == "OFF",
    pulse_res.state.get("OUTPUT1:STATE"),
)
check("no undefined-header errors were queued", not pulse_res.errors, str(pulse_res.errors))

ps = pled.settings()
check(
    "pulse settings() reports ON/OFF times and brightness",
    ps.get("pulse_on_time_s") == 10e-6
    and ps.get("pulse_count") == 0
    and abs(ps.get("pulse_freq_hz") - 2.0) < 1e-9,
    str(ps),
)
check(
    "pulse settings() converts brightness back to an absolute current",
    abs(ps.get("pulse_current_a") - 1.0e-3) < 1e-9,
    ps.get("pulse_current_a"),
)

# brightness_pct is the native form and must be accepted directly.
pled.configure_pulse(on_time_s=1e-3, off_time_s=1e-3, brightness_pct=50.0, count=10)
check(
    "brightness_pct accepted directly",
    float(pulse_res.state["SOURCE1:PULSE:BRIGHTNESS:LEVEL:AMPLITUDE"]) == 50.0,
)

for kwargs, why in [
    (dict(on_time_s=10e-6, current_a=1e-3), "no rate given"),
    (dict(on_time_s=10e-6, freq_hz=2.0, period_s=0.5, current_a=1e-3), "two rate forms"),
    (dict(on_time_s=10e-6, freq_hz=2.0), "no amplitude given"),
    (dict(on_time_s=10e-6, freq_hz=2.0, current_a=1e-3, brightness_pct=5), "two amplitudes"),
    (dict(on_time_s=0.5, freq_hz=2.0, current_a=1e-3), "on_time == period"),
    (dict(on_time_s=1.0, freq_hz=2.0, current_a=1e-3), "on_time > period"),
    (dict(on_time_s=1e-9, freq_hz=2.0, current_a=1e-3), "on_time below the 1 us floor"),
    (dict(on_time_s=20.0, off_time_s=1.0, current_a=1e-3), "on_time above the 10 s ceiling"),
    (dict(on_time_s=10e-6, off_time_s=20.0, current_a=1e-3), "off_time above the ceiling"),
    (dict(on_time_s=10e-6, freq_hz=2.0, current_a=9.9), "current over the limit"),
    (dict(on_time_s=10e-6, freq_hz=2.0, brightness_pct=150), "brightness over 100 %"),
    (dict(on_time_s=10e-6, freq_hz=2.0, current_a=1e-3, count=2000), "count over 1000"),
    (dict(on_time_s=10e-6, freq_hz=2.0, current_a=1e-3, count=2.5), "fractional count"),
]:
    try:
        pled.configure_pulse(**kwargs)
        check(f"configure_pulse rejects {why}", False)
    except ValueError:
        check(f"configure_pulse rejects {why}", True)

# PWM validation must now match the instrument's documented ranges.
for kwargs, why in [
    (dict(current_a=1e-3, freq_hz=2.0, duty_pct=0.002, count=10), "duty below the 0.1 % floor"),
    (dict(current_a=1e-3, freq_hz=2.0, duty_pct=99.95, count=10), "duty above 99.9 %"),
    (dict(current_a=1e-3, freq_hz=0.01, duty_pct=50, count=10), "freq below 0.1 Hz"),
    (dict(current_a=1e-3, freq_hz=50e3, duty_pct=50, count=10), "freq above 20 kHz"),
    (dict(current_a=1e-3, freq_hz=100, duty_pct=50, count=2000), "count over 1000"),
]:
    try:
        pled.configure_pwm(**kwargs)
        check(f"configure_pwm rejects {why}", False)
    except ValueError:
        check(f"configure_pwm rejects {why}", True)
check(
    "configure_pwm accepts count=0 as infinite",
    (pled.configure_pwm(current_a=1e-3, freq_hz=100, duty_pct=50, count=0) or True),
)

# Terminal selection and protection flags.
check(
    "terminal is settable and reads back",
    (setattr(pled, "terminal", 2) or pulse_res.state.get("OUTPUT1:TERMINAL") == "2"),
)
for bad in (0, 3):
    try:
        pled.terminal = bad
        check(f"terminal rejects {bad}", False)
    except ValueError:
        check(f"terminal rejects {bad}", True)

check(
    "pwm_train replaced pulse_train (it uses PWM mode, not pulse mode)",
    hasattr(DC2200, "pwm_train") and not hasattr(DC2200, "pulse_train"),
)

install_fake_pyvisa({"USB::33220A": fgen_res, "USB0::0x1313::0x80C8::M01271962::INSTR": led_res})

# 10. settings() reads back, and Base.attach flattens it onto a measurement.
led_res.state["SOURCE1:MODE"] = "PWM"
led_res.state["SOURCE1:PWM:CURRENT"] = "0.1"
led_res.state["SOURCE1:PWM:FREQ"] = "100"
led_res.state["SOURCE1:PWM:DCYCLE"] = "50"
led_res.state["SOURCE1:PWM:COUNT"] = "10"
s = led.settings()
check("DC2200 settings() reports PWM fields", s.get("pwm_freq_hz") == 100.0, str(s))

from daq._base import Base  # noqa: E402


class FakeMeasurement(Base):
    def __init__(self):
        self.device = "dev"


m = FakeMeasurement()
fg.sawtooth(vpp=2.0, freq_hz=500)
m.attach(bias=fg, led=led)
check(
    "attach() flattens with prefixes",
    m.bias_freq_hz == 500.0 and m.led_pwm_count == 10,
    f"bias_freq_hz={m.bias_freq_hz}, led_pwm_count={m.led_pwm_count}",
)
check("attach() drops None values", all(v is not None for v in m.__dict__.values()))
doc = m._build_document("1", "timestream", "/f.h5", "dev", None, None)
check(
    "attached settings reach the MongoDB document",
    doc.get("bias_freq_hz") == 500.0 and doc.get("led_pwm_freq_hz") == 100.0,
    f"bias_freq_hz={doc.get('bias_freq_hz')}",
)
try:
    m.attach(x=42)
    check("attach() rejects a non-instrument", False)
except TypeError:
    check("attach() rejects a non-instrument", True)

# Re-attaching after a mode change must not leave stale keys describing the old waveform.
m2 = FakeMeasurement()
fg.sawtooth(vpp=2.0, freq_hz=500)
m2.attach(bias=fg)
ramp_keys = {k for k in m2.__dict__ if k.startswith("bias_")}
check(
    "ramp attach records ramp parameters", {"bias_vpp", "bias_freq_hz", "bias_burst"} <= ramp_keys
)
fg.constant(0.5)
m2.attach(bias=fg)
dc_keys = {k for k in m2.__dict__ if k.startswith("bias_")}
check(
    "re-attach drops keys the instrument no longer reports",
    not (
        {
            "bias_vpp",
            "bias_freq_hz",
            "bias_symmetry_pct",
            "bias_burst",
            "bias_burst_mode",
            "bias_burst_phase_deg",
            "bias_gate_polarity",
        }
        & dc_keys
    ),
    str(sorted(dc_keys)),
)
check("re-attach records the new mode", m2.bias_function == "DC" and m2.bias_offset_v == 0.5)
check(
    "re-attach of an unchanged prefix does not warn about itself",
    "_attached_keys" in m2.__dict__ and "bias" in m2._attached_keys,
)
doc2 = m2._build_document("2", "timestream", "/g.h5", "dev", None, None)
check(
    "stale ramp keys do not reach the MongoDB document",
    "bias_vpp" not in doc2 and doc2.get("bias_function") == "DC",
)
check(
    "attach bookkeeping is private, so it is not saved",
    not any(k == "_attached_keys" for k in doc2),
)

# 11. fold_timestream reproduces the bench block-average.
fs, ramp_hz, n_per = 5e4, 500.0, 200
window = int(fs / ramp_hz)
period = np.arange(window) / window
rng = np.random.default_rng(0)
sig = np.tile(period, n_per) + 1j * np.tile(period[::-1], n_per)
sig = sig + rng.normal(0, 0.01, sig.shape) + 1j * rng.normal(0, 0.01, sig.shape)

t_ms, avg = fold_timestream(sig, fs, n_periods=n_per)
check("fold by n_periods gives one period", avg.shape == (2, window), str(avg.shape))
check("fold recovers the ramp", np.allclose(avg[0], period, atol=0.01))
check("time axis is one period in ms", np.isclose(t_ms[-1], (window - 1) / fs * 1e3))

t2, avg2 = fold_timestream(sig, fs, period_s=1.0 / ramp_hz)
check("fold by period_s matches fold by n_periods", np.allclose(avg, avg2))

ts_like = types.SimpleNamespace(signal=np.column_stack([sig, sig * 2]))
_, avg_t1 = fold_timestream(ts_like, fs, n_periods=n_per, tone=1)
check(
    "fold accepts a TimeStream-like object and tone index",
    np.allclose(avg_t1[0], avg[0] * 2, atol=1e-9),
)

for kwargs, why in [
    ({}, "neither period_s nor n_periods"),
    (dict(period_s=1e-3, n_periods=10), "both"),
]:
    try:
        fold_timestream(sig, fs, **kwargs)
        check(f"fold rejects {why}", False)
    except ValueError:
        check(f"fold rejects {why}", True)

try:
    fold_timestream(sig[:10], fs, period_s=1.0)
    check("fold rejects a too-short record", False)
except ValueError:
    check("fold rejects a too-short record", True)

# ---------------------------------------------------------------- transient VISA failures
#
# A DC2200 died mid-run with VI_ERROR_RSRC_NFOUND on a device that was fine minutes
# earlier, and that a probe microseconds later listed and read a clean *IDN? from -- so
# the instrument never left the bus. The blip surfaces at two points, discovery and
# open, and construction retries both. These checks pin that behaviour down, including
# the part that is easy to get wrong: releasing the VISA session when the post-open
# handshake fails. Nothing else holds a reference to a half-built instrument, so a leak
# there claims an exclusive-access USB resource until the interpreter exits, which turns
# one transient into every later open failing.

from daq.instruments import _visa as _visa_mod  # noqa: E402


class SleepRecorder:
    """Stand-in for the module's ``time``, so backoff is asserted without waiting."""

    def __init__(self):
        self.delays = []

    def sleep(self, seconds):
        self.delays.append(seconds)


class FlakyRM(FakeRM):
    """Resource manager that fails the first *n_failures* discoveries or opens.

    :param fail_stage: ``"list"`` to fail resolution (``list_resources`` returns nothing,
        as when the device is momentarily invisible), or ``"open"`` to fail the open with
        the ``VI_ERROR_RSRC_NFOUND`` seen on the bench.
    """

    def __init__(self, resources, n_failures, fail_stage):
        super().__init__(resources)
        self.n_failures = n_failures
        self.fail_stage = fail_stage
        self.list_calls = 0
        self.open_attempts = 0

    def list_resources(self):
        self.list_calls += 1
        if self.fail_stage == "list" and self.list_calls <= self.n_failures:
            return ()
        return super().list_resources()

    def open_resource(self, name):
        self.open_attempts += 1
        if self.fail_stage == "open" and self.open_attempts <= self.n_failures:
            raise OSError("VI_ERROR_RSRC_NFOUND: Insufficient location information")
        return super().open_resource(name)


def install_rm(rm):
    """Install an already-built resource manager as the fake pyvisa."""
    mod = types.ModuleType("pyvisa")
    mod.ResourceManager = lambda backend="": rm
    sys.modules["pyvisa"] = mod
    return rm


LED_NAME = "USB0::0x1313::0x80C8::M01271962::INSTR"


def fresh_led():
    return {LED_NAME: FakeResource(LED_NAME, LED_IDN, led_state())}


# Report a missing retry feature as failed checks rather than crashing partway through
# the file, so reverting the fix still yields a readable summary instead of a traceback
# that discards every result above.
_real_time = getattr(_visa_mod, "time", None)
_has_retry = _real_time is not None and hasattr(DC2200, "OPEN_RETRIES")

if not _has_retry:
    for _label in (
        "healthy open incurs no retry delay",
        "recovers from a transient open failure",
        "used exactly 3 open attempts",
        "backs off linearly between attempts",
        "recovers when the discovery probe cannot open",
        "recovers from a transient discovery failure",
        "retried resolution, not just the open",
        "error lists every attempt, not just the last",
    ):
        check(_label, False, "VisaInstrument does not retry a failed open")

try:
    if _has_retry:
        # -- a healthy instrument must incur no delay at all
        clock = SleepRecorder()
        _visa_mod.time = clock
        install_rm(FakeRM(fresh_led()))
        DC2200().close()
        check("healthy open incurs no retry delay", clock.delays == [], str(clock.delays))

        # -- open-stage transient: the bench failure, two blips then success. Constructed with
        #    an explicit resource so discovery is skipped -- autodiscovery probes candidates by
        #    opening them, which would absorb the injected faults before the real open is
        #    reached and make the attempt count mean something else.
        clock = SleepRecorder()
        _visa_mod.time = clock
        rm = install_rm(FlakyRM(fresh_led(), n_failures=2, fail_stage="open"))
        led = DC2200(resource=LED_NAME)
        check("recovers from a transient open failure", led.resource == LED_NAME, led.resource)
        check("used exactly 3 open attempts", rm.open_attempts == 3, str(rm.open_attempts))
        check(
            "backs off linearly between attempts",
            clock.delays == [DC2200.OPEN_RETRY_DELAY_S, 2 * DC2200.OPEN_RETRY_DELAY_S],
            str(clock.delays),
        )
        led.close()

        # -- the same blip during autodiscovery, where the probe rather than the real open is
        #    what cannot complete
        clock = SleepRecorder()
        _visa_mod.time = clock
        rm = install_rm(FlakyRM(fresh_led(), n_failures=2, fail_stage="open"))
        led = DC2200()
        check(
            "recovers when the discovery probe cannot open", led.resource == LED_NAME, led.resource
        )
        led.close()

        # -- discovery-stage transient: the same blip, surfacing before any open. The first
        #    version of this fix retried only the open and would still have died here.
        clock = SleepRecorder()
        _visa_mod.time = clock
        rm = install_rm(FlakyRM(fresh_led(), n_failures=2, fail_stage="list"))
        led = DC2200()
        check("recovers from a transient discovery failure", led.resource == LED_NAME, led.resource)
        check("retried resolution, not just the open", rm.list_calls == 3, str(rm.list_calls))
        led.close()

        # -- a real absence must still raise, and name every attempt rather than only the last
        clock = SleepRecorder()
        _visa_mod.time = clock
        rm = install_rm(FlakyRM(fresh_led(), n_failures=99, fail_stage="open"))
        try:
            DC2200()
            check("persistent failure still raises", False, "no exception")
        except InstrumentError as exc:
            msg = str(exc)
            check("persistent failure still raises", True)
            check(
                "error lists every attempt, not just the last",
                msg.count("attempt ") == DC2200.OPEN_RETRIES,
                f"{msg.count('attempt ')} of {DC2200.OPEN_RETRIES}",
            )
            check("error names the underlying VISA fault", "VI_ERROR_RSRC_NFOUND" in msg)

    # -- a failed handshake must release the session it already holds
    class BrokenIdn(FakeResource):
        """Opens fine, drains errors fine, then fails the *IDN? handshake."""

        def __init__(self, *args):
            super().__init__(*args)
            self.close_calls = 0

        def query(self, cmd):
            if cmd.strip().upper() == "*IDN?":
                raise OSError("VI_ERROR_TMO: Timeout expired before operation completed")
            return super().query(cmd)

        def close(self):
            self.close_calls += 1
            super().close()

    # No clock stub needed here: the handshake runs after the open succeeds, so this path
    # never retries. It is also independent of the retry feature, hence outside the guard.
    broken = BrokenIdn(LED_NAME, LED_IDN, led_state())
    install_rm(FakeRM({LED_NAME: broken}))
    try:
        # Explicit resource again: via discovery the probe's own *IDN? would fail first, so
        # construction would die in resolution and never reach the handshake under test.
        DC2200(resource=LED_NAME)
        check("a failed handshake propagates", False, "no exception")
    except Exception as exc:
        check("a failed handshake propagates", True, type(exc).__name__)
        check(
            "the original fault is not masked by cleanup",
            "VI_ERROR_TMO" in str(exc),
            str(exc)[:60],
        )
    check("a failed handshake releases the VISA session", broken.closed)
    check("the session is closed exactly once", broken.close_calls == 1, str(broken.close_calls))
finally:
    if _real_time is not None:
        _visa_mod.time = _real_time

# ---------------------------------------------------------------- summary
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label, _, detail in failed:
        print("  FAILED:", label, detail)
    sys.exit(1)
