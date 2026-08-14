# Instruments

Drivers for the non-Presto benchtop hardware in the lab, reached over VISA (SCPI).

These are **instruments, not measurements**: they produce no data arrays and do not subclass
`Base`. You compose them with the ordinary measurement classes in a notebook, and record
their state with `Base.attach()`.

| Class | Hardware | Purpose |
|---|---|---|
| `Agilent33220A` | Agilent/Keysight 33220A | Gate-bias source: constant DC or gated sawtooth ramp |
| `DC2200` | Thorlabs DC2200 | LED driver: constant current, PWM or explicit-width pulse train, or TTL-gated |

## Installation

`pyvisa` is an optional dependency and is imported lazily, so `import daq` works on analysis
machines with no VISA runtime. On the acquisition machine:

```bash
pip install daq[instruments]
```

You also need a VISA implementation:

| Platform | Backend |
|---|---|
| Windows (lab standard) | [NI-VISA](https://www.ni.com/en/support/downloads/drivers/download.ni-visa.html). Reboot if asked. Confirm the instrument appears in Device Manager or NI MAX. |
| Anything else / no NI-VISA | `pip install pyvisa-py pyusb libusb-package`, then set `DAQ_VISA_BACKEND=@py` |

Do **not** run from WSL — USB/VISA will not see the instruments reliably. Close any other
session holding the instrument (NI MAX, Thorlabs software, another notebook kernel) first.

## Configuration

| Variable | Meaning |
|---|---|
| `DAQ_FGEN_RESOURCE` | VISA resource for the function generator. Unset → autodiscover. |
| `DAQ_LED_RESOURCE` | VISA resource for the LED driver. Unset → autodiscover. |
| `DAQ_FGEN_TRIGGER_PORT` | Presto digital output port wired to the generator's gate input. Unset → 1. |
| `DAQ_LED_TRIGGER_PORT` | Presto digital output port wired to the LED's modulation input. Unset → 2. |
| `DAQ_VISA_BACKEND` | `""` for NI-VISA (default), `"@py"` for pyvisa-py. |

**You normally do not need to set these.** With the function generator, the LED driver and
any number of unrelated instruments on the same computer, `Agilent33220A()` and `DC2200()`
find their own hardware unaided:

```python
with Agilent33220A() as bias, DC2200() as led:   # no addresses anywhere
    ...
```

The rule is **exactly one match for that model**, not one device in total. Each driver lists
the visible VISA resources, asks each `*IDN?`, and keeps the one whose reply identifies it —
`33220A` for the generator, `DC2200` for the LED. A scope, a multimeter or a USB-serial gadget
answers with something else and is ignored.

It deliberately **raises** when zero or several match, rather than picking the first —
grabbing `list_resources()[0]` silently binds to the wrong instrument as soon as a second one
is plugged in, and then produces plausible-looking data from it. So you are asked to choose
only when the situation is genuinely ambiguous, e.g. two 33220As on one bench.

Each driver also carries the USB vendor/product ID of its model, and probes only matching
resources when any are present, so unrelated devices are not opened at all. That keeps
discovery quick: a device that never answers `*IDN?` would otherwise cost `PROBE_TIMEOUT_MS`
(5 s) on every construction. The hint is only an optimisation — if nothing matches it, because
the instrument is on GPIB for instance, discovery falls back to probing everything.

**When to set the variables anyway:** two units of the same model; a model whose `*IDN?` does
not contain the expected string; or a rig where you want start-up to be fully deterministic and
skip probing altogether. Setting a resource bypasses discovery entirely.

List what is connected, with each device's identity:

```python
from daq.instruments import probe_visa_resources
for resource, idn in probe_visa_resources():
    print(resource, "->", idn)
```

## Usage

Every instrument is a context manager whose exit **always** turns the output off, including
on exception. Prefer this form for anything unattended:

```python
from daq import Agilent33220A, TimeStream, trigger_for

with Agilent33220A() as bias:
    bias.sawtooth(vpp=2.0, freq_hz=500)          # gated on the Presto trigger by default
    ts = TimeStream(
        lo_freq=fr, if_freqs=[0], df=5e4,
        pixel_counts=bias.samples_for_periods(200, 5e4),
        amp=amp, output_port=1, input_port=1,
        device="my_device",
        external_trigger=trigger_for(bias),        # the port the generator says it is on
    )
    ts.attach(bias=bias)                          # settings -> HDF5 attrs + MongoDB
    ts.run()
```

Constant bias instead:

```python
with Agilent33220A() as bias:
    bias.constant(0.98)
    ts.attach(bias=bias)
    ts.run()
```

LED pulses:

```python
from daq import DC2200

with DC2200() as led:
    led.pwm_train(current_a=0.01, freq_hz=100, duty_pct=50, count=10)
```

See [LED driver (DC2200) modes](#led-driver-dc2200-modes) for a worked example of each mode.

In an exploratory cell you can skip the `with` and call `close()` when done — but then
nothing guarantees the output is de-energised if the cell raises.

### Recording instrument state

`Base.attach(**instruments)` flattens each instrument's read-back `settings()` onto the
measurement as `<prefix>_<key>` attributes, which the existing save path writes to the HDF5
attributes and the MongoDB document. Bias voltages and LED parameters become queryable via
`select_runs()` instead of surviving only inside a `notes` string.

```python
ts.attach(bias=bias, led=led)
# -> ts.bias_freq_hz, ts.bias_vpp, ts.led_pwm_count, ...
```

Values are queried back from the hardware rather than echoed from what was written, so the
record reflects what the instrument actually did.

**Attaching the bias generator also decides how the stream analyses itself.** `TimeStream`
reads `bias_function` back off the record — `RAMP` or `DC` — and `ts.analyze()` folds the
acquisition into one ramp period or plots its parity spectrum accordingly, on a reloaded file
as much as a live run. Without the `attach`, nothing in the samples says which was happening
and you get the plain I/Q plot. See *TimeStream → Analysis* in
[`daq/measurements/README.md`](../measurements/README.md). Attaching an LED is unaffected: the
DC2200 reports `mode`, not `function`, so it is never read as the gate bias.

### Synchronising the bias to an acquisition

`TimeStream(external_trigger=trigger_for(bias))` makes the Presto assert the generator's own
**trigger port** — 1 by default — for the duration of the acquisition. `sawtooth(gated=True)`
(the default) puts the 33220A in gated-burst mode, so the ramp runs only while that trigger is
asserted. Pair the two.

If the generator is wired elsewhere, tell the *instrument* (`Agilent33220A(trigger_port=2)`, or
`DAQ_FGEN_TRIGGER_PORT`) rather than the measurement, and `trigger_for` follows. See
[Routing the trigger](#routing-the-trigger).

`samples_for_periods(n_periods, sample_rate)` returns the `pixel_counts` spanning a whole
number of ramp periods **plus** the samples `TimeStream` discards at the start — pass the
same `discard_ms` you pass as `discard_start_ms`. Folding the result with
`daq.analysis.fold_timestream` requires that whole-period alignment.

The corollary: `sawtooth(gated=True)` with `external_trigger=False` records a **static** bias,
not a swept one. The generator sits at its burst start level while the gate is low, and nothing
asserts the gate. For a ramp that runs through an untriggered acquisition, use
`sawtooth(gated=False)`.

`QCTrace` packages the gated ramp with the gating already right, and `BiasHunt` the
constant-bias hunt with nothing gated. Both read out at a frequency you supply — normally the
`fr` from a fitted `Sweep` — and both leave the gate de-energised on the way out:

```python
from daq import BiasHunt, QCTrace

qct = QCTrace(readout_freq=fr, amp=amp, output_port=1, input_port=1, device="my_device")
qct.run()                       # opens and closes the 33220A itself
# ...or keep ownership of the session:
with Agilent33220A() as bias:
    qct.run(bias=bias)          # output de-energised on the way out, session left open
    BiasHunt(readout_freq=fr, amp=amp, output_port=1, input_port=1,
             v_min=0.0, v_max=2.0, device="my_device").run(bias=bias)
```

`QCTrace` gates on the generator's own `trigger_port`, so a rewired rig needs no change here
either; `QCTrace(trigger_states=…)` overrides the routing for one measurement, and a routing
that gates nothing raises rather than recording a static bias. `BiasHunt` gates nothing at all
— its bias is a DC level already written over SCPI.

## LED driver (DC2200) modes

The DC2200 has four modes, selected by the `configure_*` method you call. They differ in
**what generates the timing**, which is what should drive your choice:

| Mode | Method | Timing generated by | Use it for |
|---|---|---|---|
| Constant current | `configure_cc()` | nothing — steady output | continuous illumination, checking the LED is alive |
| PWM | `configure_pwm()` / `pwm_train()` | the DC2200's PWM engine, from a **duty cycle** | pulse trains where width and rate can be coupled |
| Pulse | `configure_pulse()` | the DC2200's pulse engine, from explicit **ON/OFF times** | short pulses at a slow rate, e.g. 10 µs at 2 Hz |
| TTL | `configure_ttl()` | an external signal on the modulation input | illumination synchronised to a `TimeStream` |

**Choosing between PWM and Pulse.** PWM takes a *duty cycle*, and the instrument's duty floor
is **0.1 %** (bench-confirmed), so the narrowest pulse PWM can produce is

```
min width = 0.001 / freq_hz
```

which couples width to rate — 10 µs only at 100 Hz, 500 µs at 2 Hz. Pulse mode takes the width
directly and so decouples the two. If your required width and repetition rate don't satisfy
that relation, you need `configure_pulse()`.

Common to all four: configuring **never** illuminates the LED. Each `configure_*` method
leaves the output disabled unless you pass `output=True`, so the LED lights only when the
output is enabled — and, in TTL mode, only while the modulation input is also high. On exit
from the `with` block the output is switched off regardless of how the block ended.

### Constant current (CC)

The simplest mode: a fixed current, on until you turn it off. No timing structure at all.

```python
from daq import DC2200

with DC2200() as led:
    led.configure_cc(current_a=0.005)   # 5 mA; LED still dark
    led.output = True                   # LED on, and stays on
    print(f"drawing {led.measured_current * 1e3:.2f} mA")
    ...
# LED off here
```

| Argument | Value | Meaning |
|---|---|---|
| `current_a` | 0.005 | Drive the LED continuously at **5 mA** |

`configure_cc()` writes `SOURCE1:MODE CC` then `SOURCE1:CCURENT:CURRENT 0.005` (Thorlabs'
own misspelling of "CCURRENT" — required verbatim) and leaves the output off. The LED lights
the moment `led.output = True` executes and stays lit until the output is disabled or the
`with` block exits.

`measured_current` reads `SENSE3:CURRENT:DATA?`, the current the instrument actually measures
— a useful check that the LED is drawing what you asked for rather than being open-circuit.

`settings()` reports `mode` and `cc_current_a`, so `attach()` records `led_cc_current_a`.

Pass `configure_cc(..., output=True)` to configure and illuminate in one call.

### PWM pulse train

The DC2200 generates a burst of `count` pulses at a set frequency and duty cycle, then stops
by itself. Pulse timing is instrument-accurate; only the moment the train *starts* is
software-timed, so this is **not** synchronised with an acquisition. TTL mode is what
synchronises to a `TimeStream`, but note it yields a steady illumination window rather than
pulses — see [Getting a genuinely short pulse](#getting-a-genuinely-short-pulse).

`pwm_train()` is the blocking convenience form: configure, fire, wait for the train to
finish, switch off.

```python
with DC2200() as led:
    led.pwm_train(current_a=0.01, freq_hz=100, duty_pct=50, count=10)
```

| Argument | Value | Meaning |
|---|---|---|
| `current_a` | 0.01 | Drive the LED at 10 mA during the on-phase of each pulse |
| `freq_hz` | 100 | Repeat every 1/100 s = **10 ms** |
| `duty_pct` | 50 | The LED is on for 50 % of each period = **5 ms on, 5 ms off** |
| `count` | 10 | Emit **10 pulses**, then stop |

So the train is 10 × 10 ms = **100 ms long** and carries 10 × 5 ms = **50 ms of total LED-on
time**:

```
10 mA ┐  ┌─┐  ┌─┐  ┌─┐          ... 10 pulses ...
      │  │ │  │ │  │ │
   0 ─┴──┘ └──┘ └──┘ └──                        ──────────
      |<5ms>|                                   ^ train ends after 100 ms
      |<--10 ms-->|
```

Step by step, the method:

1. Writes the PWM settings — `SOURCE1:MODE PWM`, then `PWM:CURRENT 0.01`, `PWM:FREQ 100`,
   `PWM:DCYCLE 50`, `PWM:COUNT 10` — and explicitly writes `OUTPUT1:STATE OFF`, so the train
   cannot start part-way through configuration.
2. Enables the output (`OUTPUT1:STATE ON`), which starts the train.
3. Sleeps `count / freq_hz + settle_s` = 0.1 s + 0.5 s = **0.6 s**, so the Python call blocks
   for 0.6 s even though the train itself lasts 100 ms. The extra `settle_s` margin means a
   slightly slow instrument is not cut off mid-train.
4. Disables the output in a `finally`, so the LED is off even if the wait is interrupted.

Afterwards the instrument is left in PWM mode with these settings and the output off, so
`settings()` — and therefore `attach()` — reports `led_pwm_current_a`, `led_pwm_freq_hz`,
`led_pwm_duty_pct` and `led_pwm_count`.

`count` must be a whole number. Very large values are firmware-limited (100000 on firmware
1.3.0 and later).

> **The duty floor is 0.1 %.** Requesting less is silently coerced up, so the pulse you get is
> wider than the one you asked for. Since width = `duty/100 / freq_hz`, PWM's narrowest pulse
> is `0.001 / freq_hz`:
>
> | Rate | Narrowest PWM pulse |
> |---|---|
> | 2 Hz | 500 µs |
> | 10 Hz | 100 µs |
> | 100 Hz | 10 µs |
>
> Always read the value back (`led.settings()["pwm_duty_pct"]`) rather than trusting what you
> wrote. For a short pulse at a slow rate, use [pulse mode](#pulse-mode-explicit-onoff-times) instead.

### Pulse mode (explicit ON/OFF times)

Takes the pulse **ON time** and **OFF time** independently rather than a duty cycle, so width
and repetition rate are decoupled. This is the mode for a narrow pulse at a slow rate — the
combination PWM cannot reach.

```python
with DC2200() as led:
    led.configure_pulse(on_time_s=10e-6, freq_hz=2.0, current_a=1.0e-3, count=0)
    led.output = True      # starts the train, non-blocking
    ...
    led.output = False
```

| Argument | Value | Meaning |
|---|---|---|
| `on_time_s` | 10e-6 | LED on for **10 µs** per pulse |
| `freq_hz` | 2.0 | Repeat every **500 ms**; the OFF time is derived as `period − on_time` |
| `current_a` | 1.0e-3 | **1 mA**, converted to a brightness percentage (see below) |
| `count` | 0 | **Infinite** — run until the output is disabled |

Instrument ranges: ON and OFF time each **0.001 ms to 10 s**, giving an overall **0.05 Hz to
500 Hz**; count 1–1000, or `0` for infinite. Specify the rate as exactly one of `off_time_s`,
`freq_hz` or `period_s`.

> **Pulse mode is brightness-based, not current-based.** The instrument takes a percentage of
> the *currently configured current limit* — 100 % means driving at the limit. `current_a` is
> accepted and converted using the queried limit, or pass `brightness_pct` directly.
>
> The consequence matters: with a 1.2 A limit, 1 mA is 0.083 %, which the instrument's
> brightness resolution may not reach. **Lower the LED current limit** to gain fine control
> near the bottom of the range — with a 10 mA limit, 1 mA is a comfortable 10 %.

`count=0` is usually what you want when the train should span an acquisition: it runs until you
disable the output, so there is no need to compute a count that outlasts the record.

```python
led.configure_pulse(on_time_s=10e-6, freq_hz=2.0, current_a=1.0e-3, count=0)
ts.attach(led=led)
led.output = True
try:
    ts.run()
finally:
    led.output = False
```

Like PWM, the train starts on the software write that enables the output, so it is **not**
synchronised to the acquisition. In this mode the SMA connector *outputs* the internal
modulation as a TTL signal, so you can feed the LED timing back to the Presto as a marker if
you need to know exactly when each pulse fired.

`settings()` reports `pulse_brightness_pct`, `pulse_on_time_s`, `pulse_off_time_s`,
`pulse_count`, the derived `pulse_freq_hz`, and `pulse_current_a` (the brightness converted
back through the limit) — so `attach()` records all of it.

### TTL modulation

"TTL" here is the ordinary digital-logic sense — *transistor–transistor logic*, i.e. a 0 V-low
/ ~5 V-high on/off signal. In this mode the LED follows such a signal on the rear-panel SMA
modulation input: on at the configured current while the input is high, off while it is low.
This is the mode to use when the illumination must line up with an acquisition, because the
timing comes from the Presto rather than from Python.

```python
with DC2200() as led:
    led.configure_ttl(current_a=0.01)   # mode + current; LED not armed
    led.output = True                   # arm; LED now follows the modulation input
```

| Argument | Value | Meaning |
|---|---|---|
| `current_a` | 0.01 | Drive the LED at **10 mA** whenever the modulation input is high |

That single current is the only parameter the mode has — on-time and repetition are whatever
the driving signal does. The input takes TTL levels — low 0–0.8 V, high 2.0–5.0 V — into 10 kΩ. Driven from a `TimeStream`,
that signal is high for the whole acquisition, so plan for an illumination window rather than
a pulse.

`configure_ttl()` writes `SOURCE1:MODE TTL` and the drive current, leaving the output off.
Enabling the output arms the stage; the LED then lights whenever the input goes high.

> Thorlabs documents TTL mode as having exactly one settable parameter but does not publish
> the SCPI header for it. `configure_ttl()` tries the candidates in
> `DC2200.TTL_CURRENT_HEADERS` against the instrument and keeps whichever is accepted, so it
> resolves on the first call rather than relying on a guess.

`settings()` reports `mode` and `ttl_current_a`.

See [Time stream synchronised with the LED](#time-stream-synchronised-with-the-led)
below for the wiring, the timing, and when exactly the LED lights.

## Routing the trigger

`TimeStream`'s `external_trigger` maps straight onto presto's `Lockin.set_trigger_out`, which
takes one state **per digital output port**: element *i* configures port *i+1*, where `0` is
off, `1` triggers on every lock-in window and `2` on every sum window (inert here — plain
`get_pixels()` does no summing). So the parameter is really "which ports go high", and what it
gates depends entirely on your wiring.

| `external_trigger` | ports asserted | the lab's default wiring |
|---|---|---|
| `False` | none | — |
| `True` (= `[1]`) | 1 | Agilent 33220A gate |
| `[0, 1]` | 2 | DC2200 modulation input |
| `[1, 1]` | 1 and 2 | both, fired together |

The Presto has four digital output ports (presto packs the states two bits at a time into a
single byte, and `Pulsed.output_digital_marker` bounds ports to 1–4), so a list longer than
four raises rather than silently overflowing the wire format.

### Let the instrument name its own port

Writing those states by hand is the one mistake this section exists to prevent, and it fails
**silently**: a gated instrument on a port nobody asserted just never fires — the ramp holds
its start level, the LED stays dark — and the acquisition succeeds with data that looks like a
dead device. So each driver carries the port it is wired to, and `trigger_for` builds the
states from the instruments:

```python
from daq import Agilent33220A, DC2200, TimeStream, trigger_for

with Agilent33220A() as bias, DC2200() as led:
    trigger_for(bias)        # -> [1]      the generator's port
    trigger_for(led)         # -> [0, 1]   the LED's port
    trigger_for(bias, led)   # -> [1, 1]   both, fired together
    ts = TimeStream(..., external_trigger=trigger_for(led))
```

The default ports are the lab's wiring — 1 for the `Agilent33220A`, 2 for the `DC2200`.
A rig that differs says so once, in whichever place fits:

| where | how | scope |
|---|---|---|
| per instrument | `Agilent33220A(trigger_port=3)`, or `bias.trigger_port = 3` | one session |
| per computer | `DAQ_FGEN_TRIGGER_PORT`, `DAQ_LED_TRIGGER_PORT` | every run on that rig |
| per model | the driver's `TRIGGER_PORT` class attribute | the repo default |

Ports must be 1–4; anything else raises where you set it, rather than producing a states list
that quietly gates nothing. `trigger_for` also accepts a bare port number (`trigger_for(3)`) if
no instrument object is in hand, and raises if an instrument's `trigger_port` is `None` or if
called with no source at all — `trigger_for(*instruments)` over a list that turned out empty
would otherwise hand back a silently ungated acquisition. Gating nothing is spelled
`external_trigger=False`, at the call site, where it is visible.

`settings()` reports `trigger_port`, so `attach()` writes it into the HDF5 attributes and the
MongoDB document — the wiring is the one part of a measurement the data file cannot otherwise
show, and the part whose being wrong looks exactly like a dead detector.

### Worked example: moving the LED to another port

Say the DC2200's modulation input has been moved from port 2 to port 4 — the LED cable now
shares a feedthrough with something else. Nothing about the measurement changes; the *rig*
changed, so you say so once:

```bash
export DAQ_LED_TRIGGER_PORT=4        # or: DC2200(trigger_port=4), for one session
```

```python
from daq import DC2200, TimeStream, trigger_for

with DC2200() as led:
    led.configure_ttl(current_a=0.01)             # mode + current; LED not armed yet
    ts = TimeStream(
        lo_freq=fr, if_freqs=[0], df=5e4, pixel_counts=50_000,
        amp=amp, output_port=1, input_port=1,
        device="my_device",
        external_trigger=trigger_for(led),        # -> [0, 0, 0, 1], follows the env var
        discard_start_ms=0,                       # keep the turn-on edge
    )
    led.output = True                             # arm immediately before the run
    ts.attach(led=led)                            # records led_trigger_port=4 in the file
    ts.run()
```

Every acquisition in every notebook follows, because none of them names a port. Written the
old way — `external_trigger=[0, 1]` copied from a recipe, or `True` out of habit — this run
would have completed normally with the LED dark for all 1 s of it, and nothing in the data or
the saved record would say why.

The same shape applies to the gate generator, and there it also reaches `QCTrace`, which
previously hardcoded port 1 and could not be routed at all:

```bash
export DAQ_FGEN_TRIGGER_PORT=3
```

```python
from daq import QCTrace

qct = QCTrace(readout_freq=fr, amp=amp, output_port=1, input_port=1, device="my_device")
qct.run()        # "QC trace: gating the ramp on Presto digital output port 3"
```

`run()` re-reads the generator each time, so if you discover mid-session that the gate is
actually on port 2, `bias.trigger_port = 2` and re-running the same object gates port 2.

**Timing is global, not per port.** presto sends one `delay`/`width` pair alongside `df`, so
every port enabled in the same acquisition is gated identically. You cannot hold one port high
across the record while pulsing another; that needs separate acquisitions.

**What the line actually does.** `TimeStream.TRIGGER_WIDTH_S` is 30 ms, but this is *not* a
30 ms one-shot. presto re-asserts the trigger at the start of **every lock-in window**, i.e.
every `1 / df` — 20 µs at `df = 50 kHz`. Since the configured width is ~1500 windows long, the
next rising edge always arrives before the falling edge is due, so:

```
acquisition starts  ->  trigger HIGH
   ... entire record, line stays high ...
acquisition ends    ->  set_trigger_out([0]) + apply_settings  ->  LOW
```

A width *shorter* than `1 / df` would instead chop the line into a pulse train at the sample
rate — a 50 kHz chopper, essentially never what you want. There is no width that yields a
single short pulse inside a long record; see [Getting a genuinely short
pulse](#getting-a-genuinely-short-pulse).

> **Inferred, not yet measured.** The continuous-high behaviour follows from the presto
> Python layer (the docstring's "at the start of every demodulation window", and the
> window-relative `(start, stop)` clock pair sent with `df`) and is consistent with the
> archived bench predecessor of `QCTrace`, which took usable QC traces under this exact
> gated-ramp, 30 ms-width configuration over whole records. (`QCTrace` itself has not yet been
> bench-run, so it is not independent evidence here.) The FPGA's exact response to `width`
> exceeding the window period has not been checked on a scope. If you have one on the bench, probing the trigger
> line during an `external_trigger` run settles this and the idle-level question below in one
> trace — and is worth doing before designing a measurement around the timing.

## Time stream synchronised with the LED

The LED is gated by the **Presto's trigger output**, not by a software call. Software timing is
useless here: `TimeStream.run()` spends an indeterminate time connecting, configuring the mixer
and tuning before acquisition actually starts, so an LED switched on from Python lands
somewhere unknown relative to the data. The trigger line is asserted immediately before
acquisition begins, which is the only reliable reference.

### Wiring

```
Presto digital output port 2  ──►  DC2200 rear-panel SMA modulation input
```

Port 2 is the lab's default; port 1 carries the 33220A gate. If yours is elsewhere, set it on
the driver — `DC2200(trigger_port=…)` or `DAQ_LED_TRIGGER_PORT` — and gate the acquisition with
`external_trigger=trigger_for(led)`. Naming the port in the measurement instead is how the LED
ends up dark for a whole run.

The DC2200 modulation input takes TTL levels (low 0–0.8 V, high 2.0–5.0 V, 10 kΩ). In TTL mode the LED drives at the
configured current while that input is high and is off while it is low, so the LED is lit for
exactly as long as the Presto asserts that port: the whole acquisition.

### Recipe

```python
from daq import DC2200, TimeStream, trigger_for

TIME_TOTAL_S, FS = 1.0, 5e4

with DC2200() as led:
    led.configure_ttl(current_a=0.01)      # sets the mode + current; LED NOT armed yet

    ts = TimeStream(
        lo_freq=fr, if_freqs=[0], df=FS,
        pixel_counts=int(FS * TIME_TOTAL_S),
        amp=amp, output_port=1, input_port=1,
        device="my_device",
        notes="LED response",
        external_trigger=trigger_for(led),  # the port the LED says it is wired to
        discard_start_ms=0,                # keep the turn-on edge -- see below
    )

    led.output = True                      # arm: from here the LED follows the trigger
    ts.attach(led=led)                     # led_mode, led_ttl_current_a, ... -> HDF5 + MongoDB
    ts.run()

ts.analyze()                               # LED turn-on at t = 0, lit for the whole record
```

> **`discard_start_ms=0` keeps the turn-on edge.** `TimeStream` drops the first 25 ms of every
> acquisition by default. The LED comes up with the RF drive at `t = 0`, so the default trim
> throws away the turn-on transient and the detector's approach to steady state — usually the
> most interesting part. It does not hide the illumination itself, since the LED stays on.

**When does the LED actually light?** Not on any of the first four lines — it lights inside
`run()`:

| Line | Hardware effect | LED |
|---|---|---|
| `led.configure_ttl(current_a=0.01)` | `SOURCE1:MODE TTL` + current; output left off | dark, guaranteed |
| `ts = TimeStream(...)` | none — pure Python object construction | dark |
| `led.output = True` | `OUTPUT1:STATE ON`, arming the output stage | dark *if the trigger idles low*; from here it tracks the modulation input |
| `ts.attach(led=led)` | VISA queries only, reading state back | unchanged |
| `ts.run()` | connect → configure mixer → tune → `set_trigger_out([0, 1], width=…)` → **`apply_settings()`** → `get_pixels()` | **lights here**, for the acquisition |

`set_trigger_out()` only *stages* the trigger; `apply_settings()` is what asserts the line, and
it runs immediately before `get_pixels()` starts acquiring. The readout tone's amplitude is
applied by that same call, so the LED and the RF drive come up together.

The one way to light the LED earlier is the idle-level caveat below: if the trigger line sits
high between acquisitions, the LED comes on at `led.output = True` and stays on.

**Configuring is not arming.** `configure_ttl()` only selects the mode and the current — it
leaves the output disabled, so it cannot illuminate the LED as a side effect. Enabling the
output (`led.output = True`) arms the output stage, and only from that point does the LED
follow the modulation input. Pass `configure_ttl(..., output=True)` to do both at once if you
want, but the two-step form above is what the recipe uses deliberately:

> **Arm as late as possible.** `TimeStream.run()` connects to the Presto, configures the mixer
> and tunes before it acquires anything, which takes appreciably longer than the acquisition
> itself. If the trigger line idles high, everything armed before that point sits illuminated
> for the whole setup — shining on the detector with no data being taken. Arming immediately
> before `run()` keeps that window as short as the API allows.
>
> The idle level of the Presto trigger output between acquisitions is **not something this
> library controls, and we have not measured it.** `run()` drives it low at the end of a
> triggered acquisition, so it should idle low once a triggered run has happened, but after a
> fresh Presto boot it is unverified. Check it on a scope before trusting a long armed window,
> and if it idles high, keep the LED disarmed until the last moment (as above).

On exit the LED is switched off even if the acquisition raises.

### Averaging repeated acquisitions

Each `run()` is one illuminated record. To average, loop — each acquisition is saved and
logged as usual:

```python
with DC2200() as led:
    led.configure_ttl(current_a=0.01, output=True)   # arm once, fires once per run()
    streams = []
    for i in range(50):
        ts = TimeStream(..., external_trigger=trigger_for(led), discard_start_ms=0,
                        notes=f"LED acquisition {i + 1}/50")
        ts.attach(led=led)
        ts.run()
        streams.append(ts)

records = np.stack([s.signal[:, 0] for s in streams])
mean_record = records.mean(axis=0)
```

Do not use `fold_timestream` for this — it folds *one* record containing many drive periods,
which is the sawtooth-bias case. Here each record holds a single illumination window, so
average across records.

### Getting a genuinely short pulse

TTL mode gives you an illumination *window* locked to the acquisition, not a short pulse
inside it. If what you need is a brief flash, the options are:

- **[PWM](#pwm-pulse-train) or [pulse mode](#pulse-mode-explicit-onoff-times)** — the DC2200
  generates the timing itself, so you get real short pulses, but the train *starts* on a
  software-timed USB write. Started before `run()` that means an unknown seconds-scale offset;
  started via `on_acquire` (below) the offset shrinks to a few milliseconds.
- **`presto.pulsed`** — `Pulsed.output_digital_marker(at_time, duration, ports)` schedules a
  digital marker of arbitrary duration at a defined time in a pulse sequence. That is a proper
  hardware-timed short pulse, but it is a different presto measurement mode from `Lockin` and
  is not wrapped by this library; `TimeStream` cannot produce it by changing a parameter.

### Short pulses at a few-ms offset: `on_acquire`

When exact sync is out of reach (no free wiring path, `presto.pulsed` not warranted) but a
few-millisecond alignment is enough, start the pulse train from **inside** `run()`, at the
last moment before data:

```
ts.run(on_acquire=...)
  ├─ connect / configure mixer / tune        seconds, variable  ← why pre-run() timing fails
  ├─ apply_settings()                        trigger (if configured) asserts here
  ├─ on_acquire()                            ← your callable: a few SCPI round trips
  └─ get_pixels()                            sample zero
```

`on_acquire` is called exactly once, after all the slow setup, immediately before
acquisition. A pulse train started there sits within **milliseconds-to-tens-of-ms** of sample
zero — set by a few SCPI round trips (each `write` brackets the command with `SYST:ERR?`
checks) racing the `get_pixels` start round-trip — instead of the seconds-scale, per-run
variable offset of anything started before `run()`:

```python
with DC2200() as led:
    led.configure_pulse(on_time_s=10e-6, freq_hz=2.0, current_a=1.0e-3, count=0)
    ts = TimeStream(..., discard_start_ms=0)
    ts.attach(led=led)
    ts.run(on_acquire=lambda: setattr(led, "output", True))
    led.output = False
```

Every record then holds pulses at exact instrument-timed spacing (500 ms here), at a roughly
repeatable offset from acquisition start.

**The offset's sign and size are not yet bench-measured.** Which side of sample zero the
train starts on decides what happens to pulse one:

- Train starts *before* sample zero → a short first pulse (10 µs here) fires before
  acquisition and is never recorded; the first *recorded* pulse lands near
  `period − offset`.
- Train starts *after* sample zero → pulse one lands at small positive `t`, where the
  default 25 ms `discard_start_ms` trim could eat it — hence `discard_start_ms=0` in the
  recipe.

Either way the actual offset is read off the data: the position of the first recorded pulse,
**modulo the pulse period**. Measure it on the first bench record before relying on a number.

Keep the callable to a single fast write. If it raises, the acquisition is abandoned and the
exception propagates; `run()` mutes the Presto outputs on the way out, but disarming the
instrument stays with its `with` block. The Presto trigger ports are untouched by this
mechanism, so a gated bias ramp on port 1 works in the same acquisition — note the hook's
latency then sits between the ramp start (trigger assertion at `apply_settings`) and sample
zero, adding the same ms-scale skew to the ramp-vs-data alignment that `fold_timestream`
assumes small (~1 % of a 500 ms period per 5 ms of offset).

### Both instruments at once

`attach()` takes any number of instruments, so a run with both LED and gate bias records both.
Trigger both ports together by naming both instruments:

```python
with DC2200() as led, Agilent33220A() as bias:
    led.configure_ttl(current_a=0.01)      # LED on its own port (2 by default)
    bias.sawtooth(vpp=2.0, freq_hz=500)    # gated ramp on the generator's port (1)
    ts = TimeStream(..., external_trigger=trigger_for(led, bias), discard_start_ms=0)
    led.output = True                      # arm last, immediately before run()
    ts.attach(led=led, bias=bias)
    ts.run()
```

Both ports share one `delay`/`width`, so this fires them simultaneously and holds both high for
the acquisition — you cannot stagger them. If the bias should ramp while the LED stays dark,
use `trigger_for(bias)` and leave the LED output disabled; the reverse is `trigger_for(led)`
with `bias.constant(...)` instead of a gated ramp.

## SCPI gotchas

Hard-won, and all required verbatim.

| Instrument | Gotcha |
|---|---|
| DC2200 | Constant-current header is `SOURCE1:CCURENT:CURRENT` — Thorlabs' own misspelling, missing the second `R`. |
| DC2200 | PWM frequency is `SOURCE1:PWM:FREQ`; the long form `FREQUENCY` is rejected with `-113 Undefined header`. |
| DC2200 | Every header is taken from the Operation Manual v1.8 §4.3.2, including the optional-node short forms (`SOURce1:PULSe[:BRIGhtness][:LEVel][:AMPLitude]` → `SOURCE1:PULSE:BRIGHTNESS:LEVEL:AMPLITUDE`). Nothing is guessed or probed. |
| DC2200 | Pulse mode takes **brightness in percent of the current limit**, not a current, and **ONTime/OFFTime in seconds** — there is no width or frequency command. `configure_pulse()` converts a requested `current_a` and rate for you. |
| DC2200 | The DC2200 has **two output terminals** and source settings apply to the selected one. Use the `terminal` property rather than inheriting the front-panel choice. |
| DC2200 | The PWM **duty cycle floor is 0.1 %** and a smaller request is coerced up silently, so the pulse is wider than asked. Read it back, or use pulse mode where the width is explicit. |
| 33220A | Burst mode and a DC carrier are mutually exclusive. Writing `FUNC DC` while `BURS:STAT` is still `ON` from an earlier ramp is a settings conflict, so `constant()` disables burst **first**. Both mode setters write their full state for this reason. |
| 33220A | Minimum programmable amplitude is 20 mVpp into high-Z (`OUTP:LOAD INF`), 10 mVpp into 50 Ω. Below that the instrument does not output what you asked for; `sawtooth()` raises instead. |
| 33220A | `OUTP:LOAD INF` is the high-impedance setting (the Siglent spelling for the same thing is `LOAD,HZ`). |
| All | The error queue is drained before every write and re-read after, so a rejected command raises `InstrumentError` instead of being silently swallowed. Stale faults are never blamed on a later command. |

## Debugging

### An instrument will not connect

**1. Get the resource string yourself, bypassing discovery entirely.** This is always the
fastest way to unblock:

```python
from daq.instruments import probe_visa_resources

for resource, idn in probe_visa_resources():
    print(f"{resource}\n    -> {idn}")
```

```
USB0::0x0957::0x0407::MY44000531::INSTR
    -> Agilent Technologies,33220A,MY44000531,2.01
GPIB0::30::INSTR
    -> <error: TimeoutError: VI_ERROR_TMO: Timeout expired before operation completed>
```

That output splits the problem three ways:

| What you see | What it means | What to do |
|---|---|---|
| **Only serial ports** (`ASRL1::INSTR` and similar), no `USB`/`GPIB` entries | Almost always: the instrument is not plugged in, or not powered on | **Check the USB cable and the front panel first.** This is the most common failure by a wide margin, and an unplugged cable is indistinguishable from a driver fault at the VISA layer |
| The instrument is **not listed at all**, but other USB/GPIB instruments are | Driver or cabling for that one device | Check Device Manager / NI MAX; not WSL |
| Listed, but `<error: ...TMO...>` | Visible but not answering — held by another process, or slower than the probe allows | Close NI MAX / other kernels; or raise `Agilent33220A.PROBE_TIMEOUT_MS` |
| Listed with an `*IDN?` that looks right | Discovery's keyword check is what is failing | Pass the resource explicitly (below) |

**2. Pass it explicitly.** Autodiscovery is a convenience, not the only path:

```python
bias = Agilent33220A(resource="USB0::0x0957::0x0407::MY44000531::INSTR")
```

Once it works, put it in the environment so notebooks need not repeat it:

```bash
export DAQ_FGEN_RESOURCE="USB0::0x0957::0x0407::MY44000531::INSTR"
```

> **Setting the environment variable from inside a running kernel does not take effect on its
> own.** Settings are cached on first access, so `os.environ[...] = ...` after `import daq` is
> silently ignored. Either set it before starting the kernel, or call
> `daq.config.reload_settings()` afterwards. Passing `resource=` sidesteps the cache entirely.

**2b. Check which VISA library is loaded.** `visa_backend_info()` names it, which distinguishes
a vendor VISA from the pure-Python backend (the latter cannot see USB instruments at all
without `pyusb` and `libusb`):

```python
from daq.instruments import visa_backend_info
print(visa_backend_info())
# IVIVisaLibrary: Visa Library at C:\Windows\system32\visa32.dll
```

**3. Read the error.** A failed autodiscovery reports every resource it saw and why each was
rejected — a model mismatch and a timeout look nothing alike and need different fixes:

```
No connected instrument identified as Agilent33220A -- looked for ['33220A'] in the
*IDN? response of each visible resource:
  USB0::0x0957::0x0407::MY123::INSTR: Agilent Technologies,33210A,MY123,1.0
  GPIB0::30::INSTR: no *IDN? response (TimeoutError: VI_ERROR_TMO: ...)
```

The first line there is the common real case: a **33210A**, not a 33220A. The driver refuses to
adopt it, because a near-identical model is exactly the sort of thing that silently produces
plausible-but-wrong data. If you know the model is compatible, pass `resource=` to skip the
check, or add the keyword to `Agilent33220A.IDN_KEYWORDS`.

### SCPI transcripts

Pass `transcript_path=` to record every write, query and error response to a text file:

```python
with Agilent33220A(transcript_path="fgen_run.txt") as bias:
    ...
```

Useful when a command is accepted but does not do what you expect — the transcript shows the
exact byte sequence and each `SYST:ERR?` response.

### Other failures

| Symptom | Likely fix |
|---|---|
| `Could not locate a VISA implementation` | Install NI-VISA, or set `DAQ_VISA_BACKEND=@py`. |
| `No VISA resources are visible` | Cable/power; check Device Manager or NI MAX; not WSL. |
| `Found N instruments identifying as ...` | Two of the same model. Set `DAQ_FGEN_RESOURCE` / `DAQ_LED_RESOURCE`, or pass `resource=`. |
| Hang, or `VI_ERROR_RSRC_NFOUND` | Another process holds the instrument; shut down other kernels. Unplug/replug. |
| `pyvisa is required ... not installed` | `pip install daq[instruments]`. |

## Adding an instrument

Subclass `VisaInstrument`, set `IDN_KEYWORDS` (and `RESOURCE_HINTS` if the resource string is
distinctive), override `env_resource()` to read its config setting, override `safe_state()` to
define "off", and extend `settings()` with read-back state. The base class handles the lazy
import, discovery, error checking, context-manager lifetime and transcript logging.
