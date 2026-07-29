# Instruments

Drivers for the non-Presto benchtop hardware in the lab, reached over VISA (SCPI).

These are **instruments, not measurements**: they produce no data arrays and do not subclass
`Base`. You compose them with the ordinary measurement classes in a notebook, and record
their state with `Base.attach()`.

| Class | Hardware | Purpose |
|---|---|---|
| `Agilent33220A` | Agilent/Keysight 33220A | Gate-bias source: constant DC or gated sawtooth ramp |
| `DC2200` | Thorlabs DC2200 | LED driver: constant current or PWM pulse train |

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
| `DAQ_VISA_BACKEND` | `""` for NI-VISA (default), `"@py"` for pyvisa-py. |

When a resource is not configured, the driver lists every VISA resource, asks each one
`*IDN?`, and keeps the single match for its model. It deliberately **raises** when zero or
several match rather than picking the first — grabbing `list_resources()[0]` silently binds
to the wrong instrument as soon as a second one is plugged in, and produces plausible-looking
data from it.

List what is connected:

```bash
python -c "import pyvisa; print(pyvisa.ResourceManager().list_resources())"
```

## Usage

Every instrument is a context manager whose exit **always** turns the output off, including
on exception. Prefer this form for anything unattended:

```python
from daq import Agilent33220A, TimeStream

with Agilent33220A() as bias:
    bias.sawtooth(vpp=2.0, freq_hz=500)          # gated on the Presto trigger by default
    ts = TimeStream(
        lo_freq=fr, if_freqs=[0], df=5e4,
        pixel_counts=bias.samples_for_periods(200, 5e4),
        amp=amp, output_port=1, input_port=1,
        device="my_device", external_trigger=True,
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
    led.pulse_train(current_a=0.01, freq_hz=100, duty_pct=50, count=10)
```

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

### Synchronising the bias to an acquisition

`TimeStream(external_trigger=True)` makes the Presto assert its trigger output when the
acquisition starts. `sawtooth(gated=True)` (the default) puts the 33220A in gated-burst mode,
so the ramp runs only while that trigger is asserted. Pair them.

`samples_for_periods(n_periods, sample_rate)` returns the `pixel_counts` spanning a whole
number of ramp periods **plus** the samples `TimeStream` discards at the start — pass the
same `discard_ms` you pass as `discard_start_ms`. Folding the result with
`daq.analysis.fold_timestream` requires that whole-period alignment.

## Time stream synchronised with an LED pulse

The LED is fired by the **Presto's trigger output**, not by a software call. Software timing
is useless here: `TimeStream.run()` spends an indeterminate time connecting, configuring the
mixer and tuning before acquisition actually starts, so an LED switched on from Python lands
somewhere unknown relative to the data. The trigger line is asserted immediately before
acquisition begins, which is the only reliable reference.

### Wiring

```
Presto trigger output  ──►  DC2200 rear-panel SMA modulation input
```

The DC2200 modulation input accepts 0–5 V at up to 250 kHz. In TTL mode the LED drives at the
configured current while that input is high and is off while it is low, so the LED is on for
exactly as long as the Presto asserts its trigger.

### Timing

`TimeStream(external_trigger=True)` asserts the trigger for **30 ms** starting at the instant
acquisition begins:

```
t = 0 ms     trigger HIGH  ->  LED on;  acquisition starts
t = 30 ms    trigger LOW   ->  LED off
t > 30 ms    acquisition continues, recording the decay
```

So a single acquisition captures the LED-on window, the turn-off, and the detector's recovery
afterwards. The 30 ms width is currently hardcoded in `TimeStream.run()`, so it is the same
for every acquisition.

> **Set `discard_start_ms=0`.** `TimeStream` discards the first 25 ms of every acquisition by
> default, which would throw away all but the last 5 ms of a 30 ms LED pulse. For LED-pulse
> measurements you almost always want the start of the record, so disable the trim. This is
> the easiest way to get a time stream that appears to show no LED response at all.

### Recipe

```python
from daq import DC2200, TimeStream

TIME_TOTAL_S, FS = 1.0, 5e4

with DC2200() as led:
    led.configure_ttl(current_a=0.01)      # sets the mode + current; LED NOT armed yet

    ts = TimeStream(
        lo_freq=fr, if_freqs=[0], df=FS,
        pixel_counts=int(FS * TIME_TOTAL_S),
        amp=amp, output_port=1, input_port=1,
        device="my_device",
        notes="LED pulse response",
        external_trigger=True,             # Presto gates the LED
        discard_start_ms=0,                # keep the pulse -- see the warning above
    )

    led.output = True                      # arm: from here the LED follows the trigger
    ts.attach(led=led)                     # led_mode, led_ttl_current_a, ... -> HDF5 + MongoDB
    ts.run()

ts.analyze()                               # the pulse sits in the first 30 ms
```

**Configuring is not arming.** `configure_ttl()` only selects the mode and the current — it
leaves the output disabled, so it cannot illuminate the LED as a side effect. Enabling the
output (`led.output = True`) arms the output stage, and only from that point does the LED
follow the modulation input. Pass `configure_ttl(..., output=True)` to do both at once if you
want, but the two-step form above is what the recipe uses deliberately:

> **Arm as late as possible.** `TimeStream.run()` connects to the Presto, configures the mixer
> and tunes before it acquires anything, which takes appreciably longer than the 30 ms pulse
> you are trying to record. If the trigger line idles high, everything armed before that point
> sits illuminated for the whole setup — shining on the detector with no data being taken.
> Arming immediately before `run()` keeps that window as short as the API allows.
>
> The idle level of the Presto trigger output between acquisitions is **not something this
> library controls, and we have not measured it.** `run()` drives it low at the end of a
> triggered acquisition, so it should idle low once a triggered run has happened, but after a
> fresh Presto boot it is unverified. Check it on a scope before trusting a long armed window,
> and if it idles high, keep the LED disarmed until the last moment (as above).

On exit the LED is switched off even if the acquisition raises.

### Repeated pulses

To average many pulses, loop the acquisition — each `run()` emits one trigger, so one pulse
per acquisition, and each is saved and logged as usual:

```python
with DC2200() as led:
    led.configure_ttl(current_a=0.01, output=True)   # arm once, fire once per run()
    streams = []
    for i in range(50):
        ts = TimeStream(..., external_trigger=True, discard_start_ms=0,
                        notes=f"LED pulse {i + 1}/50")
        ts.attach(led=led)
        ts.run()
        streams.append(ts)

pulses = np.stack([s.signal[:, 0] for s in streams])
mean_pulse = pulses.mean(axis=0)
```

Do not use `fold_timestream` for this — it folds *one* record containing many drive periods,
which is the sawtooth-bias case. Here each record holds a single pulse, so average across
records.

### Free-running pulse trains

If you do not need the pulse tied to an acquisition, `pulse_train()` runs the DC2200's own PWM
burst and blocks until it finishes, with no trigger involved:

```python
with DC2200() as led:
    led.pulse_train(current_a=0.01, freq_hz=100, duty_pct=50, count=10)
```

**What that call does, exactly.** The four arguments describe the pulse train:

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

The DC2200's own PWM engine generates the pulses and stops itself after `count` of them, so
the *pulse* timing is instrument-accurate. Only the moment the train **starts** is
software-timed — a USB write, with millisecond-scale jitter.

> **This is not synchronised with anything.** Because the start is software-timed, the train
> cannot be placed reliably inside a `TimeStream` acquisition. To pulse the LED at a known
> point in a time stream, use `configure_ttl()` and gate it from the Presto trigger, as above.

Afterwards the instrument is left in PWM mode with these settings and the output off, so
`settings()` — and therefore `attach()` — reports `led_pwm_current_a`, `led_pwm_freq_hz`,
`led_pwm_duty_pct` and `led_pwm_count`.

`count` must be a whole number. Very large values are firmware-limited (100000 on firmware
1.3.0 and later).

### Both instruments at once

`attach()` takes any number of instruments, so a run with both LED and gate bias records both:

```python
with DC2200() as led, Agilent33220A() as bias:
    led.configure_ttl(current_a=0.01)
    bias.constant(0.98)
    ts = TimeStream(..., external_trigger=True, discard_start_ms=0)
    led.output = True                      # arm last, immediately before run()
    ts.attach(led=led, bias=bias)
    ts.run()
```

Note that the single Presto trigger line gates whatever is wired to it. Driving the LED and a
gated bias ramp from the same trigger fires them together; independent timing needs a second
trigger channel.

## SCPI gotchas

Hard-won, and all required verbatim.

| Instrument | Gotcha |
|---|---|
| DC2200 | Constant-current header is `SOURCE1:CCURENT:CURRENT` — Thorlabs' own misspelling, missing the second `R`. |
| DC2200 | PWM frequency is `SOURCE1:PWM:FREQ`; the long form `FREQUENCY` is rejected with `-113 Undefined header`. |
| DC2200 | Thorlabs documents TTL mode as having one settable parameter (the current at TTL high) but does not publish its SCPI header. `configure_ttl()` tries the candidates in `DC2200.TTL_CURRENT_HEADERS` and keeps whichever the instrument accepts, so it resolves on the first call instead of relying on a guess. If all are rejected it raises, naming what to check. |
| 33220A | Burst mode and a DC carrier are mutually exclusive. Writing `FUNC DC` while `BURS:STAT` is still `ON` from an earlier ramp is a settings conflict, so `constant()` disables burst **first**. Both mode setters write their full state for this reason. |
| 33220A | Minimum programmable amplitude is 20 mVpp into high-Z (`OUTP:LOAD INF`), 10 mVpp into 50 Ω. Below that the instrument does not output what you asked for; `sawtooth()` raises instead. |
| 33220A | `OUTP:LOAD INF` is the high-impedance setting (the Siglent spelling for the same thing is `LOAD,HZ`). |
| All | The error queue is drained before every write and re-read after, so a rejected command raises `InstrumentError` instead of being silently swallowed. Stale faults are never blamed on a later command. |

## Debugging

Pass `transcript_path=` to record every write, query and error response to a text file:

```python
with Agilent33220A(transcript_path="fgen_run.txt") as bias:
    ...
```

Common failures:

| Symptom | Likely fix |
|---|---|
| `Could not locate a VISA implementation` | Install NI-VISA, or set `DAQ_VISA_BACKEND=@py`. |
| `No VISA resources are visible` | Cable/power; check Device Manager or NI MAX; not WSL. |
| `Found N instruments identifying as ...` | Set `DAQ_FGEN_RESOURCE` / `DAQ_LED_RESOURCE`, or pass `resource=`. |
| Hang, or `VI_ERROR_RSRC_NFOUND` | Another process holds the instrument; shut down other kernels. Unplug/replug. |

## Adding an instrument

Subclass `VisaInstrument`, set `IDN_KEYWORDS` (and `RESOURCE_HINTS` if the resource string is
distinctive), override `env_resource()` to read its config setting, override `safe_state()` to
define "off", and extend `settings()` with read-back state. The base class handles the lazy
import, discovery, error checking, context-manager lifetime and transcript logging.
