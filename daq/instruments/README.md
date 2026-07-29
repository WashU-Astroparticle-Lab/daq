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

## SCPI gotchas

Hard-won, and all required verbatim.

| Instrument | Gotcha |
|---|---|
| DC2200 | Constant-current header is `SOURCE1:CCURENT:CURRENT` — Thorlabs' own misspelling, missing the second `R`. |
| DC2200 | PWM frequency is `SOURCE1:PWM:FREQ`; the long form `FREQUENCY` is rejected with `-113 Undefined header`. |
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
