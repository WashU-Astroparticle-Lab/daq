"""Offline verification of ``TimeStream.run()`` against a mocked Presto Lockin.

Checks the contracts introduced by the ``on_acquire`` hook (#51) and the exception-path
output muting that landed with it:

- call order is ``apply_settings`` < ``on_acquire`` < ``get_pixels``, with the trigger staged
  before the hook fires;
- a ``run()`` without the hook behaves exactly as before, and nothing callable is stored on
  the measurement object (so ``Base._save``'s ``__dict__`` walk cannot leak it);
- a raising hook aborts before acquisition, yet the DAC tones and trigger are muted before
  the connection closes;
- a ``get_pixels`` failure also mutes;
- a failure inside the mute itself does not mask the original exception;
- the mute adds nothing to the sync-critical hook-to-``get_pixels`` gap (it runs only after
  the record is complete).

Requires ``presto`` to be importable (for ``presto.utils.untwist_downconversion``); no
hardware and no network. Skips cleanly when ``presto`` is absent. Run from the repository
root::

    python tests/test_timestream_run.py
"""

import sys

try:
    import presto  # noqa: F401
except ImportError:
    print("SKIP: presto is not installed; TimeStream cannot be imported without it")
    sys.exit(0)

import numpy as np

import daq.measurements.timestream as tsm

events = []
results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


class FakeHW:
    def set_adc_attenuation(self, *a):
        pass

    def set_dac_current(self, *a):
        pass

    def set_inv_sinc(self, *a):
        pass

    def configure_mixer(self, *a, **k):
        pass


class FakeOG:
    def set_frequencies(self, *a):
        pass

    def set_amplitudes(self, amps):
        # Distinguish the mute (scalar 0.0) from the setup-time per-tone array.
        if np.isscalar(amps) and amps == 0.0:
            events.append("mute_amps")

    def set_phases(self, *a):
        pass


class FakeIG:
    def set_frequencies(self, *a):
        pass


class FakeLockin:
    def __init__(self, *a, **k):
        self.hardware = FakeHW()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        events.append("close")

    def set_dither(self, *a):
        pass

    def tune(self, f, df):
        return 0.0, df

    def set_df(self, df):
        pass

    def set_phase_reset(self, x):
        pass

    def add_output_group(self, *a):
        return FakeOG()

    def add_input_group(self, *a):
        return FakeIG()

    def set_trigger_out(self, states, **k):
        events.append(f"trigger:{states}")

    def apply_settings(self):
        events.append("apply_settings")

    def get_pixels(self, n, **k):
        events.append("get_pixels")
        rng = np.random.default_rng(0)
        iq = rng.standard_normal((n, 1)) + 1j * rng.standard_normal((n, 1))
        return {1: (np.zeros(1), iq, iq.copy())}


class FakeLockinModule:
    Lockin = FakeLockin


tsm.lockin = FakeLockinModule
# Bypass HDF5/MongoDB -- this test targets run() ordering, not the save path.
tsm.TimeStream.save = lambda self, save_filename=None: "/dev/null"


def make_ts():
    return tsm.TimeStream(
        lo_freq=6e9,
        if_freqs=[0],
        df=5e4,
        pixel_counts=1000,
        amp=0.01,
        output_port=1,
        input_port=1,
        device="offline-test",
        external_trigger=True,
        discard_start_ms=0,
    )


# --- 1. Ordering: apply_settings < on_acquire < get_pixels; mute after the record ---------
ts = make_ts()
ts.run(presto_address="offline", on_acquire=lambda: events.append("ON_ACQUIRE"))
order = [e for e in events if e in ("apply_settings", "ON_ACQUIRE", "get_pixels", "mute_amps")]
check(
    "hook runs after apply_settings, before get_pixels, mute after the record",
    order[:4] == ["apply_settings", "ON_ACQUIRE", "get_pixels", "mute_amps"],
    str(order[:4]),
)
check(
    "trigger staged before the hook fires", events.index("trigger:[1]") < events.index("ON_ACQUIRE")
)
check("data arrays populated", ts.signal is not None and ts.signal.shape == (1000, 1))

# --- 2. No hook: unchanged behaviour, nothing callable on the object ----------------------
events.clear()
ts2 = make_ts()
ts2.run(presto_address="offline")
check(
    "run() without the hook never invokes one",
    "ON_ACQUIRE" not in events and "get_pixels" in events,
)
check(
    "nothing callable stored on the measurement object",
    not any(callable(v) for v in ts2.__dict__.values()),
)

# --- 3. Raising hook: no acquisition, outputs muted, connection still closed --------------
events.clear()
try:
    ts.run(presto_address="offline", on_acquire=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    check("raising hook propagates", False)
except RuntimeError:
    muted = "mute_amps" in events and "trigger:[0]" in events
    check("raising hook aborts before get_pixels", "get_pixels" not in events)
    check("outputs muted on the hook-exception path", muted, str(events))
    check(
        "mute happens before the connection closes",
        muted and events.index("mute_amps") < events.index("close"),
    )

# --- 4. get_pixels failure also mutes ------------------------------------------------------
events.clear()


class FailingLockin(FakeLockin):
    def get_pixels(self, n, **k):
        events.append("get_pixels")
        raise TimeoutError("hw")


class FailingModule:
    Lockin = FailingLockin


tsm.lockin = FailingModule
try:
    make_ts().run(presto_address="offline")
    check("get_pixels failure propagates", False)
except TimeoutError:
    check(
        "get_pixels failure also mutes outputs", "mute_amps" in events and "trigger:[0]" in events
    )

# --- 5. A mute failure must not mask the original exception -------------------------------
events.clear()


class DoubleFailLockin(FakeLockin):
    def get_pixels(self, n, **k):
        raise TimeoutError("original error")

    def set_trigger_out(self, states, **k):
        if states == [0]:
            raise OSError("mute failed")
        events.append(f"trigger:{states}")


class DoubleFailModule:
    Lockin = DoubleFailLockin


tsm.lockin = DoubleFailModule
try:
    make_ts().run(presto_address="offline")
    check("original exception propagates", False)
except TimeoutError as exc:
    check("mute failure does not mask the original exception", "original error" in str(exc))
except OSError:
    check("mute failure does not mask the original exception", False, "OSError masked it")

# ------------------------------------------------------------------------------ summary
failed = [label for label, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label in failed:
        print("  FAILED:", label)
    sys.exit(1)
