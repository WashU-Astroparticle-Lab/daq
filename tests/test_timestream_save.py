"""Offline verification of what ``TimeStream`` writes to HDF5 and reads back.

A stream holds five per-sample arrays that are one acquisition stored five ways (the two
demodulators, the two sidebands they combine into, and the per-tone selected sideband). Before
``save_arrays`` existed every file carried all five as complex128: at 100 kHz and 13 tones that
is 1 GB per 10 s, ten times what any analysis in the repo reads. This suite checks that::

    python tests/test_timestream_save.py

- the default file keeps ``signal`` alone, as complex64, and says so in its attributes;
- ``"pixels"`` keeps the demodulator pair and ``load()`` rebuilds the sidebands and ``signal``
  from it exactly as ``run()`` does;
- ``"all"`` with complex128 reproduces the historical file bit for bit;
- the in-memory object is untouched by either setting -- slimming the file must not slim the
  analysis that follows ``run()``;
- a file written before the attributes existed loads as ``"all"``/complex128;
- the size really drops by an order of magnitude, since that is the point.

Needs no hardware, no VISA runtime and no MongoDB; ``presto`` is stubbed when absent, because
nothing under test here touches it (``save()``/``load()`` are pure HDF5). The stub's
``untwist_downconversion`` is presto 2.17.1's own formula, so the ``"pixels"`` round trip is
checked against the real combination rather than a placeholder.
"""

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

# save() asks MongoDB for a measurement number and falls back to a timestamp when it cannot.
# Point it somewhere that refuses immediately rather than waiting out a 30 s selection timeout.
os.environ.setdefault("DAQ_MONGODB_URI", "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50")

# Test the checkout this file lives in, not whatever ``pip install -e`` points at.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _untwist(i_port, q_port):
    """presto.utils.untwist_downconversion, 2.17.1 -- (L_sideband, H_sideband)."""
    low = np.zeros_like(i_port)
    high = np.zeros_like(q_port)
    low.real += i_port.real + q_port.imag
    low.imag += q_port.real - i_port.imag
    high.real += i_port.real - q_port.imag
    high.imag += q_port.real + i_port.imag
    return low, high


try:  # pragma: no cover - depends on the machine, not on the code under test
    import presto  # noqa: F401
except ImportError:

    class _Enum:
        Mixed = "Mixed"

    _presto = types.ModuleType("presto")
    _lockin = types.ModuleType("presto.lockin")
    _lockin.Lockin = object
    _utils = types.ModuleType("presto.utils")
    _utils.untwist_downconversion = _untwist
    _utils.get_sourcecode = lambda path: [""]
    _utils.recommended_dac_config = lambda freq: {}
    _utils.asarray = lambda x: x
    _utils.rotate_opt = lambda x: x

    class _ProgressBar:
        def __init__(self, *a, **k):
            pass

        def increment(self):
            pass

        def done(self):
            pass

    _utils.ProgressBar = _ProgressBar
    _hardware = types.ModuleType("presto.hardware")
    _hardware.AdcMode = _Enum
    _hardware.DacMode = _Enum
    for _name, _module in (("lockin", _lockin), ("utils", _utils), ("hardware", _hardware)):
        setattr(_presto, _name, _module)
        sys.modules[f"presto.{_name}"] = _module
    sys.modules["presto"] = _presto
    print("INFO: presto is not installed; using a stub (save()/load() do not touch it)")

import h5py  # noqa: E402

from daq.measurements.timestream import SAVE_ARRAYS, TIME_AXIS_ARRAYS, TimeStream  # noqa: E402

results = []


def check(label, condition, detail=""):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


N_SAMPLES, N_TONES, DF = 20_000, 2, 1_000.0
DISCARD_MS = 25.0  # 25 samples at 1 kHz, so the trim is visible after load()


def make_stream(**kwargs):
    """A two-tone stream whose arrays look like run() left them, without any hardware."""
    ts = TimeStream(
        lo_freq=5e9,
        if_freqs=[1e6, 2e6],
        is_usb=[True, False],
        df=DF,
        pixel_counts=N_SAMPLES,
        amp=[0.1, 0.1],
        output_port=1,
        input_port=1,
        device="fake_device",
        discard_start_ms=DISCARD_MS,
        **kwargs,
    )
    rng = np.random.default_rng(20260917)
    shape = (N_SAMPLES, N_TONES)
    ts.freq_arr = np.array([1e6, 2e6])
    ts.pixel_i = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 1e-3
    ts.pixel_q = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 1e-3
    ts.lsb, ts.usb = _untwist(ts.pixel_i, ts.pixel_q)
    ts.freqs_usb = ts.lo_freq + ts.if_freqs
    ts.freqs_lsb = ts.lo_freq - ts.if_freqs
    ts.signal = np.where(ts.is_usb[np.newaxis, :], ts.usb, ts.lsb)
    ts.signal_freqs = np.where(ts.is_usb, ts.freqs_usb, ts.freqs_lsb)
    return ts


def datasets(path):
    with h5py.File(path, "r") as h5f:
        return {name: (h5f[name].dtype.name, h5f[name].shape) for name in h5f}


def attrs(path):
    with h5py.File(path, "r") as h5f:
        return {k: v.decode() if isinstance(v, bytes) else v for k, v in h5f.attrs.items()}


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # ---- 1. the default file: signal alone, complex64, declared ---------------------------
    ts = make_stream()
    check("default save_arrays is 'signal'", ts.save_arrays == "signal", ts.save_arrays)
    check("default save_dtype is complex64", ts.save_dtype == "complex64", ts.save_dtype)
    default_path = tmp / "default.h5"
    ts.save(save_filename=str(default_path))
    ds = datasets(default_path)
    present = [name for name in TIME_AXIS_ARRAYS if name in ds]
    check("default file holds signal only", present == ["signal"], str(present))
    check("signal is complex64 on disk", ds["signal"][0] == "complex64", ds["signal"][0])
    check("signal keeps its full, untrimmed shape", ds["signal"][1] == (N_SAMPLES, N_TONES))
    a = attrs(default_path)
    check(
        "file attributes declare the storage",
        a.get("save_arrays") == "signal" and a.get("save_dtype") == "complex64",
        f"{a.get('save_arrays')!r}, {a.get('save_dtype')!r}",
    )
    with h5py.File(default_path, "r") as h5f:
        on_disk = h5f["signal"][()]
    check(
        "on-disk signal is the complex64 cast of the in-memory one",
        np.array_equal(on_disk, ts.signal.astype(np.complex64)),
    )
    check(
        "in-memory arrays are untouched: all five present, complex128",
        all(
            getattr(ts, n) is not None and getattr(ts, n).dtype == np.complex128
            for n in TIME_AXIS_ARRAYS
        ),
    )
    # The small arrays are still there -- only the per-sample ones are subject to the choice.
    check(
        "frequency-axis arrays are always saved",
        all(n in ds for n in ("freq_arr", "signal_freqs", "freqs_usb", "freqs_lsb", "if_freqs")),
        str(sorted(ds)),
    )

    # ---- 2. loading the default file ---------------------------------------------------
    loaded = TimeStream.load(str(default_path))
    n_trim = int(round(DISCARD_MS * 1e-3 * DF))
    check(
        "load() trims the leading discard from signal",
        loaded.signal.shape == (N_SAMPLES - n_trim, N_TONES),
        str(loaded.signal.shape),
    )
    check(
        "loaded signal matches to complex64 precision",
        np.allclose(loaded.signal, ts.signal[n_trim:], rtol=1e-6, atol=0),
    )
    check(
        "arrays not in the file load as None",
        all(getattr(loaded, n) is None for n in ("pixel_i", "pixel_q", "lsb", "usb")),
    )
    check(
        "load() restores the storage settings",
        loaded.save_arrays == "signal" and loaded.save_dtype == "complex64",
    )
    check("signal_freqs survive", np.array_equal(loaded.signal_freqs, ts.signal_freqs))
    # Re-saving a loaded slim file must not trip over the arrays it never had.
    resaved = tmp / "resaved.h5"
    loaded.save(save_filename=str(resaved))
    check(
        "a loaded slim file re-saves cleanly",
        [n for n in TIME_AXIS_ARRAYS if n in datasets(resaved)] == ["signal"],
    )

    # ---- 3. "pixels": the demodulator pair, sidebands rebuilt on load ------------------
    ts_px = make_stream(save_arrays="pixels")
    px_path = tmp / "pixels.h5"
    ts_px.save(save_filename=str(px_path))
    ds = datasets(px_path)
    present = [n for n in TIME_AXIS_ARRAYS if n in ds]
    check(
        "'pixels' file holds pixel_i and pixel_q only",
        present == ["pixel_i", "pixel_q"],
        str(present),
    )
    check(
        "pixels are complex64 on disk",
        ds["pixel_i"][0] == "complex64" and ds["pixel_q"][0] == "complex64",
    )
    loaded_px = TimeStream.load(str(px_path))
    # The pair is stored as complex64, so the rebuilt sidebands carry its rounding: about
    # 1e-7 of the sample scale, which is a large *relative* error only where a sum of two
    # demodulator components happens to cancel. Compare against the scale, not the element.
    tol = 1e-6 * np.abs(ts_px.usb).max()
    check(
        "load() rebuilds lsb/usb from the pixel pair",
        loaded_px.lsb is not None
        and loaded_px.usb is not None
        and np.allclose(loaded_px.usb, ts_px.usb[n_trim:], rtol=0, atol=tol)
        and np.allclose(loaded_px.lsb, ts_px.lsb[n_trim:], rtol=0, atol=tol),
    )
    check(
        "load() rebuilds signal with the per-tone sideband choice",
        loaded_px.signal is not None
        and np.allclose(loaded_px.signal, ts_px.signal[n_trim:], rtol=0, atol=tol),
    )

    # ---- 4. "all" + complex128: the historical file, bit for bit -----------------------
    ts_all = make_stream(save_arrays="all", save_dtype="complex128")
    all_path = tmp / "all.h5"
    ts_all.save(save_filename=str(all_path))
    ds = datasets(all_path)
    check(
        "'all' file holds every time-axis array",
        all(n in ds for n in TIME_AXIS_ARRAYS),
        str(sorted(ds)),
    )
    check("complex128 is honoured", all(ds[n][0] == "complex128" for n in TIME_AXIS_ARRAYS))
    with h5py.File(all_path, "r") as h5f:
        exact = all(np.array_equal(h5f[n][()], getattr(ts_all, n)) for n in TIME_AXIS_ARRAYS)
    check("complex128 arrays are stored exactly", exact)

    # ---- 5. a file from before the attributes existed --------------------------------
    with h5py.File(all_path, "a") as h5f:
        del h5f.attrs["save_arrays"]
        del h5f.attrs["save_dtype"]
    legacy = TimeStream.load(str(all_path))
    check(
        "a pre-save_arrays file loads as 'all'/complex128",
        legacy.save_arrays == "all" and legacy.save_dtype == "complex128",
        f"{legacy.save_arrays!r}, {legacy.save_dtype!r}",
    )
    check(
        "...with every array present", all(getattr(legacy, n) is not None for n in TIME_AXIS_ARRAYS)
    )

    # ---- 6. size: the reason this exists -------------------------------------------------
    ratio = default_path.stat().st_size / all_path.stat().st_size
    check("default file is under a fifth of the historical one", ratio < 0.2, f"ratio {ratio:.3f}")

    # ---- 7. validation ------------------------------------------------------------------
    for bad in ({"save_arrays": "sideband"}, {"save_dtype": "float32"}):
        try:
            make_stream(**bad)
            check(f"{bad} is refused", False)
        except ValueError:
            check(f"{bad} is refused", True)
    check(
        "SAVE_ARRAYS names only real arrays",
        all(set(v) <= set(TIME_AXIS_ARRAYS) for v in SAVE_ARRAYS.values()),
    )

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
