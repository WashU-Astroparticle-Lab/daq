"""Offline verification of single-tone and simultaneous multitone ``StdDevSweep``.

The sweep composes ``QCTrace`` and inherits its readout, ramp and trigger handling, so these
checks swap ``TimeStream`` in the shared readout module for a synthetic response and drive the
real ``QCTrace`` folding, the real save/load round trip and the plot:

- the ranked statistic and the diagnostic curves come out of the synthetic traces as expected,
  on the *tuned* sample rate, and the winner is the largest;
- the principal-axis statistic is invariant under a rotation and offset of the I/Q plane,
  where ``std(I)`` alone is not;
- the raw-stream and fixed-projection statistics, and their edge cases;
- every point is gated on the generator's own port (or an explicit override), the bias is
  de-energised on success and on failure, and a failed re-run leaves no stale winner;
- the record round-trips through HDF5, the MongoDB document carries the frequency axis and a
  calibrated power per point, and a loaded sweep re-reads the generator when re-run;
- invalid configurations are refused before any hardware is touched.

The single-tone ``TimeStream`` swap targets ``daq.measurements._gate_bias``; the multitone
swap targets ``daq.measurements.sweep_std_dev``. Additional checks cover independent winners,
sidebands, per-tone amplitudes, shared raw records, stored QC tone indices and 13-device scans.

Requires ``presto`` to be importable (``StdDevSweep`` imports it transitively); no hardware and
no network. The database calls are stubbed and the data folder pointed at a temporary directory,
so the real save path runs without MongoDB. Run from the repository root::

    python tests/test_sweep_std_dev.py
"""

import itertools
import os
import sys
import tempfile
import warnings
from types import SimpleNamespace

# Run as a script, tests/ is what lands on sys.path; put the checkout being edited first, so an
# editable install of another checkout is not what gets exercised.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import presto  # noqa: F401
except ImportError:
    print("SKIP: presto is not installed; StdDevSweep cannot be imported without it")
    sys.exit(0)

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import daq._base as base_mod
import daq.measurements._gate_bias as gate_bias_mod
import daq.measurements.sweep_std_dev as sweep_mod
from daq import StdDevSweep

results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- stand-ins

FREQS = [2.7e9, 2.8e9, 2.9e9]
RAMP_HZ = 500.0
FS = 5000.0
TUNED_FS = 6000.0  # deliberately different from the requested rate
N_PERIODS = 10
DISCARD_MS = 2.0
#: (I, Q) scale of the synthetic response at each frequency. The response is a straight line
#: in the I/Q plane, so its principal-axis spread is sqrt(I^2 + Q^2) times the spread of the
#: underlying ramp, and the winner is the middle frequency.
SCALES = {2.7e9: (1, 2), 2.8e9: (3, 1), 2.9e9: (2, 1)}

tmp = tempfile.TemporaryDirectory()
documents = []
numbers = itertools.count(1)

# Keep the real save path, but off the network and in a scratch folder.
base_mod.get_next_number = lambda: f"{next(numbers):08d}"
base_mod.insert_measurement = lambda document: documents.append(document) or "offline"
base_mod.get_data_folder = lambda: tmp.name


class FakeTimeStream:
    """Produces a ramp-shaped response on the tuned rate; records what it was asked for."""

    instances = []
    fail_at = None
    fail_on_acquisition = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.df = TUNED_FS
        self.signal = None
        FakeTimeStream.instances.append(self)

    def attach(self, **instruments):
        self.attached = instruments

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        if (
            self.kwargs["lo_freq"] == FakeTimeStream.fail_at
            or len(FakeTimeStream.instances) == FakeTimeStream.fail_on_acquisition
        ):
            raise RuntimeError("simulated acquisition failure")
        n = self.kwargs["pixel_counts"] - round(self.kwargs["discard_start_ms"] * 1e-3 * self.df)
        per = int(round(self.df / RAMP_HZ))
        k = np.arange(n)
        # A repeating ramp, plus block-to-block offsets that cancel under folding (an even
        # number of whole blocks fits the record).
        wave = (k % per) / per
        offsets = np.where((k // per) % 2, -0.2, 0.2)
        signs = np.where(self.kwargs.get("is_usb", [True]), 1, -1)
        self.signal_freqs = self.kwargs["lo_freq"] + signs * np.asarray(self.kwargs["if_freqs"])
        self.signal = np.column_stack(
            [0.01 * complex(*SCALES[freq]) * wave + offsets for freq in self.signal_freqs]
        )
        path = os.path.join(tmp.name, f"raw-{len(FakeTimeStream.instances)}.h5")
        with h5py.File(path, "w") as h5f:
            h5f["signal"] = self.signal
        return path


class FakeBias:
    """A 33220A stand-in: knows which port it is wired to, records what it was told."""

    def __init__(self, trigger_port=2):
        self.trigger_port = trigger_port
        self.output = False
        self.closed = False
        self.ramps = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.output = False
        self.closed = True

    def sawtooth(self, **kwargs):
        self.ramps.append(kwargs)
        self.output = True

    def samples_for_periods(self, n_periods, sample_rate, *, freq_hz, discard_ms):
        return round(n_periods * sample_rate / freq_hz) + round(discard_ms * 1e-3 * sample_rate)


gate_bias_mod.TimeStream = FakeTimeStream


def make(**kwargs):
    params = dict(
        readout_freqs=FREQS,
        amp=0.01,
        output_port=4,
        input_port=1,
        ramp_vpp=1.2,
        ramp_freq_hz=RAMP_HZ,
        sampling_frequency=FS,
        num_periods=N_PERIODS,
        discard_start_ms=DISCARD_MS,
        device="offline-QPD",
        filter="test-chain",
        notes="frequency optimization",
    )
    params.update(kwargs)
    return StdDevSweep(**params)


def run(sweep, bias=None, **kwargs):
    FakeTimeStream.instances = []
    with warnings.catch_warnings():
        # The 0.01 FS drive sits below the calibration's verified floor; not what is under test.
        warnings.simplefilter("ignore")
        return sweep.run(bias, **kwargs)


def trace_of(z):
    """A folded-trace stand-in holding the complex samples *z*."""
    return SimpleNamespace(avg_iq=np.vstack([z.real, z.imag]))


PER = int(round(TUNED_FS / RAMP_HZ))
SIGMA = np.std(np.arange(PER) / PER) * 0.01

# ---------------------------------------------------------------- folded acquisition

sweep = make()
bias = FakeBias()
path = run(sweep, bias, presto_address="test-presto", presto_port=123, ext_ref_clk=True)

check("inherits the shared gate-bias readout", isinstance(sweep, gate_bias_mod.GateBiasMeasurement))
check(
    "a sweep carries no single readout_freq",
    not hasattr(sweep, "readout_freq"),
)
check(
    "std(I) per frequency follows the synthetic response",
    np.allclose(sweep.std_i_arr, SIGMA * np.array([1, 3, 2])),
    str(sweep.std_i_arr),
)
check(
    "std(Q) per frequency follows the synthetic response",
    np.allclose(sweep.std_q_arr, SIGMA * np.array([2, 1, 1])),
)
check(
    "principal-axis std is the line's full length",
    np.allclose(sweep.std_arr, SIGMA * np.sqrt([5, 10, 5]))
    and np.array_equal(sweep.std_arr, sweep.std_principal_arr),
    str(sweep.std_arr),
)
check(
    "the winner is the largest spread",
    sweep.best_freq == 2.8e9 and sweep.best_std == sweep.std_arr[1],
)
check("best_qc_file is the winner's QC record", sweep.best_qc_file == sweep.qc_files[1])
check(
    "the fold ran on the tuned rate, one period of samples",
    np.array_equal(sweep.sample_counts, [PER] * 3)
    and np.array_equal(sweep.sampling_frequencies, [TUNED_FS] * 3),
    f"{sweep.sample_counts} {sweep.sampling_frequencies}",
)
check("a caller-owned bias is de-energised but not closed", not bias.output and not bias.closed)
check("one gated ramp per frequency", len(bias.ramps) == 3 and all(r["gated"] for r in bias.ramps))
streams = FakeTimeStream.instances
check(
    "each point reads out at its own frequency with the shared readout",
    [s.kwargs["lo_freq"] for s in streams] == FREQS
    and all(s.kwargs["amp"] == 0.01 and s.kwargs["output_port"] == 4 for s in streams),
)
check(
    "each point spans whole ramp periods plus the discarded start",
    all(
        s.kwargs["pixel_counts"] == 110 and s.kwargs["discard_start_ms"] == DISCARD_MS
        for s in streams
    ),
)
check(
    "each point is gated on the generator's own port",
    all(np.array_equal(s.kwargs["external_trigger"], [0, 1]) for s in streams)
    and np.array_equal(sweep.trigger_states, [0, 1]),
)
check(
    "Presto connection parameters reach every point",
    all(
        s.run_kwargs == dict(presto_address="test-presto", presto_port=123, ext_ref_clk=True)
        for s in streams
    ),
)
check(
    "each point's QC trace note carries the sweep's note",
    all(
        s.kwargs["notes"].startswith("frequency optimization -- Std dev sweep point")
        for s in streams
    ),
    streams[0].kwargs["notes"],
)
check(
    "constituent files exist",
    all(os.path.exists(p) for p in sweep.qc_files + sweep.raw_files) and len(sweep.raw_files) == 3,
)

# ---------------------------------------------------------------- record and round trip

doc = documents[-1]
check("the summary is its own measurement type", doc["type"] == "sweep_std_dev", doc["type"])
check(
    "the document carries the frequency axis beside the curve",
    doc.get("readout_freqs") == FREQS and doc.get("std_arr") == sweep.std_arr.tolist(),
)
check(
    "the document carries a calibrated power per point, and no scalar power",
    isinstance(doc.get("power_dbm_arr"), list)
    and len(doc["power_dbm_arr"]) == 3
    and "power_dbm" not in doc,
    str(doc.get("power_dbm_arr")),
)
check(
    "the document names the winner and the files",
    doc["best_freq"] == 2.8e9 and doc["qc_files"] == sweep.qc_files,
)

restored = StdDevSweep.load(path)
arrays_match = all(
    np.array_equal(getattr(restored, name), getattr(sweep, name))
    for name in (
        "readout_freqs",
        "std_arr",
        "std_i_arr",
        "std_q_arr",
        "std_principal_arr",
        "principal_axes",
        "sample_counts",
        "sampling_frequencies",
        "trigger_states",
    )
)
scalars_match = all(
    getattr(restored, name) == getattr(sweep, name)
    for name in (
        "best_freq",
        "best_std",
        "best_qc_file",
        "qc_files",
        "raw_files",
        "device",
        "filter",
        "notes",
        "quantity",
        "trace_source",
        "ddof",
        "ramp_vpp",
        "num_periods",
    )
)
check("the record round-trips through HDF5", arrays_match and scalars_match)
check("loaded paths are str, not bytes", all(isinstance(p, str) for p in restored.qc_files))

fig = restored.analyze()
ax = fig.axes[0]
check(
    "analyze() draws I, Q and principal-axis curves and marks the maximum",
    np.array_equal(ax.lines[0].get_ydata(), restored.std_i_arr)
    and np.array_equal(ax.lines[1].get_ydata(), restored.std_q_arr)
    and np.array_equal(ax.lines[2].get_ydata(), restored.std_principal_arr)
    and "Maximum" in ax.lines[3].get_label()
    and "GHz" in ax.get_xlabel(),
)
plt.close("all")

run(restored, FakeBias(trigger_port=1))
check(
    "re-running a loaded sweep re-reads the generator's port",
    np.array_equal(restored.trigger_states, [1])
    and all(np.array_equal(s.kwargs["external_trigger"], [1]) for s in FakeTimeStream.instances),
)

# ---------------------------------------------------------------- other statistics

raw = make(trace_source="raw", quantity="complex", ddof=1)
run(raw, FakeBias())
expected = [np.std(s.signal[:, 0], ddof=1) for s in FakeTimeStream.instances]
check(
    "the raw statistic is taken over the trimmed stream",
    np.allclose(raw.std_arr, expected) and np.array_equal(raw.sample_counts, [98] * 3),
)

square = np.array([1, 1j, -1, -1j])
projections_ok = all(
    np.isclose(make(quantity=q).standard_deviation(trace_of(square)), v)
    for q, v in (
        ("principal", np.sqrt(0.5)),
        ("complex", 1.0),
        ("abs", 0.0),
        ("real", np.sqrt(0.5)),
        ("imag", np.sqrt(0.5)),
    )
)
check("each projection gives its own spread of the unit square", projections_ok)
check(
    "the combined I/Q spread ignores rotation and offset",
    np.isclose(
        make(quantity="complex").standard_deviation(trace_of(square * np.exp(0.73j) + 2 + 3j)), 1.0
    ),
)

# ---------------------------------------------------------------- principal-axis invariance

phase = np.linspace(0, 2 * np.pi, 120, endpoint=False)
pca = make(ddof=1)
original, rotated, along_i, aligned = [], [], [], []
for major, minor, angle in ((1, 0.3, 0.1), (3, 0.2, 1.4), (2, 0.5, -0.6)):
    z = major * np.cos(phase) + 1j * minor * np.sin(phase)
    original.append(pca.standard_deviation(trace_of(z)))
    z = z * np.exp(1j * angle) + 7 - 13j
    rotated.append(pca.standard_deviation(trace_of(z)))
    along_i.append(make(quantity="real").standard_deviation(trace_of(z)))
    axis = pca._principal_axis(z)
    aligned.append(
        np.isclose(np.linalg.norm(axis), 1)
        and np.isclose(abs(axis @ [np.cos(angle), np.sin(angle)]), 1)
    )
check(
    "principal-axis std is invariant under rotation and offset",
    np.allclose(original, rotated, rtol=1e-12),
)
check(
    "principal-axis std is the ellipse's semi-major spread",
    np.allclose(rotated, np.array([1, 3, 2]) * np.sqrt(60 / 119)),
)
check("the fitted axis is the rotated major axis", all(aligned))
check(
    "std(I) alone would rank the ellipses differently",
    int(np.argmax(rotated)) == 1 and int(np.argmax(along_i)) != 1,
    str(along_i),
)
check(
    "a flat trace has zero spread, offset or not",
    pca.standard_deviation(trace_of(np.full(12, 5 + 5j))) == 0,
)

# ---------------------------------------------------------------- ties, zero curve, one point

tie = make(readout_freqs=[2.9e9, 2.7e9], quantity="complex")
run(tie, FakeBias())
tie.std_arr[:] = 1.0
tie._select_best()
check("an exact tie names the first frequency acquired", tie.best_freq == 2.9e9)
tie.std_arr[:] = 0.0
tie._select_best()
check(
    "an all-zero curve still names a frequency, with zero spread",
    tie.best_freq == 2.9e9 and tie.best_std == 0 and tie.best_qc_file == tie.qc_files[0],
)
zero_path = tie.save(os.path.join(tmp.name, "zero.h5"))
with h5py.File(zero_path, "r") as h5f:
    check(
        "an all-zero curve saves its winner rather than dropping it",
        h5f.attrs["best_freq"] == 2.9e9 and h5f.attrs["best_std"] == 0,
    )
check(
    "an all-zero curve is flagged on the plot",
    any("not meaningful" in t.get_text() for t in tie.analyze().axes[0].texts),
)
plt.close("all")
single = make(readout_freqs=[2.8e9])
run(single, FakeBias())
check("a single frequency is its own winner", single.best_freq == 2.8e9)

# ---------------------------------------------------------------- failure and cleanup

sweep = make()
bias = FakeBias()
run(sweep, bias)
old_files = list(sweep.qc_files)
FakeTimeStream.fail_at = 2.8e9
try:
    run(sweep, bias)
    failed = False
except RuntimeError as err:
    failed = "simulated acquisition" in str(err)
FakeTimeStream.fail_at = None
check("a failing point aborts the sweep", failed)
check(
    "a failed re-run leaves no stale winner or curve",
    sweep.best_freq is None and sweep.best_std is None and sweep.std_arr is None,
)
check("a failed run still de-energises the bias", not bias.output)
completed = [p for p in os.listdir(tmp.name) if p.endswith("-qc_trace.h5")]
check(
    "the completed point's files remain on disk",
    len(completed) > len(old_files)
    and not any(os.path.basename(p) in old_files for p in completed[-1:]),
)
for label, call in (("analyze", sweep.analyze), ("save", sweep.save)):
    try:
        call()
        refused = False
    except RuntimeError:
        refused = True
    check(f"{label}() refuses an incomplete sweep", refused)

owned = FakeBias()
sweep_mod.Agilent33220A = lambda: owned
run(make(trigger_states=[0, 1, 1]))
check(
    "an explicit routing gates every point",
    all(np.array_equal(s.kwargs["external_trigger"], [0, 1, 1]) for s in FakeTimeStream.instances),
)
check("a self-opened bias session is closed and de-energised", owned.closed and not owned.output)
failed_bias = FakeBias()
sweep_mod.Agilent33220A = lambda: failed_bias
FakeTimeStream.fail_at = 2.7e9
try:
    run(make())
except RuntimeError:
    pass
FakeTimeStream.fail_at = None
check(
    "a self-opened session is closed on failure too", failed_bias.closed and not failed_bias.output
)

# ---------------------------------------------------------------- earlier draft's names

old_path = os.path.join(tmp.name, "old-draft.h5")
with h5py.File(path, "r") as src, h5py.File(old_path, "w") as dst:
    for name in src:
        if name != "readout_freqs":
            src.copy(name, dst)
    dst["freq_arr"] = src["readout_freqs"][()]
    for key, value in src.attrs.items():
        dst.attrs[key] = value
check(
    "a file from the earlier draft, with freq_arr, still loads",
    np.array_equal(StdDevSweep.load(old_path).readout_freqs, FREQS),
)

# ---------------------------------------------------------------- refused configurations

refused = []
for kwargs in (
    dict(readout_freqs=[]),
    dict(readout_freqs=2.8e9),
    dict(readout_freqs=[[[2.8e9]]]),
    dict(readout_freqs=[np.nan]),
    dict(readout_freqs=[np.inf]),
    dict(readout_freqs=[-1.0]),
    dict(amp=np.nan),
    dict(amp=1.5),
    dict(ramp_vpp=np.inf),
    dict(ramp_freq_hz=np.inf),
    dict(ramp_offset_v=np.nan),
    dict(sampling_frequency=0),
    dict(discard_start_ms=-1),
    dict(num_periods=0),
    dict(ddof=-1),
    dict(ddof=0.5),
    dict(ddof=True),
    dict(ddof=PER * 10),  # a folded period at the requested rate holds 10 samples
    dict(trigger_states=False),
    dict(trigger_states=[0, 0]),
    dict(quantity="invalid"),
    dict(trace_source="invalid"),
):
    try:
        make(**kwargs)
        refused.append(str(kwargs))
    except ValueError:
        pass
check("invalid configurations raise ValueError in __init__", not refused, "; ".join(refused))

try:
    run(make(device=None), FakeBias())
    refused_device = False
except ValueError:
    refused_device = not FakeTimeStream.instances
check("a missing device is refused before any acquisition", refused_device)

with warnings.catch_warnings(record=True) as drift:
    warnings.simplefilter("always")
    make(ramp_freq_hz=300.0)
check(
    "a non-integral samples-per-period ratio warns, as for QCTrace",
    any("not a whole multiple" in str(w.message) for w in drift),
)

bad_inputs = []
for iq in (np.zeros((2, 0)), np.zeros((2, 1)), np.zeros((3, 4)), np.array([[0, np.nan], [1, 2]])):
    try:
        make().standard_deviation(SimpleNamespace(avg_iq=iq))
        bad_inputs.append(str(iq.shape))
    except ValueError:
        pass
check("malformed folded traces raise ValueError", not bad_inputs, "; ".join(bad_inputs))
missing = 0
for sweep_obj, trace in (
    (make(), SimpleNamespace(avg_iq=None)),
    (make(trace_source="raw"), SimpleNamespace(qc_stream=None)),
):
    try:
        sweep_obj.standard_deviation(trace)
    except RuntimeError:
        missing += 1
check("a trace without the chosen record raises RuntimeError", missing == 2)

# ---------------------------------------------------------------- simultaneous tones

sweep_mod.TimeStream = FakeTimeStream
SCALES.update({3.1e9: (4, 1), 3.2e9: (1, 1), 3.3e9: (1, 5)})
grid = np.column_stack([FREQS, np.asarray(FREQS) + 400e6])
multi = make(readout_freqs=grid, amp=[0.01, 0.02], tone_labels=["lower", "upper"])
bias = FakeBias()
multi_path = run(multi, bias, presto_address="test-presto", ext_ref_clk=True)
expected = SIGMA * np.sqrt([[5, 17], [10, 2], [5, 26]])
check("multitone uses one acquisition per row, not per device", len(FakeTimeStream.instances) == 3)
check(
    "multitone principal curves rank each device independently",
    np.allclose(multi.std_arr, expected) and np.array_equal(multi.best_freq, [2.8e9, 3.3e9]),
)
check(
    "multitone diagnostic shapes follow the frequency matrix",
    multi.principal_axes.shape == (3, 2, 2) and multi.sample_counts.shape == (3, 2),
)
check(
    "multitone uses the tuned sample rate for each fold",
    np.all(multi.sample_counts == PER) and np.all(multi.sampling_frequencies == TUNED_FS),
)
streams = FakeTimeStream.instances
check(
    "multitone LO and signed sidebands reconstruct every physical frequency",
    all(np.array_equal(stream.signal_freqs, row) for stream, row in zip(streams, grid))
    and all(np.array_equal(stream.kwargs["is_usb"], [False, True]) for stream in streams),
)
check(
    "per-tone drive is not split or rescaled",
    all(np.array_equal(stream.kwargs["amp"], [0.01, 0.02]) for stream in streams),
)
check(
    "multitone forwards trigger routing, timing and connection parameters",
    all(
        np.array_equal(stream.kwargs["external_trigger"], [0, 1])
        and stream.kwargs["pixel_counts"] == 110
        and stream.kwargs["discard_start_ms"] == DISCARD_MS
        and stream.run_kwargs["presto_address"] == "test-presto"
        and stream.run_kwargs["ext_ref_clk"]
        for stream in streams
    ),
)
check("multitone bias output is off after acquisition", not bias.output and not bias.closed)
check(
    "multitone metadata preserves frequency, curve and power matrix shapes",
    documents[-1]["readout_freqs"] == grid.tolist()
    and np.shape(documents[-1]["power_dbm_arr"]) == grid.shape
    and documents[-1]["tone_labels"] == ["lower", "upper"],
)
from daq.calibrations import amp_to_power_dbm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    expected_power = [
        [float(amp_to_power_dbm(f / 1e9, a)) for f, a in zip(row, multi.amp)] for row in grid
    ]
check(
    "power calibration pairs each frequency with its own tone amplitude",
    np.allclose(documents[-1]["power_dbm_arr"], expected_power),
)
loaded = StdDevSweep.load(multi_path)
check(
    "multitone HDF5 restores matrices, labels, amplitudes, LO and independent winners",
    np.array_equal(loaded.std_arr, multi.std_arr)
    and np.array_equal(loaded.principal_axes, multi.principal_axes)
    and np.array_equal(loaded.best_freq, multi.best_freq)
    and np.array_equal(loaded.amp, multi.amp)
    and loaded.lo_freq == multi.lo_freq
    and loaded.tone_labels == multi.tone_labels
    and loaded.qc_files == multi.qc_files
    and loaded.raw_files == multi.raw_files
    and loaded.best_qc_file == multi.best_qc_file,
)
from daq import QCTrace

winning_trace = QCTrace.load(loaded.best_qc_file[1])
check(
    "winning QC record names its shared raw file and tone",
    winning_trace.tone == 1
    and winning_trace.qc_file == multi.raw_files[2]
    and winning_trace.readout_freq == 3.3e9,
)
expected_iq = winning_trace.avg_iq.copy()
winning_trace.fold(streams[2])
check(
    "loaded QC trace re-folds its stored tone by default",
    np.allclose(winning_trace.avg_iq, expected_iq),
)
check(
    "each row saves one shared raw stream and one QC record per tone",
    len(multi.raw_files) == 3
    and np.shape(multi.qc_files) == (3, 2)
    and all(
        QCTrace.load(path).qc_file == multi.raw_files[row]
        for row, paths in enumerate(multi.qc_files)
        for path in paths
    ),
)
fig = loaded.analyze()
check(
    "multitone plot contains one labeled panel and maximum per device",
    len(fig.axes) == 2
    and [ax.get_title() for ax in fig.axes] == ["lower", "upper"]
    and all(np.all(np.diff(ax.lines[0].get_xdata()) >= 0) for ax in fig.axes),
)
plt.close("all")

raw_multi = make(readout_freqs=grid, amp=0.01, trace_source="raw", ddof=1)
run(raw_multi, FakeBias())
check(
    "scalar multitone amplitude broadcasts to each tone", np.array_equal(raw_multi.amp, [0.01] * 2)
)
check(
    "raw multitone std uses each selected column without mixing devices",
    np.allclose(
        raw_multi.std_arr,
        [
            [
                np.sqrt(np.linalg.eigvalsh(np.cov(np.vstack([z.real, z.imag])))[-1])
                for z in stream.signal.T
            ]
            for stream in FakeTimeStream.instances
        ],
    ),
)
for quantity in ("complex", "real", "imag", "abs"):
    choice = make(readout_freqs=grid, quantity=quantity, trace_source="raw")
    run(choice, FakeBias())
    reference = []
    for stream in FakeTimeStream.instances:
        z = stream.signal
        values = {"complex": z, "real": z.real, "imag": z.imag, "abs": np.abs(z)}[quantity]
        reference.append(np.std(values, axis=0))
    check(
        f"multitone raw {quantity} statistic agrees with numpy",
        np.allclose(choice.std_arr, reference),
    )

# Every tone's arbitrary IQ rotation must preserve its curve and winner.
saved_scales = SCALES.copy()
for i, freq in enumerate(grid.flat):
    value = complex(*SCALES[freq]) * np.exp(1j * (0.4 + i * 0.7))
    SCALES[freq] = (value.real, value.imag)
rotated_multi = make(readout_freqs=grid, ddof=0)
run(rotated_multi, FakeBias())
check(
    "independent IQ rotations preserve every multitone principal curve",
    np.allclose(rotated_multi.std_arr, expected, rtol=1e-12)
    and np.array_equal(rotated_multi.best_freq, multi.best_freq),
)
SCALES.clear()
SCALES.update(saved_scales)

loaded.std_arr[:] = 0
loaded._select_best()
check(
    "multitone ties and zero curves choose first row independently",
    np.array_equal(loaded.best_freq, grid[0]) and np.all(loaded.best_std == 0),
)
check(
    "zero variation is flagged for each device",
    all(
        any("not meaningful" in text.get_text() for text in ax.texts)
        for ax in loaded.analyze().axes
    ),
)
plt.close("all")
single_row = make(readout_freqs=grid[:1], lo_freq=3e9)
run(single_row, FakeBias())
check(
    "a one-row multitone scan returns one frequency per device",
    np.array_equal(single_row.best_freq, grid[0]) and single_row.lo_freq == 3e9,
)
one_column = make(readout_freqs=np.array(FREQS)[:, None])
run(one_column, FakeBias())
check(
    "a one-column matrix keeps matrix result semantics",
    one_column.std_arr.shape == (3, 1) and one_column.best_freq.shape == (1,),
)

bias = FakeBias(trigger_port=3)
run(loaded, bias)
check(
    "a loaded multitone sweep re-reads the generator routing",
    np.array_equal(loaded.trigger_states, [0, 0, 1]),
)
explicit = make(readout_freqs=grid, trigger_states=[0, 1, 1])
run(explicit, FakeBias())
check(
    "multitone respects an explicit trigger override",
    all(
        np.array_equal(stream.kwargs["external_trigger"], [0, 1, 1])
        for stream in FakeTimeStream.instances
    ),
)

FakeTimeStream.fail_at = loaded.lo_freq
try:
    run(loaded, bias)
    failed = False
except RuntimeError:
    failed = True
FakeTimeStream.fail_at = None
check(
    "a failed multitone re-run clears all results and turns off the bias",
    failed
    and not bias.output
    and loaded.std_arr is None
    and loaded.best_freq is None
    and loaded.qc_files is None
    and loaded.raw_files is None,
)
for label, call in (("save", loaded.save), ("analyze", loaded.analyze)):
    try:
        call()
        refused = False
    except RuntimeError:
        refused = True
    check(f"{label} refuses an incomplete multitone scan", refused)

bad_multi = []
for params in (
    dict(amp=[0.1]),
    dict(amp=[0.1, 0.2, 0.3]),
    dict(amp=[0.5, 0.5]),
    dict(amp=0.6),
    dict(amp=[0, 0.1]),
    dict(amp=[-0.1, 0.1]),
    dict(amp=[np.nan, 0.1]),
    dict(amp=[0.1, np.inf]),
    dict(lo_freq=np.nan),
    dict(lo_freq=-1),
    dict(lo_freq=2e9),
    dict(readout_freqs=[[2e9, 3e9]]),
    dict(readout_freqs=[[2.8e9, 2.8e9]]),
    dict(readout_freqs=[[]]),
    dict(tone_labels=["one"]),
    dict(tone_labels="ab"),
    dict(tone_labels=["", "two"]),
    dict(readout_freqs=[np.linspace(2.7e9, 2.8e9, 193)], amp=0.001),
):
    try:
        make(**dict(dict(readout_freqs=grid), **params))
        bad_multi.append(str(params))
    except ValueError:
        pass
check("invalid multitone plans are refused before acquisition", not bad_multi, str(bad_multi))

owned_multi_bias = FakeBias()
sweep_mod.Agilent33220A = lambda: owned_multi_bias
partial = make(readout_freqs=grid)
before_files = set(os.listdir(tmp.name))
FakeTimeStream.fail_on_acquisition = 2
try:
    run(partial)
    failed = False
except RuntimeError:
    failed = True
FakeTimeStream.fail_on_acquisition = None
new_qc_files = [
    name for name in set(os.listdir(tmp.name)) - before_files if name.endswith("-qc_trace.h5")
]
check(
    "a later multitone acquisition failure preserves completed QC files and closes owned bias",
    failed
    and len(new_qc_files) == 2
    and owned_multi_bias.closed
    and not owned_multi_bias.output
    and partial.std_arr is None
    and partial.best_freq is None,
)
with h5py.File(multi.qc_files[0][0], "a") as old_qc:
    del old_qc.attrs["tone"]
check(
    "QC records predating tone metadata still load as tone zero",
    QCTrace.load(multi.qc_files[0][0]).tone == 0,
)

# The user's 13-device plan must still acquire only one stream per offset.
centers = 2.7e9 + np.arange(13) * 10e6
thirteen_grid = centers[None, :] + np.array([-1e6, 0, 1e6])[:, None]
for row in thirteen_grid:
    for freq in row:
        SCALES[freq] = (1, 2)
thirteen = make(readout_freqs=thirteen_grid, amp=0.02)
run(thirteen, FakeBias())
check(
    "13 devices share three acquisitions for three offsets",
    len(FakeTimeStream.instances) == 3
    and thirteen.std_arr.shape == (3, 13)
    and thirteen.best_freq.shape == (13,)
    and all(stream.signal.shape[1] == 13 for stream in FakeTimeStream.instances),
)

tmp.cleanup()

# ------------------------------------------------------------------------------ summary
failed = [label for label, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label in failed:
        print("  FAILED:", label)
    sys.exit(1)
