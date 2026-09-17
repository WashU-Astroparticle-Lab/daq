"""Offline verification of ``daq.analysis.qc_periods`` against a synthetic gate-ramped trace.

The per-period digitisation reduces a sawtooth-biased readout to one state per gate period:
project on the principal axis, take the std inside each period, threshold at the midpoint of
the two populations, ``1`` = suppressed QC. Checked here on a trace whose suppressed periods
are known::

    python tests/test_qc_periods.py

- the principal axis recovers the direction the QC curve was drawn along;
- period boundaries follow **timestamps**: at a non-integral ``fs * period`` the sample counts
  alternate and the last edge stays within a sample of the truth, instead of drifting;
- ``ddof`` scales the std as it should, and ``phase_fraction`` shifts the origin;
- the two-mode cut resolves the populations, refuses a unimodal histogram, and the states
  reproduce the injected truth with unknowns as ``-1``, never ``0``;
- onsets are the ``0 -> 1`` transitions only.

Loads the module by path so it runs with numpy and scipy alone (no ``presto``).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "qc_periods", ROOT / "daq" / "analysis" / "qc_periods.py"
)
qc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qc  # dataclasses resolve annotations through sys.modules
spec.loader.exec_module(qc)

results = []


def check(label, condition, detail=""):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- synthetic trace

FS, RAMP_HZ, N_PERIODS = 100e3, 5e3, 2000
PER = int(FS / RAMP_HZ)  # 20
ANGLE = np.deg2rad(30.0)
A_NORMAL, A_SUPPRESSED, NOISE = 1e-3, 1e-4, 5e-5
rng = np.random.default_rng(20260917)

# Suppressed runs: known starts and lengths (in periods).
truth = np.zeros(N_PERIODS, dtype=np.int8)
RUNS = [(100 + 37 * k, (1, 2, 3, 5, 8)[k % 5]) for k in range(45)]  # ~9 % of periods suppressed
for start, length in RUNS:
    truth[start : start + length] = 1
RUN_STARTS = np.array([start for start, _ in RUNS])
qc_curve = np.sin(2 * np.pi * np.arange(PER) / PER)  # one QC feature per gate period
amp = np.where(truth == 1, A_SUPPRESSED, A_NORMAL)
x = (amp[:, None] * qc_curve[None, :]).ravel()
direction = np.exp(1j * ANGLE)
z = (
    (0.5 + 0.2j)
    + x * direction
    + NOISE * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
)

# ---------------------------------------------------------------- principal axis

axis, origin = qc.principal_axis(z)
check(
    "principal axis recovers the QC direction",
    abs(abs(axis @ [np.cos(ANGLE), np.sin(ANGLE)]) - 1) < 1e-3,
    str(axis),
)
check("origin is the mean", abs(origin - z.mean()) < 1e-12)
proj = qc.project(z, axis, origin)
check(
    "project() gives a real series of the same length", proj.shape == z.shape and np.isrealobj(proj)
)

# ---------------------------------------------------------------- period_std

ps = qc.period_std(z, FS, 1 / RAMP_HZ)
check("one row per complete period", ps.std.shape == (N_PERIODS, 1), str(ps.std.shape))
check("every period holds 20 samples", np.all(ps.counts == PER))
check("edges cover the record exactly", ps.sample_edges[0] == 0 and ps.sample_edges[-1] == z.size)
check("time edges are sample edges over fs", np.allclose(ps.time_edges_s, ps.sample_edges / FS))
expected_normal = A_NORMAL * np.std(qc_curve, ddof=1)
normal_med = np.median(ps.std[truth == 0, 0])
supp_med = np.median(ps.std[truth == 1, 0])
check(
    "normal periods spread as the injected curve",
    abs(normal_med - expected_normal) / expected_normal < 0.05,
    f"{normal_med:.3e} vs {expected_normal:.3e}",
)
check("suppressed periods spread far less", supp_med < 0.2 * normal_med, f"{supp_med:.3e}")
check("centers_s sit mid-period", np.allclose(ps.centers_s[:3], (np.arange(3) + 0.5) / RAMP_HZ))

ps0 = qc.period_std(z, FS, 1 / RAMP_HZ, ddof=0)
check("ddof=0 is sqrt(19/20) of ddof=1", np.allclose(ps0.std, ps.std * np.sqrt((PER - 1) / PER)))

# A supplied axis is used as given (here the true direction, versus the fitted one).
ps_axis = qc.period_std(z, FS, 1 / RAMP_HZ, axis=[np.cos(ANGLE), np.sin(ANGLE)])
check(
    "a supplied axis is honoured",
    np.allclose(ps_axis.axis, [[np.cos(ANGLE), np.sin(ANGLE)]])
    and np.allclose(ps_axis.std, ps.std, rtol=5e-2),
)

# Timestamp boundaries at a non-integral ratio: 100 kHz / 4999 Hz = 20.004 samples per period.
period_odd = 1 / 4999.0
ps_odd = qc.period_std(z, FS, period_odd)
n_expected = int(np.floor(z.size / FS / period_odd))
check(
    "non-integral ratio: counts alternate between 20 and 21",
    set(np.unique(ps_odd.counts).tolist()) <= {20, 21} and 21 in ps_odd.counts,
    str(np.unique(ps_odd.counts)),
)
check(
    "non-integral ratio: the last edge tracks the timestamp, not the block count",
    abs(ps_odd.sample_edges[-1] - round(n_expected * period_odd * FS)) <= 1
    and ps_odd.n_periods == n_expected,
    f"{ps_odd.sample_edges[-1]} vs {n_expected * period_odd * FS:.1f}",
)

ps_shift = qc.period_std(z, FS, 1 / RAMP_HZ, phase_fraction=0.25)
check(
    "phase_fraction shifts the origin by that fraction",
    ps_shift.sample_edges[0] == round(0.25 * PER) and ps_shift.n_periods == N_PERIODS - 1,
)

z2 = np.stack([z, z * np.exp(1j * 0.7)], axis=1)
ps2 = qc.period_std(z2, FS, 1 / RAMP_HZ)
check(
    "two tones: one column each, rotation-invariant std",
    ps2.std.shape == (N_PERIODS, 2) and np.allclose(ps2.std[:, 0], ps2.std[:, 1], rtol=1e-6),
)

z_nan = z.copy()
z_nan[PER * 10 + 3] = np.nan
ps_nan = qc.period_std(z_nan, FS, 1 / RAMP_HZ, axis=axis, origin=origin)
check(
    "a non-finite sample makes only its period nan",
    np.isnan(ps_nan.std[10, 0]) and np.isfinite(ps_nan.std[[9, 11], 0]).all(),
)

for bad in (dict(fs=0), dict(period_s=0), dict(phase_fraction=1.0), dict(ddof=-1), dict(ddof=25)):
    kw = dict(fs=FS, period_s=1 / RAMP_HZ)
    kw.update(bad)
    try:
        qc.period_std(z, kw.pop("fs"), kw.pop("period_s"), **kw)
        check(f"{bad} is refused", False)
    except ValueError:
        check(f"{bad} is refused", True)

# ---------------------------------------------------------------- two_mode_cut

cut = qc.two_mode_cut(ps.std[:, 0])
check(
    "two modes resolved on the bimodal histogram", cut.resolved and np.isfinite(cut.cut), cut.reason
)
check("the cut lies between the populations", supp_med < cut.cut < normal_med, f"{cut.cut:.3e}")
check("modes are ordered and bracket the cut", cut.modes[0] < cut.cut < cut.modes[1])
check(
    "diagnostics carry the histogram",
    cut.hist_counts.sum() == N_PERIODS and cut.smoothed.size == cut.hist_counts.size,
)

uni = qc.two_mode_cut(ps.std[truth == 0, 0])
check(
    "a unimodal histogram is left unresolved (nan), not forced",
    not uni.resolved and np.isnan(uni.cut),
    uni.reason,
)
few = qc.two_mode_cut(ps.std[:50, 0])
check(
    "too few values is refused with a reason",
    not few.resolved and "finite values" in few.reason,
    few.reason,
)

# ---------------------------------------------------------------- classify / onsets

states = qc.classify_periods(ps.std, cut.cut)
check(
    "states are int8 with the std's shape", states.dtype == np.int8 and states.shape == ps.std.shape
)
agree = np.mean(states[:, 0] == truth)
check("states reproduce the injected truth", agree > 0.995, f"agreement {agree:.4f}")
check("unresolved cut gives -1 everywhere", np.all(qc.classify_periods(ps.std, np.nan) == -1))
check("nan std gives -1, not 0", qc.classify_periods(ps_nan.std, cut.cut)[10, 0] == -1)
states2 = qc.classify_periods(ps2.std, [cut.cut, cut.cut])
check(
    "per-tone cuts broadcast",
    states2.shape == (N_PERIODS, 2) and np.array_equal(states2[:, 0], states2[:, 1]),
)
try:
    qc.classify_periods(ps2.std, [1.0, 2.0, 3.0])
    check("wrong number of cuts is refused", False)
except ValueError:
    check("wrong number of cuts is refused", True)

onsets = qc.state_onsets(truth)[0]
check("onsets are the 0->1 transitions", np.array_equal(onsets, RUN_STARTS), str(onsets[:6]))
onsets_t = qc.state_onsets(states, ps.time_edges_s)[0]
check(
    "onsets with edges are period start times",
    onsets_t.size >= 0.95 * RUN_STARTS.size
    and np.all(np.isin(np.round(onsets_t * RAMP_HZ).astype(int), RUN_STARTS)),
    f"{onsets_t.size} onsets",
)
edge_case = qc.state_onsets(np.array([1, 1, 0, -1, 1, 0, 1], dtype=np.int8))[0]
check(
    "a run starting at the record or after an unknown is not an onset",
    edge_case.tolist() == [6],
    str(edge_case),
)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
