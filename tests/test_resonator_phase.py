"""Offline verification of ``resonator_phase`` / ``dtheta_to_dx`` on a synthetic notch.

A KID's response is read as the phase angle about the resonance circle's centre (straxion's
``DxRecords`` convention: off-resonance at 0, resonance at pi) and converted to a fractional
resonance shift. Checked against a synthetic notch with a known environmental term, fitted
with ``fit_notch``, then read out at a fixed frequency while the resonance is moved by known
amounts::

    python tests/test_resonator_phase.py

- the unshifted operating point gives ``dtheta ~ 0`` and ``dr ~ 0``;
- a known ``delta_fr`` comes back from ``dtheta_to_dx`` as ``delta_fr / fr``, with the sign of
  a resonance that moved down being negative, over shifts up to a linewidth;
- the far-off-resonance and on-resonance angles are 0 and pi;
- a ``Sweep``-like object carrying ``fit_results`` is accepted.

Loads ``resonator.py`` by path; skips itself when ``resonator_tools`` is absent.
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parent.parent / "daq" / "analysis" / "resonator.py"
_spec = importlib.util.spec_from_file_location("daq_analysis_resonator", _MODULE_PATH)
resonator = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = resonator
_spec.loader.exec_module(resonator)

if not resonator.resonator_tools_available():
    print("SKIP  resonator_tools is not installed; nothing to verify")
    sys.exit(0)

results = []


def check(label, condition, detail=""):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


FR, QL, QC, PHI = 2.85e9, 2.6e4, 1.5e5, 0.12  # a KID-like notch (Ql/|Qc| = 0.17, a 1.6 dB dip)
A, ALPHA, DELAY = 0.4, 0.8, 4.0e-9


def s21(freq, fr=FR):
    norm = 1.0 - (QL / abs(QC)) * np.exp(1j * PHI) / (1.0 + 2j * QL * (np.asarray(freq) / fr - 1.0))
    return resonator.environmental_term(np.asarray(freq, dtype=float), A, ALPHA, DELAY) * norm


freq = np.linspace(FR - 1e6, FR + 1e6, 2001)
rng = np.random.default_rng(3)
resp = s21(freq) + 2e-5 * (rng.standard_normal(freq.size) + 1j * rng.standard_normal(freq.size))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    port = resonator.fit_notch(freq, resp)
fit = port.fitresults
check(
    "fit recovers fr and the circle diameter Ql/|Qc|",
    abs(fit["fr"] - FR) < 2e3 and abs(fit["Ql"] / fit["absQc"] / (QL / abs(QC)) - 1) < 0.03,
    f"{fit['fr'] - FR:+.0f} Hz, d {fit['Ql'] / fit['absQc']:.4f} vs {QL / abs(QC):.4f}; "
    f"Ql {fit['Ql'] / QL - 1:+.1%} (upstream bias on a shallow notch with delay)",
)

F_RO = FR + 30e3  # read out slightly above resonance, as a KID often is
linewidth = FR / QL

# ---------------------------------------------------------------- operating point

res = resonator.resonator_phase(s21(F_RO), fit, F_RO)
check(
    "unshifted readout: dtheta offset is what the Ql bias predicts",
    abs(res["dtheta"]) < 0.3,
    f"{res['dtheta']:.4f} rad (fit Ql/true = {fit['Ql'] / QL:.3f})",
)
check("...and dr ~ 0", abs(res["dr"]) < 0.05, f"{res['dr']:.4f}")
check("theta_ro is between 0 and pi for a readout above resonance", 0 < res["theta_ro"] < np.pi)
check(
    "centre and radius describe the fitted circle",
    abs(res["radius"] - 0.5 * fit["Ql"] / fit["absQc"]) < 1e-12
    and abs(res["center"] + res["radius"] - 1) < 1e-12,
)

# Eight linewidths off, inside the swept span: the fitted delay is only trusted there.
far = resonator.resonator_phase(s21(FR + 8 * linewidth), fit, FR + 8 * linewidth)
check("far off resonance theta -> 0", abs(far["theta"]) < 0.15, f"{far['theta']:.3f}")
on = resonator.resonator_phase(s21(fit["fr"]), fit, fit["fr"])
check("on resonance theta ~ pi", abs(abs(on["theta"]) - np.pi) < 0.05, f"{on['theta']:.3f}")

# ---------------------------------------------------------------- resonance shifts
#
# What a KID pulse is: an excursion of the resonance from its operating point. A constant
# offset of the angle (the fit's Ql bias, a 1e-4 normalisation error amplified by 1/radius on
# a small circle) shifts where the operating point sits, which any per-window baseline
# removes; what must be right is the *excursion* dx(delta) - dx(0) against delta_fr / fr.

shifts = np.array([-1.0, -0.3, -0.1, -0.01, 0.0, 0.01, 0.1, 0.3, 1.0]) * linewidth
z = np.array([s21(F_RO, fr=FR + d) for d in shifts])
res = resonator.resonator_phase(z, fit, F_RO)
truth = shifts / FR
moved = shifts != 0

dx = resonator.dtheta_to_dx(res["dtheta"], fit, F_RO, sweep=(freq, resp))
exc = dx - dx[shifts == 0]
# 5 %: the table is the sweep itself, sampled every 1 kHz with noise, and the smallest shift
# here is 1.1 kHz -- the limit is the sweep's resolution, not the mapping.
check(
    "sweep table: excursions recover delta_fr / fr to 5 % over +-1 linewidth",
    np.all(np.abs(exc[moved] - truth[moved]) < 0.05 * np.abs(truth[moved])),
    str(np.round((exc[moved] - truth[moved]) / truth[moved], 4)),
)
check(
    "sweep table: zero shift maps back to the readout frequency (< 100 Hz)",
    abs(dx[shifts == 0][0]) * FR < 100,
    f"{dx[shifts == 0][0]:.2e}",
)
check(
    "a resonance that moved down gives a negative excursion",
    np.all(exc[shifts < 0] < 0) and np.all(exc[shifts > 0] > 0),
)
check(
    "dtheta is monotonic in the shift",
    np.all(np.diff(res["dtheta"]) > 0) or np.all(np.diff(res["dtheta"]) < 0),
)
check(
    "dr stays ~0 for pure frequency shifts",
    np.all(np.abs(res["dr"]) < 0.05),
    f"max {np.abs(res['dr']).max():.3f}",
)
check("outputs keep the input shape", res["dtheta"].shape == z.shape and dx.shape == z.shape)

# The closed-form path is exact given the true circle -- the formula is right -- and inherits
# the fit's Ql bias otherwise, which is why the sweep table is the recommended path.
true_fit = dict(
    fit,
    Ql=QL,
    absQc=abs(QC),
    fr=FR,
    phi0=PHI,
    environmental_amp_norm=A,
    environmental_alpha=ALPHA,
    environmental_delay=DELAY,
)
res_true = resonator.resonator_phase(z, true_fit, F_RO, warn=False)
dx_true = resonator.dtheta_to_dx(res_true["dtheta"], true_fit, F_RO)
exc_true = dx_true - dx_true[shifts == 0]
check(
    "model path with the true circle: excursions exact to 1 %",
    np.all(np.abs(exc_true[moved] - truth[moved]) < 0.01 * np.abs(truth[moved])),
    str(np.round((exc_true[moved] - truth[moved]) / truth[moved], 4)),
)
dx_model = resonator.dtheta_to_dx(res["dtheta"], fit, F_RO)
exc_model = dx_model - dx_model[shifts == 0]
ql_bias = fit["Ql"] / QL - 1
check(
    "model path with the fitted circle: excursion error is of order the Ql bias",
    np.all(np.abs(exc_model[moved] - truth[moved]) < (abs(ql_bias) + 0.1) * np.abs(truth[moved])),
    f"Ql bias {ql_bias:+.1%}, max rel err {np.max(np.abs(exc_model[moved] - truth[moved]) / np.abs(truth[moved])):.2f}",
)

# ---------------------------------------------------------------- inputs


class FakeSweep:
    fit_results = fit


alt = resonator.resonator_phase(z, FakeSweep(), F_RO)
check("a Sweep-like object with fit_results is accepted", np.allclose(alt["dtheta"], res["dtheta"]))
for bad in (None, {"fr": FR}):
    try:
        resonator.resonator_phase(z, bad, F_RO)
        check(f"fit={bad!r} is refused", False)
    except (TypeError, KeyError):
        check(f"fit={bad!r} is refused", True)
try:
    resonator.resonator_phase(z, fit, float("nan"))
    check("nan readout_freq is refused", False)
except ValueError:
    check("nan readout_freq is refused", True)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
