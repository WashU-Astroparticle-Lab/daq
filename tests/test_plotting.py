"""Offline verification of ``plot_iq_comparison``'s single-frequency normalization.

Runs with no hardware, no VISA runtime, no MongoDB and no ``resonator_tools`` -- the checks
touch only ``_readout_env`` and ``_to_basis``, which need numpy and the numpy-only
``environmental_term`` helper::

    python tests/test_plotting.py

Prints one PASS/FAIL line per check and exits non-zero if any check fails.

This suite exists because of a specific, hard-to-see basis bug. ``plot_iq_comparison``
normalizes the sweep trace point by point, each frequency by its own environmental term, but a
time stream and its folded QC points sit at a *single* frequency and need that term evaluated
**there**. The function used to evaluate it at ``fr`` unconditionally, which is only right for
a stream taken on resonance.

The term carries the cable delay, ``env(f) = a e^{i alpha} e^{-2 pi i f tau}``, so dividing
data taken at ``f_ro`` by ``env(fr)`` leaves ``S21 * exp(-2 pi i (f_ro - fr) tau)``: a rigid
rotation of the cloud about the origin. Against a circle of radius ``Ql / (2 |Qc|)``, a few
hundred kHz of detuning on a shallow dip displaces the cloud by more than its own radius, and
nothing about the resulting plot says so -- it reads as a broken measurement.

Note in particular check 5: the *distance from the circle* is not a reliable tell, because a
rotation can carry the cloud back near some other arc of the ring. Only the displacement from
where the data belongs is diagnostic.
"""

import importlib.util
import sys
import types
import warnings
from pathlib import Path

import numpy as np

results = []


def check(label, condition, detail=""):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ``daq/__init__.py`` imports the measurement classes, which need ``presto``. Neither module
# under test does, so they are loaded straight from their files -- but ``plotting`` imports
# ``.resonator`` relatively, so they are loaded under a synthetic parent package whose
# ``__path__`` points at ``daq/analysis``. That keeps the analysis layer verifiable on a
# workstation with no presto install.
_ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "daq" / "analysis"
_PKG = "daq_analysis_shim"
_pkg = types.ModuleType(_PKG)
_pkg.__path__ = [str(_ANALYSIS_DIR)]
sys.modules[_PKG] = _pkg

_spec = importlib.util.spec_from_file_location(f"{_PKG}.plotting", _ANALYSIS_DIR / "plotting.py")
plotting = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = plotting
_spec.loader.exec_module(plotting)

_readout_env = plotting._readout_env
_to_basis = plotting._to_basis


# ------------------------------------------------- synthetic notch resonator

FR, QL, ABSQC, PHI0 = 2.8e9, 1.0e4, 1.0e5, 0.15
A, ALPHA, TAU = 0.7, 0.9, 50e-9

DIP = QL / ABSQC  # dip depth = circle diameter in the resonator basis
CENTRE, RADIUS = 1 - DIP / 2, DIP / 2

FREQ_ARR = FR + np.linspace(-2e6, 2e6, 401)


def env_true(freq):
    """The ground-truth environmental term of Eqn. 1."""
    return A * np.exp(1j * ALPHA) * np.exp(-2j * np.pi * np.atleast_1d(freq) * TAU)


def s21_raw(freq):
    """Raw (electronic-basis) S21 of the synthetic resonator."""
    freq = np.atleast_1d(freq)
    canonical = 1 - DIP * np.exp(1j * PHI0) / (1 + 2j * QL * (freq - FR) / FR)
    return env_true(freq) * canonical


# The ``fitresults`` mapping ``fit_notch`` produces, built the way ``fit_notch`` builds it.
FIT = {
    "fr": FR,
    "Ql": QL,
    "absQc": ABSQC,
    "phi0": PHI0,
    "environmental_term": env_true(FREQ_ARR),
    "environmental_amp_norm": A,
    "environmental_alpha": ALPHA,
    "environmental_delay": TAU,
}


def to_resonator(ts, readout_freq):
    """Project single-frequency data the way ``plot_iq_comparison`` does."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = _readout_env(FIT, FREQ_ARR, readout_freq, "resonator")
    return _to_basis(ts, env, PHI0, "resonator")


def off_ring(z):
    """Distance from the resonator-basis circle, in ring radii."""
    return float(np.max(np.abs(np.abs(np.asarray(z) - CENTRE) - RADIUS)) / RADIUS)


# ------------------------------------------------- 1. on resonance is unchanged

on_res = to_resonator(s21_raw(FR), FR)
check(
    "on-resonance data lands on the ring",
    off_ring(on_res) < 1e-9,
    f"off by {off_ring(on_res):.1e} radii",
)

# ------------------------------------------------- 2. detuned data lands on the ring

for delta in (100e3, 300e3, 1e6, 5e6):
    fixed = to_resonator(s21_raw(FR + delta), FR + delta)
    check(
        f"detuned {delta / 1e3:.0f} kHz lands on the ring with readout_freq",
        off_ring(fixed) < 1e-9,
        f"off by {off_ring(fixed):.1e} radii",
    )

check(
    "readout_freq outside the swept span is evaluated analytically, not interpolated",
    off_ring(to_resonator(s21_raw(FR + 50e6), FR + 50e6)) < 1e-9,
    "50 MHz detuning, far outside the +/-2 MHz sweep",
)

# ------------------------------------------------- 3. omitting it displaces the cloud

for delta in (100e3, 300e3, 1e6):
    ts = s21_raw(FR + delta)
    displacement = float(np.abs(to_resonator(ts, None) - to_resonator(ts, FR + delta))[0])
    # |S21| ~ 1 off resonance, so the chord of a rotation by theta is ~theta.
    predicted = 2 * abs(np.sin(np.pi * delta * TAU)) * float(np.abs(s21_raw(FR + delta))[0]) / A
    check(
        f"detuned {delta / 1e3:.0f} kHz is displaced without readout_freq",
        np.isclose(displacement, predicted, rtol=1e-9),
        f"{displacement / RADIUS:.2f} ring radii, matches 2|S|sin(pi*delta*tau)",
    )

# ------------------------------------------------- 4. the error is a rigid rotation


def to_fractional(ts, readout_freq):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = _readout_env(FIT, FREQ_ARR, readout_freq, "fractional")
    return _to_basis(ts, env, PHI0, "fractional")


ts_300 = s21_raw(FR + 300e3)
frac_bad, frac_good = to_fractional(ts_300, None), to_fractional(ts_300, FR + 300e3)
check(
    "the mis-normalization preserves modulus (a rotation, not a rescale)",
    np.allclose(np.abs(frac_bad), np.abs(frac_good)),
    f"|z| {float(np.abs(frac_bad)[0]):.9f} vs {float(np.abs(frac_good)[0]):.9f}",
)
rotation = float(np.angle(frac_bad / frac_good)[0])
check(
    "the rotation angle is exactly -2*pi*(f_ro - fr)*tau",
    np.isclose(rotation, -2 * np.pi * 300e3 * TAU, atol=1e-12),
    f"{np.degrees(rotation):.4f} deg",
)

# ------------------------------------------------- 5. off-ring distance is not the tell

# A rotation about the origin can carry the cloud back near a *different* arc of the ring, so
# "it looks like it is on the circle" does not mean it was normalized correctly -- it can sit
# at the wrong detuning instead. This is why the fix is a correct normalization rather than a
# check on the plotted distance.
near_ring = off_ring(to_resonator(s21_raw(FR + 300e3), None))
displaced = float(
    np.abs(to_resonator(s21_raw(FR + 300e3), None) - to_resonator(s21_raw(FR + 300e3), FR + 300e3))[
        0
    ]
    / RADIUS
)
check(
    "off-ring distance alone does not reveal the error",
    near_ring < 0.5 < displaced,
    f"{near_ring:.2f} radii from the ring, but {displaced:.2f} radii from where it belongs",
)

# ------------------------------------------------- 6. the warning is scoped

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _readout_env(FIT, FREQ_ARR, None, "resonator")
    _readout_env(FIT, FREQ_ARR, None, "fractional")
    noisy = len(caught)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _readout_env(FIT, FREQ_ARR, None, "electronic")  # env is never divided out
    _readout_env(FIT, FREQ_ARR, FR, "resonator")  # frequency was given
    quiet = len(caught)
check(
    "warns when the frequency is assumed and matters, silent otherwise",
    noisy == 2 and quiet == 0,
    f"{noisy} warnings when it matters, {quiet} when it does not",
)

# ------------------------------------------------- 7. validation

try:
    _readout_env(FIT, FREQ_ARR, -1.0, "resonator")
    check("a non-positive readout_freq raises", False)
except ValueError:
    check("a non-positive readout_freq raises", True)


print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
