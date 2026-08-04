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

_nspec = importlib.util.spec_from_file_location(f"{_PKG}.noise", _ANALYSIS_DIR / "noise.py")
noise = importlib.util.module_from_spec(_nspec)
sys.modules[_nspec.name] = noise
_nspec.loader.exec_module(noise)
from_elec_to_reson = noise.from_elec_to_reson


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


# ------------------------------------------------- 8. from_elec_to_reson, the other consumer


class FakeSweep:
    """The attributes ``from_elec_to_reson`` reads off a fitted Sweep."""

    freq_arr = FREQ_ARR
    resp_arr = s21_raw(FREQ_ARR)
    fit_results = dict(FIT, Qc_dia_corr=ABSQC, Qi_dia_corr=2.0e4)


SW = FakeSweep()


def reson(ts, readout_freq):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_elec_to_reson(ts, SW, readout_freq)


ts_det = s21_raw(FR + 300e3)
tsz_good, _, _ = reson(ts_det, FR + 300e3)
check(
    "from_elec_to_reson lands on the ring with readout_freq",
    off_ring(tsz_good) < 1e-9,
    f"off by {off_ring(tsz_good):.1e} radii",
)
check(
    "from_elec_to_reson agrees with plot_iq_comparison's projection",
    np.allclose(tsz_good, to_resonator(ts_det, FR + 300e3)),
    "the two consumers now share one implementation",
)

# ------------------------------------------------- 9. omitting it mixes dissipation into frequency

# Fluctuations placed purely along the *frequency* (arc / imaginary) axis. Under the wrong
# normalization the tsz plane is rotated by theta, mixing the two axes. The leak is NOT
# symmetric: rad = Re/q but arc = Im/(-2q), so rad_bad = cos(theta)*rad - 2*sin(theta)*arc and
# the dissipation channel picks up 4*sin^2(theta) of the frequency channel's power, while the
# frequency channel picks up only sin^2(theta)/4 of the dissipation channel's. Second order in
# theta, but multiplied by the ratio of the two -- which is the whole point of separating them.
THETA = 2 * np.pi * 300e3 * TAU
rng = np.random.default_rng(0)
base = s21_raw(FR + 300e3)[0]
env_at = float(A)  # |env| is flat in frequency; only its phase moves
# Build a cloud whose resonator-basis fluctuation is purely imaginary.
pure_arc = np.exp(1j * PHI0) * 1j * rng.normal(scale=1e-3, size=20_000)
ts_cloud = (base / env_true(FR + 300e3)[0] + pure_arc) * env_true(FR + 300e3)[0]

_, rad_good, arc_good = reson(ts_cloud, FR + 300e3)
_, rad_bad, arc_bad = reson(ts_cloud, None)

leak = np.var(rad_bad - rad_bad.mean()) / np.var(arc_good - arc_good.mean())
check(
    "omitting readout_freq leaks frequency noise into dissipation as 4*sin^2(theta)",
    np.isclose(leak, 4 * np.sin(THETA) ** 2, rtol=0.05),
    f"leak {leak:.3e} vs 4sin^2(theta) {4 * np.sin(THETA) ** 2:.3e} "
    f"(theta = {np.degrees(THETA):.2f} deg)",
)
# The reverse direction, to pin the asymmetry rather than just one side of it.
pure_rad = np.exp(1j * PHI0) * rng.normal(scale=1e-3, size=20_000)
ts_rad = (base / env_true(FR + 300e3)[0] + pure_rad) * env_true(FR + 300e3)[0]
_, rr_good, aa_good = reson(ts_rad, FR + 300e3)
_, rr_bad, aa_bad = reson(ts_rad, None)
leak_rev = np.var(aa_bad - aa_bad.mean()) / np.var(rr_good - rr_good.mean())
check(
    "the reverse leak is sin^2(theta)/4 -- the mixing is asymmetric",
    np.isclose(leak_rev, np.sin(THETA) ** 2 / 4, rtol=0.05),
    f"leak {leak_rev:.3e} vs sin^2(theta)/4 {np.sin(THETA) ** 2 / 4:.3e}, "
    f"a factor {leak / leak_rev:.0f} apart",
)
check(
    "the true dissipation axis carried no power to begin with",
    np.var(rad_good - rad_good.mean()) < 1e-12 * np.var(arc_good - arc_good.mean()),
    "so the leak above is entirely the mis-normalization",
)

# ------------------------------------------------- 10. plot_iq_comparison wires it to ts AND qc

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    check(
        "plot_iq_comparison passes the readout term to both ts and qc", True, "SKIP: no matplotlib"
    )
else:
    calls = []
    real_to_basis = plotting._to_basis

    def spy(z, env, phi0, basis):
        calls.append(np.asarray(env))
        return real_to_basis(z, env, phi0, basis)

    class FakePort:
        fitresults = SW.fit_results
        z_data_sim = s21_raw(FREQ_ARR) / env_true(FREQ_ARR)

    resonator_mod = sys.modules[f"{_PKG}.resonator"]
    real_fit_notch = getattr(resonator_mod, "fit_notch", None)
    plotting._to_basis = spy
    resonator_mod.fit_notch = lambda *a, **k: FakePort()
    try:
        plotting.plot_iq_comparison(
            ts_det, SW, qc=ts_det, basis="resonator", readout_freq=FR + 300e3
        )
    finally:
        plotting._to_basis = real_to_basis
        if real_fit_notch is not None:
            resonator_mod.fit_notch = real_fit_notch

    expected = _readout_env(SW.fit_results, FREQ_ARR, FR + 300e3, "resonator")
    check(
        "plot_iq_comparison passes the readout term to both ts and qc",
        len(calls) == 3
        and calls[0].size == FREQ_ARR.size  # sweep trace: the full array
        and np.isclose(calls[1], expected)  # cloud: the scalar at f_ro
        and np.isclose(calls[2], expected),  # QC points: the same scalar
        f"{len(calls)} projections: array for the trace, f_ro scalar for ts and qc",
    )


# ------------------------------------------------- plot_psd: labels, panels, per-panel fits
#
# plot_psd exists because the two PSD channels mean different things in different bases
# (dissipation/frequency against I/Q), and because a fit drawn over a panel must come from
# that panel's own array. Reading a stored fit off a measurement object is what these checks
# rule out: it describes whichever single channel was last computed, so overlaying it on both
# panels compares a spectrum against a model of a different spectrum.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

F_BW_PSD = 5000.0
F_PSD = np.arange(0.0, 2501.0)  # includes the f = 0 bin, which the plot must drop
GAMMA_A, GAMMA_B = 50.0, 500.0
PSD_A = noise.parity_psd_model(F_PSD, fidelity=0.9, gamma_p=GAMMA_A, f_bw=F_BW_PSD)
PSD_B = noise.parity_psd_model(F_PSD, fidelity=0.9, gamma_p=GAMMA_B, f_bw=F_BW_PSD)

try:
    import iminuit  # noqa: F401

    _HAVE_IMINUIT = True
except ImportError:  # pragma: no cover - depends on the workstation
    _HAVE_IMINUIT = False
    print("SKIP  plot_psd fit checks (iminuit not installed)")

axes, fits = plotting.plot_psd(F_PSD, PSD_A, PSD_B, basis="resonator", f_bw=F_BW_PSD, fit=False)
check(
    "plot_psd labels the resonator basis dissipation / frequency",
    [a.get_ylabel() for a in axes] == ["Dissipation PSD [1/Hz]", "Frequency PSD [1/Hz]"],
    f"{[a.get_ylabel() for a in axes]}",
)
check(
    "plot_psd drops the f = 0 bin, which no log axis can show",
    all(np.min(line.get_xdata()) > 0 for a in axes for line in a.get_lines()),
)
check("plot_psd with fit=False returns no fits", fits == {"a": None, "b": None})
plt.close("all")

axes, _ = plotting.plot_psd(F_PSD, PSD_A, PSD_B, basis="electronic", f_bw=F_BW_PSD, fit=False)
check(
    "plot_psd labels the electronic basis I / Q",
    [a.get_ylabel() for a in axes] == ["I PSD [FS$^2$/Hz]", "Q PSD [FS$^2$/Hz]"],
    f"{[a.get_ylabel() for a in axes]}",
)
plt.close("all")

axes, _ = plotting.plot_psd(
    F_PSD, PSD_A, PSD_B, labels=("|S|", "phase"), units="a.u.$^2$/Hz", f_bw=F_BW_PSD, fit=False
)
check(
    "plot_psd honours a labels / units override",
    [a.get_ylabel() for a in axes] == ["|S| PSD [a.u.$^2$/Hz]", "phase PSD [a.u.$^2$/Hz]"],
)
plt.close("all")

axes, fits = plotting.plot_psd(F_PSD, PSD_A, None, f_bw=F_BW_PSD, fit=False)
check("plot_psd draws one panel when psd_b is None", len(axes) == 1 and fits["b"] is None)
plt.close("all")

if _HAVE_IMINUIT:
    axes, fits = plotting.plot_psd(F_PSD, PSD_A, PSD_B, f_bw=F_BW_PSD)
    check(
        "plot_psd fits each panel from its own array",
        np.isclose(fits["a"]["gamma_p"], GAMMA_A, rtol=0.05)
        and np.isclose(fits["b"]["gamma_p"], GAMMA_B, rtol=0.05),
        f"gamma_p = {fits['a']['gamma_p']:.1f} / {fits['b']['gamma_p']:.1f} Hz "
        f"against {GAMMA_A} / {GAMMA_B}",
    )
    # The load-bearing property: the panel's fit is what fitting that panel's array gives.
    # A fit lifted off a measurement object would pass none of this.
    direct = noise.fit_parity_psd(F_PSD, PSD_B, f_bw=F_BW_PSD)
    check(
        "plot_psd's panel fit equals a direct fit of the same array",
        np.isclose(fits["b"]["gamma_p"], direct["gamma_p"], rtol=1e-9),
        f"{fits['b']['gamma_p']:.6f} against {direct['gamma_p']:.6f}",
    )
    check(
        "plot_psd frames the y-axis on the binned points, not the raw scatter",
        all(
            np.isclose(a.get_ylim()[0], np.min(r["psd_binned"]) / 30.0)
            for a, r in zip(axes, (fits["a"], fits["b"]))
        ),
    )
    plt.close("all")

    # f_bw only sets the white floor; inferring it as 2 * f[-1] is one bin off at most.
    _, fits_inferred = plotting.plot_psd(F_PSD, PSD_A, None)
    check(
        "plot_psd infers f_bw from the frequency axis",
        np.isclose(fits_inferred["a"]["gamma_p"], fits["a"]["gamma_p"], rtol=1e-3),
    )
    plt.close("all")

    # 2-D input: one PSD per row, and fit_parity_psd's own dict/list convention is kept.
    stacked = np.vstack([PSD_A, PSD_B])
    _, fits_2d = plotting.plot_psd(F_PSD, stacked, None, f_bw=F_BW_PSD)
    check(
        "plot_psd fits a 2-D channel row by row",
        isinstance(fits_2d["a"], list)
        and len(fits_2d["a"]) == 2
        and np.isclose(fits_2d["a"][0]["gamma_p"], GAMMA_A, rtol=0.05)
        and np.isclose(fits_2d["a"][1]["gamma_p"], GAMMA_B, rtol=0.05),
    )
    plt.close("all")

    # A fit that raises must not cost the spectrum.
    real_fit = noise.fit_parity_psd
    noise.fit_parity_psd = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        axes_bad, fits_bad = plotting.plot_psd(F_PSD, PSD_A, None, f_bw=F_BW_PSD)
    finally:
        noise.fit_parity_psd = real_fit
    check(
        "plot_psd survives a failed fit and still draws the spectrum",
        fits_bad["a"] is None and len(axes_bad[0].get_lines()) >= 1,
    )
    plt.close("all")

# Validation: every one of these is a silent-wrong-plot if it slips through.
for label, kwargs in (
    ("both channels None", dict(psd_a=None, psd_b=None)),
    ("an unknown basis", dict(psd_a=PSD_A, basis="fractional")),
    ("a PSD whose length differs from f", dict(psd_a=PSD_A[:-1])),
    ("the wrong number of axes in ax", dict(psd_a=PSD_A, psd_b=PSD_B, ax=(plt.subplots()[1],))),
):
    try:
        plotting.plot_psd(F_PSD, fit=False, **kwargs)
        check(f"plot_psd rejects {label}", False)
    except ValueError:
        check(f"plot_psd rejects {label}", True)
    plt.close("all")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
