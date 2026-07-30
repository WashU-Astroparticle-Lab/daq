"""Offline verification of ``daq.analysis.resonator`` against stock ``resonator_tools``.

Runs with no hardware, no VISA runtime and no MongoDB. It needs the optional
``resonator_tools`` dependency and skips cleanly (exit 0) when that is absent::

    pip install resonator_tools
    python tests/test_resonator.py

Prints one PASS/FAIL line per check and exits non-zero if any check fails.

This suite exists because of a specific, hard-to-see bug. DAQ needs the
environmental term ``a·e^{iα}·e^{-2πifτ}`` of Eqn. 1 to move between the electronic
and resonator bases, but upstream ``notch_port.autofit()`` computes it as a local
and discards it. DAQ silently depended on a private fork of ``resonator_tools`` that
stored it in ``fitresults``; on a stock install, ``plot_iq_comparison`` and
``from_elec_to_reson`` died with ``KeyError: 'environmental_term'``. Nothing in the
repository declared or checked that dependency.

``daq.analysis.resonator.fit_notch`` now recovers the term through the public
``do_calibration()`` API. The checks below fit a synthetic resonator with a *known*
injected environmental term and verify both that the recovery matches ground truth
and that it exactly reproduces upstream's own normalization -- so a future upstream
change to the calibration convention fails here rather than in the lab.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

results = []


def check(label, condition, detail=""):
    results.append((label, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


# ``daq/__init__.py`` imports the measurement classes, which need ``presto``. This
# module has no such dependency -- it only touches numpy and resonator_tools -- so it
# is loaded straight from its file. That keeps the analysis layer verifiable on an
# analysis workstation with no presto install.
_MODULE_PATH = Path(__file__).resolve().parent.parent / "daq" / "analysis" / "resonator.py"
_spec = importlib.util.spec_from_file_location("daq_analysis_resonator", _MODULE_PATH)
resonator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resonator)

ResonatorFitError = resonator.ResonatorFitError
environmental_term = resonator.environmental_term
fit_notch = resonator.fit_notch
resonator_tools_available = resonator.resonator_tools_available

if not resonator_tools_available():
    print("SKIP  resonator_tools is not installed; nothing to verify")
    print("      install it with: pip install resonator_tools")
    sys.exit(0)


# ------------------------------------------------- synthetic notch resonator

FR_TRUE, QL_TRUE, QC_TRUE, PHI_TRUE = 5.0e9, 2.0e4, 3.0e4, 0.15
A_TRUE, ALPHA_TRUE, DELAY_TRUE = 0.37, 0.9, 4.2e-9


def make_sweep(noise=2e-4, seed=0, n=4001):
    """Build a synthetic S21 notch trace with a known environmental term."""
    freq = np.linspace(FR_TRUE - 2e6, FR_TRUE + 2e6, n)
    s21_norm = 1.0 - (QL_TRUE / abs(QC_TRUE)) * np.exp(1j * PHI_TRUE) / (
        1.0 + 2j * QL_TRUE * (freq / FR_TRUE - 1.0)
    )
    env = environmental_term(freq, A_TRUE, ALPHA_TRUE, DELAY_TRUE)
    resp = env * s21_norm
    if noise:
        rng = np.random.default_rng(seed)
        resp = resp + rng.normal(0, noise, n) + 1j * rng.normal(0, noise, n)
    return freq, resp, env


freq_arr, resp_arr, env_true = make_sweep()

# ------------------------------------------------- the bug this guards against

from resonator_tools import circuit  # noqa: E402

stock = circuit.notch_port(freq_arr, resp_arr)
stock.autofit()

# Which flavour is installed? The old WashU fork patched environmental_term into
# fitresults; upstream does not. Both must work -- that is what makes migrating off
# the fork safe -- so this is reported, not asserted.
IS_FORK = "environmental_term" in stock.fitresults
print(
    f"INFO  installed resonator_tools {'is the WashU fork' if IS_FORK else 'is stock upstream'}"
    f" (autofit {'does' if IS_FORK else 'does not'} provide environmental_term)"
)

# ------------------------------------------------- fit_notch supplies it

port = fit_notch(freq_arr, resp_arr)

for key in (
    "environmental_term",
    "environmental_baseline",
    "environmental_amp_norm",
    "environmental_alpha",
    "environmental_delay",
    "environmental_A2",
    "environmental_frcal",
):
    check(f"fit_notch provides {key}", key in port.fitresults)

check(
    "fit_notch preserves the standard fitresults keys",
    all(k in port.fitresults for k in ("fr", "Ql", "absQc", "phi0", "Qi_dia_corr", "fr_err")),
)
check(
    "fit_notch returns a real notch_port (z_data_sim present)",
    getattr(port, "z_data_sim", None) is not None and port.z_data_sim.shape == freq_arr.shape,
)

# ------------------------------------------------- recovery matches ground truth

env_fit = np.asarray(port.fitresults["environmental_term"])
rel_err = np.max(np.abs(env_fit - env_true) / np.abs(env_true))
check(
    "recovered environmental term matches the injected truth",
    rel_err < 1e-3,
    f"max rel err {rel_err:.2e}",
)
check(
    # Loose on purpose: the fitted delay is the noisiest of the recovered scalars, and
    # across seeds at this noise level it lands anywhere in ~[4e-5, 6e-3] relative. A
    # 1e-3 bound passes on seed 0 but fails on most others -- it would be a flaky check
    # measuring the noise realization, not the code. The quantity that actually matters
    # is the assembled environmental term, bounded tightly above.
    "recovered delay matches the injected delay",
    abs(port.fitresults["environmental_delay"] - DELAY_TRUE) / DELAY_TRUE < 1e-2,
    f"{port.fitresults['environmental_delay']:.6e} vs {DELAY_TRUE:.6e}",
)
check(
    "recovered amp_norm matches the injected amplitude",
    abs(port.fitresults["environmental_amp_norm"] - A_TRUE) / A_TRUE < 1e-2,
    f"{port.fitresults['environmental_amp_norm']:.6f} vs {A_TRUE:.6f}",
)

# ------------------------------------------------- exactness vs upstream

# do_normalization is z_norm = (z_raw - baseline) / env. Checked in product form so
# it stays well conditioned where the normalized response dips toward zero.
baseline = np.asarray(port.fitresults["environmental_baseline"])
residual = np.max(np.abs(np.asarray(port.z_data) * env_fit - (resp_arr - baseline)))
scale = float(np.median(np.abs(resp_arr)))
check(
    "environmental term exactly reproduces upstream normalization",
    residual / scale < 1e-10,
    f"relative residual {residual / scale:.2e}",
)
check(
    "baseline is identically zero (upstream pins A2 = 0)",
    np.all(baseline == 0.0) and port.fitresults["environmental_A2"] == 0.0,
)

if IS_FORK:
    # Running on the old fork: prove the adapter reproduces the exact values the lab
    # has been analysing with, so migrating off the fork changes no published number.
    fork_env = np.asarray(stock.fitresults["environmental_term"])
    drift = np.max(np.abs(env_fit - fork_env)) / float(np.median(np.abs(fork_env)))
    check(
        "adapter reproduces the fork's environmental_term exactly",
        drift < 1e-12,
        f"relative drift {drift:.2e}",
    )
    for key in ("environmental_amp_norm", "environmental_alpha", "environmental_delay"):
        same = np.isclose(port.fitresults[key], stock.fitresults[key], rtol=1e-12, atol=0.0)
        check(f"adapter reproduces the fork's {key}", same)

# ------------------------------------------------- fit quality is unchanged

check(
    "fit_notch reproduces the stock fit (fr)",
    abs(port.fitresults["fr"] - stock.fitresults["fr"]) < 1e-6,
    "adapter must not perturb the fit",
)
check(
    "fit_notch reproduces the stock fit (Ql)",
    abs(port.fitresults["Ql"] - stock.fitresults["Ql"]) / stock.fitresults["Ql"] < 1e-9,
)
check(
    "fitted fr matches the synthetic resonator",
    abs(port.fitresults["fr"] - FR_TRUE) / FR_TRUE < 1e-6,
    f"{port.fitresults['fr']:.6e} vs {FR_TRUE:.6e}",
)
check(
    "fitted phi0 matches the synthetic resonator",
    abs(port.fitresults["phi0"] - PHI_TRUE) < 1e-2,
    f"{port.fitresults['phi0']:.4f} vs {PHI_TRUE:.4f}",
)

# ------------------------------------------------- independence from the fork

# The whole point of the adapter is that it never reads the fork's extra fitresults
# keys. Asserting that only via IS_FORK would leave the blind spot that caused the
# original bug: on the lab machine, where the fork IS installed, a revert to reading
# fitresults["environmental_term"] directly would still pass every check above.
# So simulate a stock install regardless of what is installed -- strip the fork's keys
# right after autofit, and require fit_notch to derive them anyway.
_real_autofit = circuit.notch_port.autofit


def _stock_autofit(self, *args, **kwargs):
    _real_autofit(self, *args, **kwargs)
    for key in [k for k in self.fitresults if k.startswith("environmental_")]:
        del self.fitresults[key]


circuit.notch_port.autofit = _stock_autofit
try:
    stripped = fit_notch(freq_arr, resp_arr)
    check(
        "fit_notch derives environmental_term without the fork's keys",
        "environmental_term" in stripped.fitresults,
    )
    check(
        "derived term is identical with the fork's keys stripped",
        np.array_equal(np.asarray(stripped.fitresults["environmental_term"]), env_fit),
    )
except Exception as exc:  # noqa: BLE001 - report, do not mask
    check("fit_notch derives environmental_term without the fork's keys", False, repr(exc))
    check("derived term is identical with the fork's keys stripped", False, repr(exc))
finally:
    circuit.notch_port.autofit = _real_autofit

# ------------------------------------------------- fcrop path

f_ctr = freq_arr[np.argmin(np.abs(resp_arr))]
span = float(freq_arr.max() - freq_arr.min())
fcrop = (f_ctr - span / 4, f_ctr + span / 4)

cropped = fit_notch(freq_arr, resp_arr, fcrop=fcrop)
check(
    "fcrop fit still supplies environmental_term over the full grid",
    np.asarray(cropped.fitresults["environmental_term"]).shape == freq_arr.shape,
)
crop_resid = np.max(
    np.abs(
        np.asarray(cropped.z_data) * np.asarray(cropped.fitresults["environmental_term"])
        - (resp_arr - np.asarray(cropped.fitresults["environmental_baseline"]))
    )
)
check(
    "fcrop fit stays consistent with upstream normalization",
    crop_resid / scale < 1e-10,
    f"relative residual {crop_resid / scale:.2e}",
)

stock_crop = circuit.notch_port(freq_arr, resp_arr)
stock_crop.autofit(fcrop=fcrop)
check(
    "fcrop fit reproduces the stock cropped fit (fr)",
    abs(cropped.fitresults["fr"] - stock_crop.fitresults["fr"]) < 1e-6,
)

# ------------------------------------------------- guesses are forwarded

guessed = fit_notch(freq_arr, resp_arr, fr_guess=FR_TRUE, Ql_guess=QL_TRUE)
check(
    "fr_guess/Ql_guess are accepted and still yield a good fit",
    abs(guessed.fitresults["fr"] - FR_TRUE) / FR_TRUE < 1e-6,
    f"{guessed.fitresults['fr']:.6e}",
)

# ------------------------------------------------- consumers get what they need

# from_elec_to_reson indexes environmental_term at the resonance; plot_iq_comparison
# divides the whole sweep trace by it. Both need a finite, non-zero array.
check(
    "environmental_term is finite and non-zero everywhere",
    np.all(np.isfinite(env_fit)) and np.all(np.abs(env_fit) > 0),
)
fr_idx = int(np.argmin(np.abs(freq_arr - port.fitresults["fr"])))
check(
    "environmental_term is usable at the resonance index",
    np.isfinite(env_fit[fr_idx]) and abs(env_fit[fr_idx]) > 0,
    f"|env(fr)| = {abs(env_fit[fr_idx]):.4f}",
)

# The electronic -> fractional transform used by the consumers must invert cleanly.
round_trip = np.max(np.abs((resp_arr / env_fit) * env_fit - resp_arr)) / scale
check("dividing by environmental_term round-trips", round_trip < 1e-12)

# ------------------------------------------------- error surface

check("ResonatorFitError is a RuntimeError", issubclass(ResonatorFitError, RuntimeError))

# The consistency check guards the *basis transformation*, not the *fit quality* -- a
# poor fit is reported through fr_err/chi_square as usual. Verify the guard actually
# fires when the environmental term stops explaining upstream's normalization, which
# is how a future change to the calibration convention would show up.
try:
    resonator._check_consistency(
        port,
        resp_arr,
        env_fit * 1.05,  # a 5% gain error: silently wrong basis, no exception otherwise
        baseline,
    )
    check("consistency check rejects a wrong environmental term", False, "did not raise")
except ResonatorFitError:
    check("consistency check rejects a wrong environmental term", True)

try:
    resonator._check_consistency(port, resp_arr, env_fit, baseline)
    check("consistency check accepts the correct environmental term", True)
except ResonatorFitError as exc:
    check("consistency check accepts the correct environmental term", False, str(exc))

# A check that silently skips itself is worse than no check, so every branch that
# cannot complete the comparison must raise.
try:
    resonator._check_consistency(port, resp_arr, env_fit * np.nan, baseline)
    check("consistency check rejects a non-finite environmental term", False, "did not raise")
except ResonatorFitError:
    check("consistency check rejects a non-finite environmental term", True)

try:
    resonator._check_consistency(port, np.zeros_like(resp_arr), env_fit, baseline)
    check("consistency check rejects an all-zero response", False, "did not raise")
except ResonatorFitError:
    check("consistency check rejects an all-zero response", True)

# The consumers divide by environmental_term alone, so a non-zero baseline slope would
# bias every basis transformation without tripping the normalization identity.
_real_do_calibration = circuit.notch_port.do_calibration


def _nonzero_a2(self, *args, **kwargs):
    delay, amp_norm, alpha, fr, ql, _a2, frcal = _real_do_calibration(self, *args, **kwargs)
    return delay, amp_norm, alpha, fr, ql, 5e-9, frcal


circuit.notch_port.do_calibration = _nonzero_a2
try:
    fit_notch(freq_arr, resp_arr)
    check("fit_notch rejects a non-zero baseline slope (A2)", False, "did not raise")
except ResonatorFitError:
    check("fit_notch rejects a non-zero baseline slope (A2)", True)
finally:
    circuit.notch_port.do_calibration = _real_do_calibration

# A degenerate input must not yield a silently wrong basis: either it raises, or the
# environmental term it returns still explains the normalization exactly.
try:
    tiny = fit_notch(freq_arr[:5], resp_arr[:5])
    tiny_env = np.asarray(tiny.fitresults["environmental_term"])
    tiny_base = np.asarray(tiny.fitresults["environmental_baseline"])
    tiny_resid = np.max(np.abs(np.asarray(tiny.z_data) * tiny_env - (resp_arr[:5] - tiny_base)))
    check(
        "a degenerate fit still returns a self-consistent basis",
        tiny_resid / float(np.median(np.abs(resp_arr[:5]))) < 1e-10,
        f"relative residual {tiny_resid / float(np.median(np.abs(resp_arr[:5]))):.2e}",
    )
except Exception as exc:
    check("a degenerate fit raises rather than returning wrong data", True, type(exc).__name__)

# ------------------------------------------------- summary
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    for label, _, detail in failed:
        print("  FAILED:", label, detail)
    sys.exit(1)
