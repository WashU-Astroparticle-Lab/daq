# -*- coding: utf-8 -*-
"""Resonator circle fitting on top of the upstream ``resonator_tools`` package.

This module is the single entry point every DAQ measurement uses to fit a notch-port
resonator. It exists because DAQ needs the *environmental term* of Eqn. 1 --
``a·e^{iα}·e^{-2πifτ}``, the cable delay and gain/phase prefactor that
:func:`~daq.analysis.noise.from_elec_to_reson` and
:func:`~daq.analysis.plotting.plot_iq_comparison` divide out to move between the
electronic and resonator bases.

Upstream ``resonator_tools`` computes exactly those quantities inside
``notch_port.autofit()`` and then discards them: they are locals, absent from
``fitresults``. DAQ used to paper over this with a private fork that stored them as
extra ``fitresults`` keys, which made every install silently dependent on that fork
-- a stock ``pip install resonator_tools`` raised ``KeyError: 'environmental_term'``
deep inside plotting.

The fork is unnecessary. ``notch_port.do_calibration()`` is public API and returns
``(delay, amp_norm, alpha, fr, Ql, A2, frcal)`` -- every value the fork saved. So we
let stock ``autofit()`` own the whole fitting algorithm and simply re-run the
(deterministic) calibration to recover the scalars, then rebuild the environmental
term analytically and merge it into ``fitresults``.

Two facts from the upstream implementation make this exact rather than approximate:

- ``get_delay()`` sets ``A2 = 0.0`` unconditionally, in both branches of its
  ``ignoreslope`` test, so the baseline ``A2·(f - frcal)`` is identically zero.
- ``do_normalization()`` is then just ``z_norm = z_raw / environmental_term``.

:func:`fit_notch` checks both on every call -- it raises if ``A2`` is ever non-zero,
and if the recovered term fails to reproduce upstream's own normalization -- so a
future release that changes the calibration convention fails loudly instead of
returning a subtly wrong basis transformation. The ``A2`` check matters because the
consumers divide by ``environmental_term`` alone and ignore ``environmental_baseline``:
validating only the normalization identity would let a non-zero baseline through.
"""

import warnings
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "ResonatorFitError",
    "environmental_term",
    "fit_notch",
    "readout_environmental_term",
    "resonator_tools_available",
]

#: Relative tolerance for the self-consistency check in :func:`fit_notch`. The
#: reconstruction is exact up to floating-point round-off (observed ~1e-14 on a
#: 4001-point sweep), so this is loose enough to never fire on numerical noise and
#: tight enough to catch any real change in the upstream calibration convention.
_CONSISTENCY_RTOL = 1e-8

_INSTALL_HINT = (
    "resonator_tools is required for resonator fitting. Install it with:\n"
    "    pip install resonator_tools\n"
    "(DAQ needs the upstream package -- the old WashU fork is no longer required.)"
)


class ResonatorFitError(RuntimeError):
    """Raised when a resonator fit cannot be completed or fails validation."""


def resonator_tools_available() -> bool:
    """Report whether the optional ``resonator_tools`` dependency can be imported.

    Probes ``resonator_tools.circuit`` rather than the top-level package:
    ``resonator_tools/__init__.py`` is empty, so ``import resonator_tools`` succeeds
    even on an install whose ``circuit`` submodule is broken.

    :return: ``True`` when :mod:`resonator_tools.circuit` is importable.
    :rtype: bool
    """
    try:
        import resonator_tools.circuit  # noqa: F401
    except ImportError:
        return False
    return True


def _import_circuit() -> ModuleType:
    """Import ``resonator_tools.circuit`` with an actionable error message.

    :return: The ``resonator_tools.circuit`` module.
    :raises ImportError: If ``resonator_tools`` is not installed.
    """
    try:
        from resonator_tools import circuit
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(_INSTALL_HINT) from exc
    return circuit


def environmental_term(
    freq_arr: npt.NDArray[np.float64],
    amp_norm: float,
    alpha: float,
    delay: float,
) -> npt.NDArray[np.complex128]:
    """Evaluate the environmental term ``a·e^{iα}·e^{-2πifτ}`` of Eqn. 1.

    This is the multiplicative gain, phase offset and cable delay that the readout
    chain imposes on the resonator response. Dividing raw S21 by it yields the
    normalized response used for the circle fit.

    :param freq_arr: Frequencies in Hz at which to evaluate the term.
    :param amp_norm: Amplitude prefactor ``a`` (dimensionless gain).
    :param alpha: Constant phase offset ``α`` in radians.
    :param delay: Electrical delay ``τ`` in seconds.
    :return: Complex environmental term, same shape as *freq_arr*.
    :rtype: numpy.ndarray
    """
    freq_arr = np.asarray(freq_arr, dtype=float)
    return amp_norm * np.exp(1j * alpha) * np.exp(-2j * np.pi * freq_arr * delay)


def readout_environmental_term(
    fit: Dict[str, Any],
    readout_freq: Optional[float],
    *,
    freq_arr: Optional[npt.NDArray[np.float64]] = None,
    warn: bool = True,
    caller: str = "This basis transformation",
    remedy: str = "Pass readout_freq=<acquisition frequency in Hz>",
    stacklevel: int = 3,
) -> complex:
    """Return the environmental term at the frequency single-frequency data was taken at.

    A frequency sweep is normalized point by point, each frequency by its own environmental
    term. A time stream, its folded QC points and anything else acquired at a *fixed* tone are
    all at **one** frequency, and need that term evaluated **there**.

    Evaluating it at ``fr`` instead is not a harmless approximation, because the term carries
    the cable delay, ``env(f) = a e^{i alpha} e^{-2 pi i f tau}``. Dividing data taken at
    ``f_ro`` by ``env(fr)`` leaves ``S21 * exp(-2 pi i (f_ro - fr) tau)`` -- a rigid rotation
    about the origin, by ``theta = 2 pi (f_ro - fr) tau``. What that costs depends on the
    consumer:

    - Plotting a cloud against the fitted circle, it is a *position* error of order ``theta``
      against a circle of radius ``Ql / (2 |Qc|)``, so on a shallow resonance a few hundred
      kHz is enough to put the cloud entirely off the ring.
    - Splitting fluctuations into dissipation and frequency response, it mixes the two axes,
      and asymmetrically: the dissipation channel picks up ``4 sin^2(theta)`` of the frequency
      channel's power while the frequency channel picks up only ``sin^2(theta)/4`` of the
      dissipation channel's (see :func:`~daq.analysis.noise.from_elec_to_reson` for why they
      differ by 16). Second order in ``theta``, but multiplied by the *ratio* of the two noise
      powers -- so when frequency noise dominates dissipation noise by 20 dB, a ``theta`` of
      0.094 rad makes the dissipation spectrum read about 4.5x high.

    The term is rebuilt analytically from the fit's own scalars rather than interpolated out
    of ``environmental_term``, so it is exact and stays valid for an *f_ro* outside the swept
    span. This is the same construction :func:`fit_notch` uses to build that array, so the two
    are consistent by definition.

    :param fit: A :func:`fit_notch` ``fitresults`` mapping.
    :param readout_freq: Frequency in hertz the single-frequency data was acquired at.
        ``None`` falls back to ``fr``, i.e. assumes the data was taken on resonance.
    :param freq_arr: Sweep frequencies. When given, the ``None`` fallback reproduces the
        historical grid-snapped value read out of ``environmental_term``; otherwise the
        fallback is evaluated analytically at ``fr``.
    :param warn: Whether to warn on the ``None`` fallback. Pass ``False`` where the term
        cannot affect the result (e.g. a display basis that never divides it out).
    :param caller: Name of the calling operation, for the warning text.
    :param remedy: How the caller's user supplies the frequency, for the warning text.
    :param stacklevel: ``warnings.warn`` stack level. The default assumes one wrapper between
        this function and the user's call site; raise it if the chain is deeper.
    :raises ValueError: If *readout_freq* is not positive and finite.
    :returns: The complex environmental term to divide the single-frequency data by.

    """
    fr = float(fit["fr"])

    if readout_freq is None:
        if warn:
            tau = float(fit["environmental_delay"])
            absqc = float(fit["absQc"])
            # Circle diameter in the normalized bases. Called the dip depth loosely, but the
            # two coincide only at phi0 = 0, so the message says diameter.
            diameter = float(fit["Ql"]) / absqc if absqc else float("nan")
            sensitivity = (
                f" For this fit (tau = {tau * 1e9:.1f} ns, circle diameter Ql/|Qc| = "
                f"{diameter:.3f}) that is {4 * np.pi * 1e5 * tau / diameter:.2f} ring radii "
                "per 100 kHz of detuning, while 2*pi*Delta*tau stays well below 1."
                if np.isfinite(diameter) and diameter > 0
                else ""
            )
            warnings.warn(
                f"{caller} is normalizing single-frequency data by the environmental term at "
                f"fr = {fr / 1e9:.6f} GHz, which assumes it was acquired on resonance. "
                f"{remedy} to normalize at the frequency actually used. Data taken off "
                "resonance instead keeps an uncorrected cable-delay phase "
                "2*pi*(f_ro - fr)*tau, which rotates it rigidly about the origin."
                + sensitivity,
                stacklevel=stacklevel,
            )
        if freq_arr is not None:
            env = np.asarray(fit["environmental_term"])
            return complex(env[int(np.argmin(np.abs(np.asarray(freq_arr) - fr)))])
        readout_freq = fr

    validate_readout_freq(readout_freq)

    return complex(
        environmental_term(
            np.array([float(readout_freq)]),
            fit["environmental_amp_norm"],
            fit["environmental_alpha"],
            fit["environmental_delay"],
        )[0]
    )


def validate_readout_freq(readout_freq: float, name: str = "readout_freq") -> float:
    """Validate an acquisition frequency.

    ``nan`` is rejected explicitly: it passes a bare ``<= 0`` test and then silently poisons
    every downstream value with ``nan`` rather than failing where the mistake was made.

    :param readout_freq: Frequency in hertz.
    :param name: Parameter name, for the error message.
    :raises ValueError: If *readout_freq* is not positive and finite.
    :returns: The validated frequency as a float.

    """
    value = float(readout_freq)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {readout_freq!r}")
    return value


def _crop_mask(
    freq_arr: npt.NDArray[np.float64],
    fcrop: Optional[Tuple[float, float]],
) -> npt.NDArray[np.bool_]:
    """Reproduce the frequency mask ``notch_port.autofit`` builds from *fcrop*.

    Recomputed here from the public *fcrop* argument rather than read off the
    port's private ``_fid`` attribute, so this module touches no upstream internals.

    :param freq_arr: Sweep frequencies in Hz.
    :param fcrop: ``(f_min, f_max)`` crop window in Hz, or ``None`` for no crop.
    :return: Boolean mask selecting the fitted points.
    :rtype: numpy.ndarray
    """
    if fcrop is None:
        return np.ones(freq_arr.size, dtype=bool)
    f_min, f_max = fcrop
    return np.logical_and(freq_arr >= f_min, freq_arr <= f_max)


def _calibration_results(
    port: Any,
    freq_arr: npt.NDArray[np.float64],
    resp_arr: npt.NDArray[np.complex128],
    fcrop: Optional[Tuple[float, float]],
    electric_delay: Optional[float],
    guesses: Dict[str, float],
) -> Dict[str, Any]:
    """Recover the calibration scalars and build the environmental-term entries.

    Re-runs :meth:`notch_port.do_calibration` on the same cropped data
    ``autofit()`` used. The routine is deterministic, so this reproduces the exact
    scalars ``autofit()`` computed internally and then discarded.

    :param port: A ``notch_port`` on which ``autofit()`` has already run.
    :param freq_arr: Sweep frequencies in Hz.
    :param resp_arr: Raw complex sweep response.
    :param fcrop: Crop window passed to ``autofit()``, or ``None``.
    :param electric_delay: Fixed electrical delay passed to ``autofit()``, or ``None``.
    :param guesses: Optional ``fr_guess``/``Ql_guess`` forwarded to the calibration.
    :return: Mapping of ``environmental_*`` keys to merge into ``fitresults``.
    :rtype: dict
    """
    fid = _crop_mask(freq_arr, fcrop)
    delay, amp_norm, alpha, _fr, _Ql, a2, frcal = port.do_calibration(
        freq_arr[fid],
        resp_arr[fid],
        ignoreslope=True,
        guessdelay=True,
        fixed_delay=electric_delay,
        **guesses,
    )

    # Every consumer (plot_iq_comparison, from_elec_to_reson) divides by
    # environmental_term alone and ignores environmental_baseline, which is only
    # correct while the baseline is zero. Upstream's get_delay() pins A2 = 0.0 in both
    # branches of its ignoreslope test, so this holds today -- but it is an assumption
    # about someone else's code, so assert it rather than trust it. A non-zero A2 would
    # otherwise silently bias every basis transformation.
    if a2 != 0.0:
        raise ResonatorFitError(
            f"resonator_tools returned a non-zero baseline slope A2={a2!r}. DAQ's "
            "basis transformations divide by environmental_term alone and assume the "
            "baseline is zero, so they would be silently wrong. Either subtract "
            "environmental_baseline in the consumers or pin an older resonator_tools; "
            "daq/analysis/resonator.py needs updating to match."
        )

    env = environmental_term(freq_arr, amp_norm, alpha, delay)
    baseline = a2 * (freq_arr - frcal)

    return {
        "environmental_term": env,
        "environmental_baseline": baseline,
        "environmental_amp_norm": float(amp_norm),
        "environmental_alpha": float(alpha),
        "environmental_delay": float(delay),
        "environmental_A2": float(a2),
        "environmental_frcal": float(frcal),
    }


def _check_consistency(
    port: Any,
    resp_arr: npt.NDArray[np.complex128],
    env: npt.NDArray[np.complex128],
    baseline: npt.NDArray[np.float64],
) -> None:
    """Assert that the recovered environmental term explains upstream's normalization.

    Upstream's ``do_normalization`` is ``z_norm = (z_raw - baseline) / env``. This
    checks the equivalent product form ``z_norm · env == z_raw - baseline``, which is
    well conditioned even where the normalized response dips toward zero on
    resonance.

    Every branch that cannot complete the comparison raises: a check that silently
    skips itself is worse than no check, because callers read "no exception" as
    "validated".

    :param port: A fitted ``notch_port``.
    :param resp_arr: Raw complex sweep response.
    :param env: Reconstructed environmental term.
    :param baseline: Reconstructed baseline term.
    :raises ResonatorFitError: If the reconstruction disagrees with upstream, or if
        the comparison cannot be carried out at all.
    """
    # The consumers divide by env, so a non-finite entry would silently poison every
    # downstream array with NaN rather than raising anywhere.
    if not np.all(np.isfinite(env)):
        raise ResonatorFitError(
            f"Recovered environmental term contains {int(np.sum(~np.isfinite(env)))} "
            "non-finite entries; the resonator fit did not converge to a usable "
            "calibration."
        )

    z_norm = np.asarray(port.z_data)
    residual = np.abs(z_norm * env - (resp_arr - baseline))

    # Normalize by the largest response, not the median: the residual is round-off on
    # the largest term, so scaling it by a much smaller median would flag a correct
    # fit on high-dynamic-range data.
    scale = float(np.max(np.abs(resp_arr)))
    if not np.isfinite(scale) or scale == 0.0:
        raise ResonatorFitError(
            "Cannot validate the environmental term: the sweep response has "
            f"max |resp_arr| = {scale!r}, so there is no scale to compare against."
        )

    worst = float(np.max(residual)) / scale
    # `not (worst <= rtol)` rather than `worst > rtol`, so a NaN residual fails the
    # check instead of slipping through (every comparison with NaN is False).
    if not (worst <= _CONSISTENCY_RTOL):
        raise ResonatorFitError(
            "Recovered environmental term does not reproduce the normalization "
            f"performed by resonator_tools (relative residual {worst:.3e} > "
            f"{_CONSISTENCY_RTOL:.0e}). This usually means the installed "
            "resonator_tools changed its calibration convention; "
            "daq/analysis/resonator.py needs updating to match."
        )


def fit_notch(
    freq_arr: npt.NDArray[np.float64],
    resp_arr: npt.NDArray[np.complex128],
    fcrop: Optional[Tuple[float, float]] = None,
    electric_delay: Optional[float] = None,
    fr_guess: Optional[float] = None,
    Ql_guess: Optional[float] = None,
) -> Any:
    """Fit a notch-port resonator and return the port with an augmented ``fitresults``.

    Runs the stock ``resonator_tools`` ``notch_port.autofit()`` -- upstream owns the
    entire fitting algorithm -- then adds the calibration quantities upstream
    discards. The returned object is a genuine ``notch_port``, so ``z_data_sim``,
    ``z_data_sim_norm``, ``f_data`` and the usual ``fitresults`` keys (``fr``,
    ``Ql``, ``absQc``, ``Qi_dia_corr``, ``phi0``, errors, ...) are all present as
    normal, plus:

    ``environmental_term``
        Complex array ``a·e^{iα}·e^{-2πifτ}`` over *freq_arr*.
    ``environmental_baseline``
        The baseline ``A2·(f - frcal)`` (identically zero with current upstream).
    ``environmental_amp_norm``, ``environmental_alpha``, ``environmental_delay``, ``environmental_A2``, ``environmental_frcal``
        The underlying scalars.

    :param freq_arr: Sweep frequencies in Hz.
    :param resp_arr: Complex sweep response (raw S21).
    :param fcrop: Optional ``(f_min, f_max)`` crop window in Hz restricting the fit.
    :param electric_delay: Optional fixed electrical delay in seconds. When
        ``None`` the delay is fitted.
    :param fr_guess: Optional initial guess for the resonance frequency in Hz.
    :param Ql_guess: Optional initial guess for the loaded quality factor.
    :return: The fitted ``resonator_tools.circuit.notch_port``.
    :raises ImportError: If ``resonator_tools`` is not installed.
    :raises ResonatorFitError: If the environmental term fails its consistency check.
    """
    circuit = _import_circuit()

    freq_arr = np.asarray(freq_arr, dtype=float)
    resp_arr = np.asarray(resp_arr, dtype=complex)

    guesses: Dict[str, float] = {}
    if fr_guess is not None:
        guesses["fr_guess"] = float(fr_guess)
    if Ql_guess is not None:
        guesses["Ql_guess"] = float(Ql_guess)

    port = circuit.notch_port(freq_arr, resp_arr)
    port.autofit(fcrop=fcrop, electric_delay=electric_delay, **guesses)

    extras = _calibration_results(port, freq_arr, resp_arr, fcrop, electric_delay, guesses)
    _check_consistency(
        port, resp_arr, extras["environmental_term"], extras["environmental_baseline"]
    )
    port.fitresults.update(extras)

    return port
