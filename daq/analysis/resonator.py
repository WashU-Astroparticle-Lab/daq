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

:func:`fit_notch` verifies that identity on every call, so if a future upstream
release changes the calibration convention this fails loudly instead of returning a
subtly wrong basis transformation.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from resonator_tools import circuit

__all__ = [
    "ResonatorFitError",
    "environmental_term",
    "fit_notch",
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

    :return: ``True`` when :mod:`resonator_tools` is importable.
    :rtype: bool
    """
    try:
        import resonator_tools  # noqa: F401
    except ImportError:
        return False
    return True


def _import_circuit() -> "circuit":
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

    :param port: A fitted ``notch_port``.
    :param resp_arr: Raw complex sweep response.
    :param env: Reconstructed environmental term.
    :param baseline: Reconstructed baseline term.
    :raises ResonatorFitError: If the reconstruction disagrees with upstream.
    """
    z_norm = np.asarray(port.z_data)
    residual = np.abs(z_norm * env - (resp_arr - baseline))
    scale = float(np.median(np.abs(resp_arr)))
    if scale == 0.0 or not np.isfinite(scale):
        return
    worst = float(np.max(residual)) / scale
    if worst > _CONSISTENCY_RTOL:
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
