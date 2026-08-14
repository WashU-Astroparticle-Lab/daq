# -*- coding: utf-8 -*-
"""Adapter onto :mod:`qpd.reconstruction`, the parity-reconstruction package.

**No parity algorithm lives here.** ``qpd`` is the lab's model of this exact measurement --
the forward simulator, the transmon theory, and the inverse problem of recovering when the
parity flipped from an I/Q trace -- and this module exists only to hand a
:class:`~daq.measurements.timestream.TimeStream` to it and hand the answer back. The same
relationship :mod:`daq.analysis.resonator` has with ``resonator_tools``: the acquisition layer
owns the data, the analysis package owns the method.

What ``qpd`` provides, and what would otherwise have to be reinvented badly here:

- :func:`qpd.reconstruction.reconstruct_parity_flips_static` -- two-blob emission model fitted
  blind to the trace, then a two-state HMM decode. The discrimination axis, the noise width
  and the flip rate are all learned from the data.
- :func:`qpd.reconstruction.reconstruct_parity_flips_ramped` -- the same for a *swept* gate,
  where the branches move, cross blind, and reset with the ramp.
- :func:`qpd.reconstruction.detect_bursts` -- clusters the flip train against its Poisson
  background with a **trials-corrected** scan statistic.
- ``degenerate`` / ``contrast`` on the result -- whether the fitted model latched onto noise,
  which is the check to run before believing any of the rest.
- :func:`qpd.reconstruction.benchmark_reconstruction` -- replays the fidelity fitted to *your*
  trace into surrogates that do have truth, so the quoted efficiency is the efficiency on the
  data you actually took. Not wrapped here; call it directly on ``ts.signal[:, tone]``.

``qpd`` is an optional dependency, imported lazily so ``import daq`` works without it and the
reconstruction degrades to "not available" rather than to a worse method.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

__all__ = [
    "PROJECTIONS",
    "detect_bursts",
    "project_readout",
    "qpd_available",
    "reconstruct_parity",
]

#: Real projections of a complex readout a parity analysis can be run on.
#:
#: ``"proj"`` is the default and the right one for a two-level readout: parity moves the
#: resonator between two points of the IQ plane, and a fixed axis sees only the ``cos(angle)``
#: of the line joining them -- for a separation that is mostly a phase shift, almost none of
#: it. The axis itself comes from ``qpd``'s blind fit, not from anything here.
PROJECTIONS = ("proj", "abs", "real", "imag")


def qpd_available() -> bool:
    """Return whether the ``qpd`` package can be imported.

    :returns: ``True`` when parity reconstruction is available.

    """
    try:
        import qpd.reconstruction  # noqa: F401
    except ImportError:
        return False
    return True


def _require_qpd():
    """Import :mod:`qpd.reconstruction`, or explain how to get it.

    :raises ImportError: If ``qpd`` is not installed.
    :returns: The :mod:`qpd.reconstruction` module.

    """
    try:
        import qpd.reconstruction as reconstruction
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Parity reconstruction needs the qpd package: pip install "
            "git+https://github.com/WashU-Astroparticle-Lab/qpd.git (or -e a local checkout). "
            "It owns the reconstruction method; daq only hands it the acquisition."
        ) from err
    return reconstruction


def project_readout(
    z: npt.ArrayLike, quantity: str = "proj", *, model: Optional[Any] = None
) -> npt.NDArray[np.floating]:
    """Reduce a complex readout to the real series a spectrum can be taken of.

    ``"proj"`` projects onto the discrimination axis ``qpd`` fits to the cloud
    (:func:`qpd.reconstruction.fit_two_blobs`, whose model carries the projection), which is
    where the whole parity signal lives. The other three are the fixed axes, kept because they
    are what :class:`~daq.measurements.bias_hunt.BiasHunt` ranks by and are sometimes what you
    want to look at; they need no ``qpd``.

    The operating point is kept for the fixed axes -- where the readout sits is worth plotting
    even when only its fluctuation is worth spectrating. The projected axis is centred on the
    cloud's own origin, since it has no meaningful zero of its own.

    :param z: Complex readout samples.
    :param quantity: One of :data:`PROJECTIONS`.
    :param model: A fitted ``qpd`` blob model to project with, e.g. ``result.model`` from a
        reconstruction already run. Saves refitting; ignored unless *quantity* is ``"proj"``.
    :raises ValueError: If *quantity* is not one of :data:`PROJECTIONS`.
    :raises ImportError: If *quantity* is ``"proj"`` and ``qpd`` is not installed.
    :returns: The real-valued series.

    """
    z = np.asarray(z)
    if quantity == "proj":
        if model is None:
            model = _require_qpd().fit_two_blobs(z)
        return np.asarray(model.project(z), dtype=np.float64)
    if quantity == "abs":
        return np.abs(z)
    if quantity == "real":
        return np.real(z)
    if quantity == "imag":
        return np.imag(z)
    raise ValueError(f"quantity must be one of {list(PROJECTIONS)}, got {quantity!r}")


def reconstruct_parity(
    iq: npt.ArrayLike,
    fs: float,
    *,
    ramped: bool = False,
    **kwargs: Any,
) -> Any:
    """Recover the parity-flip times of a readout trace, via ``qpd``.

    Dispatches to the reconstruction that matches how the gate was biased, which is the same
    distinction :attr:`~daq.measurements.timestream.TimeStream.bias_mode` already draws:

    - a **constant** bias holds ``n_g`` fixed, so the readout is two stationary blobs --
      :func:`qpd.reconstruction.reconstruct_parity_flips_static`;
    - a **sawtooth** bias sweeps ``n_g``, so the branches move, cross blind and reset with the
      ramp -- :func:`qpd.reconstruction.reconstruct_parity_flips_ramped`, which models all
      three. The static routine would fit them as noise.

    Both are blind: no device or resonator parameter is supplied.

    **Check ``degenerate`` and ``contrast`` on the result before using it.** A model that has
    latched onto noise fails quietly, and its fidelity estimate stays high while it does.

    :param iq: Complex readout samples of one tone.
    :param fs: Sample rate in hertz -- the *tuned* ``TimeStream.df``.
    :param ramped: Use the swept-gate reconstruction rather than the fixed-gate one.
    :param kwargs: Passed through to the ``qpd`` routine.
    :raises ImportError: If ``qpd`` is not installed.
    :returns: ``qpd``'s reconstruction result.

    """
    reconstruction = _require_qpd()
    routine = (
        reconstruction.reconstruct_parity_flips_ramped
        if ramped
        else reconstruction.reconstruct_parity_flips_static
    )
    return routine(reconstruction.as_complex_trace(np.asarray(iq)), float(fs), **kwargs)


def detect_bursts(
    flip_times: npt.ArrayLike,
    background_rate_hz: float,
    duration_s: float,
    **kwargs: Any,
) -> List[Any]:
    """Find rapid-switching bursts in a reconstructed flip train, via ``qpd``.

    A pass-through to :func:`qpd.reconstruction.detect_bursts` on **its own defaults** --
    nothing is retuned here.

    Worth knowing about one of those defaults, since it changes what the burst list means:
    ``max_p_value`` is ``None``, so every dense cluster is returned with its trials-corrected
    p-value attached rather than filtered on it. At low flip rates that is already a burst
    list -- flips are far apart, so a cluster is genuinely anomalous. At high rates it is a
    list of every coincidence: on a 4 s synthetic trace of pure 200 Hz background it returns
    94 clusters, all of them chance, which collapse to 0 at ``max_p_value=0.01`` while an
    injected 10x burst survives as exactly one span. Read ``burst.p_value``, or pass a
    threshold, when the rate is high.

    :param flip_times: Reconstructed tunnelling times in seconds.
    :param background_rate_hz: Poisson rate of the background telegraph -- normally the
        reconstruction's own ``rate_hz``.
    :param duration_s: Length of the record in seconds, for the trials correction.
    :param kwargs: Passed through (``max_gap``, ``min_flips``, ``max_p_value``).
    :raises ImportError: If ``qpd`` is not installed.
    :returns: ``qpd``'s list of detected bursts.

    """
    return _require_qpd().detect_bursts(
        np.asarray(flip_times, dtype=np.float64),
        float(background_rate_hz),
        duration=float(duration_s),
        **kwargs,
    )


def summarize(result: Any, bursts: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Flatten a ``qpd`` reconstruction into the scalars a plot annotation needs.

    :param result: A ``qpd`` reconstruction result.
    :param bursts: Its detected bursts, if any.
    :returns: Mapping of ``n_flips``, ``rate_hz``, ``contrast``, ``degenerate``,
        ``decoded_fidelity`` and ``n_bursts``.

    """
    return {
        "n_flips": int(np.asarray(result.flip_times).size),
        "rate_hz": float(getattr(result, "rate_hz", np.nan)),
        "contrast": float(getattr(result, "contrast", np.nan)),
        "degenerate": bool(getattr(result, "degenerate", False)),
        "decoded_fidelity": float(getattr(result, "decoded_fidelity", np.nan)),
        "n_bursts": 0 if bursts is None else len(bursts),
    }
