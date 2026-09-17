# -*- coding: utf-8 -*-
"""Per-period digitisation of a gate-ramped readout.

A sawtooth-biased device traces one quantum-capacitance (QC) curve per gate period. When the
island is poisoned the curve is suppressed, so the spread of the samples *within* one period
is a per-period detector of that state: large spread, normal QC; small spread, suppressed. This
module owns that reduction -- the principal axis the samples are projected on, the
per-period standard deviation, the two-mode threshold between the populations, and the
resulting state sequence -- so the notebooks that consume it agree on every definition.

It imports only numpy, so it is usable on an analysis machine with no ``presto`` install.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

__all__ = ["principal_axis"]


def principal_axis(
    z: npt.ArrayLike,
) -> Tuple[npt.NDArray[np.floating], complex]:
    """Return the direction in the I/Q plane along which *z* spreads the most, and its mean.

    The largest-eigenvalue eigenvector of the *centred* I/Q covariance. The divisor of the
    covariance (``N`` or ``N - ddof``) scales both eigenvalues alike and so does not move the
    eigenvector, which is why no ``ddof`` appears here. An eigenvector's sign is arbitrary; it
    is fixed so the larger component is positive, purely so a saved axis is deterministic.
    When the two eigenvalues are equal the direction is ambiguous, but the spread along it is
    not.

    This is the same eigenvector ``qpd.reconstruction.estimate_direction`` returns (which adds
    a noise estimate from the minor axis); it lives here so the measurement layer can use it
    without ``qpd``. It is *not* ``qpd``'s two-blob discrimination axis: a ramped QC trace is
    a continuous curve, not two clusters, so its covariance is the right object.

    :param z: Complex samples, any shape; flattened.
    :raises ValueError: If the covariance is not finite or there are fewer than two samples.
    :returns: ``(axis, origin)`` -- the ``[I, Q]`` unit vector and the complex mean the
        samples were centred on. Project with
        ``axis[0] * (z - origin).real + axis[1] * (z - origin).imag``.

    """
    z = np.asarray(z, dtype=np.complex128).ravel()
    if z.size < 2:
        raise ValueError(f"principal_axis needs at least two samples, got {z.size}")
    origin = complex(z.mean())
    centered = z - origin
    iq = np.vstack((centered.real, centered.imag))
    covariance = (iq @ iq.T) / z.size
    if not np.all(np.isfinite(covariance)):
        raise ValueError("The I/Q covariance is not finite")
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return axis, origin


def project(
    z: npt.ArrayLike, axis: npt.ArrayLike, origin: complex = 0.0
) -> npt.NDArray[np.floating]:
    """Project complex samples onto an ``[I, Q]`` axis about *origin*.

    :param z: Complex samples, any shape.
    :param axis: ``[I, Q]`` unit vector, e.g. from :func:`principal_axis`.
    :param origin: Complex point the projection is measured from.
    :returns: Real array with the shape of *z*.

    """
    z = np.asarray(z, dtype=np.complex128) - origin
    axis = np.asarray(axis, dtype=np.float64)
    return axis[0] * z.real + axis[1] * z.imag


__all__.append("project")
