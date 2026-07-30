# -*- coding: utf-8 -*-
"""Presto digital-output trigger routing.

The Presto gates external instruments through four digital output ports, and presto's
:meth:`presto.lockin.Lockin.set_trigger_out` takes one state **per port** -- element *i*
configures port *i+1*. Which instrument an acquisition actually gates is therefore a pure
wiring fact: ``[1]`` fires whatever is on port 1, ``[0, 1]`` whatever is on port 2.

That fact used to live in prose (and, in :class:`~daq.measurements.qc_trace.QCTrace`, in a
hardcoded ``external_trigger=True``), so a rig wired differently from the lab's default
produced a *silent* failure -- the gated instrument never fires and the acquisition looks
like a dead detector rather than a misconfiguration. This module makes the wiring a property
of the instrument instead: each driver carries a :attr:`~daq.instruments._visa.VisaInstrument.
trigger_port`, and :func:`trigger_for` turns any set of instruments into the states list a
``TimeStream`` wants::

    with Agilent33220A() as bias, DC2200() as led:
        ts = TimeStream(..., external_trigger=trigger_for(bias, led))

This module deliberately imports nothing but :mod:`numpy`, so instrument code can use it on a
machine with no ``presto`` install.

"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

TriggerAny = Union[bool, Sequence[int], npt.NDArray[np.integer]]

#: Presto exposes four digital output ports. presto packs the per-port trigger
#: states two bits at a time into a single uint8 (``Msg__LckSetDf``), and
#: ``Pulsed.output_digital_marker`` bounds its ports to 1-4, so a longer states
#: list would silently overflow the wire format rather than address a 5th port.
MAX_TRIGGER_PORTS = 4


def validate_trigger_port(port: Optional[int], *, name: str = "trigger_port") -> Optional[int]:
    """Check that *port* names a Presto digital output port.

    :param port: Port number, 1-based, or ``None`` for "not wired to the trigger".
    :param name: Name to use in the error message.
    :raises ValueError: If *port* is not an integer in ``1 .. MAX_TRIGGER_PORTS``.
    :returns: The port as a plain ``int``, or ``None``.

    """
    if port is None:
        return None
    if isinstance(port, bool) or not isinstance(port, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer port number between 1 and {MAX_TRIGGER_PORTS} "
            f"(or None), got {port!r}"
        )
    port = int(port)
    if not 1 <= port <= MAX_TRIGGER_PORTS:
        raise ValueError(
            f"{name}={port} is not a Presto digital output port; they are numbered 1 to "
            f"{MAX_TRIGGER_PORTS}"
        )
    return port


def trigger_port_of(source: Any) -> int:
    """Return the digital output port *source* is wired to.

    :param source: A port number, or an instrument carrying a ``trigger_port`` attribute
        (every :class:`~daq.instruments._visa.VisaInstrument` does).
    :raises ValueError: If the port is out of range, or the instrument's ``trigger_port`` is
        ``None`` (nothing declares where it is wired).
    :raises TypeError: If *source* is neither a port number nor an instrument.
    :returns: The port number, 1-based.

    """
    if isinstance(source, (int, np.integer)) and not isinstance(source, bool):
        return validate_trigger_port(int(source))  # type: ignore[return-value]
    if hasattr(source, "trigger_port"):
        port = source.trigger_port
        if port is None:
            raise ValueError(
                f"{type(source).__name__}.trigger_port is None, so there is no way to know "
                "which Presto digital output port gates it. Set it on the instrument "
                "(trigger_port=...), via its DAQ_*_TRIGGER_PORT environment variable, or pass "
                "the port number directly."
            )
        name = f"{type(source).__name__}.trigger_port"
        return validate_trigger_port(port, name=name)  # type: ignore[return-value]
    raise TypeError(
        "expected a Presto digital output port number or an instrument with a trigger_port "
        f"attribute, got {type(source).__name__}"
    )


def trigger_for(*sources: Any, state: int = 1) -> npt.NDArray[np.int64]:
    """Build the per-port trigger states that gate *sources*.

    The whole point of this helper is that a measurement should never have to name a port
    number. Ask for the instruments you want gated and the wiring each one carries decides
    the rest::

        trigger_for(bias)            # -> [1]      (33220A on port 1)
        trigger_for(led)             # -> [0, 1]   (DC2200 on port 2)
        trigger_for(bias, led)       # -> [1, 1]   (both, fired together)
        trigger_for(3)               # -> [0, 0, 1]  explicit port, no instrument needed

    Note that presto's ``delay`` and ``width`` are **global**, not per port, so instruments
    gated in the same acquisition are gated identically; independent timing needs separate
    acquisitions.

    :param sources: Instruments (anything with a ``trigger_port``) and/or port numbers.
    :param state: Trigger state to set for each port: ``1`` fires on every lock-in window,
        ``2`` on every sum window.
    :raises ValueError: If *state* is not ``1`` or ``2``, or a port is unknown or out of range.
    :raises TypeError: If a source is neither an instrument nor a port number.
    :returns: The states array, ready to pass as ``TimeStream(external_trigger=...)``. Empty
        when no source is given, i.e. no port is triggered.

    """
    if state not in (1, 2):
        raise ValueError(
            f"state must be 1 (every lock-in window) or 2 (every sum window), got {state}"
        )
    ports = [trigger_port_of(source) for source in sources]
    if not ports:
        return np.zeros(0, dtype=np.int64)
    states = np.zeros(max(ports), dtype=np.int64)
    for port in ports:
        states[port - 1] = state
    return states


def resolve_trigger_states(external_trigger: TriggerAny) -> npt.NDArray[np.int64]:
    """Normalise an ``external_trigger`` argument to presto's per-port states list.

    presto's :meth:`presto.lockin.Lockin.set_trigger_out` takes one state **per digital
    output port**: element *i* configures port *i+1*, where ``0`` means no trigger, ``1``
    triggers on every lock-in window and ``2`` on every sum window. Which instrument is
    gated is therefore purely a question of what is wired to which port::

        [1]     or [1, 0]  -- port 1 only  (e.g. a gated Agilent 33220A ramp)
        [0, 1]             -- port 2 only  (e.g. a DC2200 in TTL mode)
        [1, 1]             -- both ports, fired together

    ``True`` is accepted as a shorthand for ``[1]``, which is what every caller written
    before per-port routing existed meant, and ``False`` for "no trigger at all".
    :func:`trigger_for` builds the same list from the instruments themselves, which is
    preferable to writing port numbers into a measurement by hand.

    Note that ``delay`` and ``width`` are **global**, not per port -- presto sends them as
    a single pair alongside ``df`` -- so ports enabled in the same acquisition necessarily
    share their timing. Independent timing needs separate acquisitions.

    :param external_trigger: ``True``/``False``, or a sequence of per-port states.
    :return: The resolved states as an integer array; empty when no port is triggered.
    :raises ValueError: If a state is outside ``{0, 1, 2}``, is non-integral, or more than
        :data:`MAX_TRIGGER_PORTS` ports are addressed.

    """
    if isinstance(external_trigger, (bool, np.bool_)):
        return np.array([1], dtype=np.int64) if external_trigger else np.zeros(0, np.int64)
    raw = np.atleast_1d(np.asarray(external_trigger))
    # Reject non-integral input rather than truncating it. A state computed as
    # 0.999... would otherwise silently become 0 -- the trigger never fires and
    # the acquisition looks like a dead detector rather than a misconfiguration.
    if np.issubdtype(raw.dtype, np.floating):
        if not np.all(raw == np.round(raw)):
            raise ValueError(f"external_trigger states must be whole numbers, got {raw.tolist()}")
    elif not np.issubdtype(raw.dtype, np.integer) and not np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(
            "external_trigger must be a bool or a sequence of integer per-port states, "
            f"got dtype {raw.dtype}"
        )
    states = raw.astype(np.int64)
    if states.ndim != 1:
        raise ValueError(f"external_trigger must be one-dimensional, got shape {states.shape}")
    if states.size > MAX_TRIGGER_PORTS:
        raise ValueError(
            f"external_trigger addresses {states.size} ports, but presto has only "
            f"{MAX_TRIGGER_PORTS} digital output ports; a longer list overflows the "
            "uint8 that carries the packed states"
        )
    if np.any((states < 0) | (states > 2)):
        raise ValueError(
            "external_trigger states must be 0 (off), 1 (every lock-in window) or "
            f"2 (every sum window), got {states.tolist()}"
        )
    return states


def describe_trigger_states(states: TriggerAny) -> str:
    """Render trigger states as a human-readable port list, for messages and notes.

    :param states: Anything :func:`resolve_trigger_states` accepts.
    :returns: E.g. ``"port 1"``, ``"ports 1, 2"`` or ``"no port"``.

    """
    resolved = resolve_trigger_states(states)
    ports = [str(ii + 1) for ii, value in enumerate(resolved.tolist()) if value]
    if not ports:
        return "no port"
    return f"port{'s' if len(ports) > 1 else ''} {', '.join(ports)}"
