# -*- coding: utf-8 -*-
"""Shared PyVISA plumbing for benchtop instruments.

:class:`VisaInstrument` provides the pieces every SCPI instrument in the lab needs and that
ad-hoc bench scripts routinely get wrong:

- ``pyvisa`` is imported lazily, so ``import daq`` still works on analysis machines with no
  VISA runtime installed.
- The VISA resource is resolved *by name* -- explicit argument, then environment variable,
  then autodiscovery filtered by ``*IDN?`` -- never by grabbing ``list_resources()[0]``,
  which silently picks the wrong box as soon as a second instrument is plugged in.
- Every write is bracketed by a ``SYST:ERR?`` drain and check, so a rejected command raises
  :class:`InstrumentError` instead of being silently swallowed.
- The instrument is a context manager whose ``__exit__`` always forces a safe state, so the
  output cannot be left energised by an exception mid-sequence.

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import get_visa_backend

logger = logging.getLogger(__name__)


class InstrumentError(RuntimeError):
    """Raised when an instrument reports a SCPI error, or cannot be found or opened."""


def _import_pyvisa():
    """Import and return the ``pyvisa`` module, with an actionable error if it is missing.

    :raises InstrumentError: If ``pyvisa`` is not installed.
    :returns: The imported ``pyvisa`` module.

    """
    try:
        import pyvisa
    except ImportError as exc:  # pragma: no cover - depends on the host environment
        raise InstrumentError(
            "pyvisa is required to talk to benchtop instruments but is not installed. "
            "Install it with `pip install pyvisa` (plus the NI-VISA runtime on Windows), "
            "or `pip install daq[instruments]`."
        ) from exc
    return pyvisa


def _get_resource_manager(backend: str):
    """Return a PyVISA resource manager for *backend*.

    :param backend: PyVISA backend spec -- ``""`` for the system (NI-)VISA library, ``"@py"``
        for the pure-Python ``pyvisa-py`` backend.
    :raises InstrumentError: If no VISA implementation can be loaded.
    :returns: A ``pyvisa.ResourceManager``. PyVISA caches these internally, so repeated calls
        with the same backend reuse one manager rather than opening a new one per command --
        the bench scripts this replaces opened one per instrument call.

    """
    pyvisa = _import_pyvisa()
    try:
        return pyvisa.ResourceManager(backend)
    except Exception as exc:
        raise InstrumentError(
            f"Could not open a VISA resource manager (backend={backend!r}): {exc}. "
            "On Windows install the NI-VISA runtime; otherwise set DAQ_VISA_BACKEND='@py' "
            "to use the pure-Python backend."
        ) from exc


def visa_backend_info(backend: Optional[str] = None) -> str:
    """Describe the VISA implementation actually in use.

    The single most useful fact when no instrument is visible. ``pyvisa-py`` cannot enumerate
    USB instruments unless ``pyusb`` and ``libusb`` are installed, and silently lists only
    serial and TCPIP resources when they are not -- which looks identical to an unplugged
    instrument. Knowing which library is loaded distinguishes the two immediately.

    :param backend: PyVISA backend spec. Defaults to ``DAQ_VISA_BACKEND``.
    :returns: A one-line description of the loaded VISA library.

    """
    rm = _get_resource_manager(get_visa_backend() if backend is None else backend)
    library = rm.visalib
    detail = f"{type(library).__name__}: {library}"
    if type(library).__name__ == "PyVisaLibrary":
        detail += (
            " -- pure-Python backend; USB instruments require pyusb and libusb, and are "
            "invisible without them. On Windows prefer NI-VISA (DAQ_VISA_BACKEND='')."
        )
    return detail


def probe_visa_resources(
    backend: Optional[str] = None,
    *,
    timeout_ms: int = 5000,
    read_termination: str = "\n",
    write_termination: str = "\n",
) -> List[Tuple[str, str]]:
    """Ask every visible VISA resource to identify itself.

    The first thing to run when an instrument will not connect: it separates "the resource is
    not visible at all" (a driver or cabling problem) from "it is visible but does not answer"
    (held by another process, or slow) from "it answers but with an unexpected model" (the
    ``IDN_KEYWORDS`` check is what is failing).

    ::

        >>> from daq.instruments import probe_visa_resources
        >>> for resource, idn in probe_visa_resources():
        ...     print(resource, "->", idn)
        USB0::0x0957::0x0407::MY44000531::INSTR -> Agilent Technologies,33220A,MY44000531,2.01
        USB0::0x1313::0x80C8::M01271962::INSTR -> Thorlabs,DC2200,M01271962,1.0

    :param backend: PyVISA backend spec. Defaults to ``DAQ_VISA_BACKEND``.
    :param timeout_ms: How long to wait for each ``*IDN?`` response.
    :param read_termination: Read termination used while probing.
    :param write_termination: Write termination used while probing.
    :raises InstrumentError: If no VISA resource manager can be opened at all.
    :returns: One ``(resource, description)`` pair per visible resource, in discovery order,
        where *description* is the ``*IDN?`` response or an ``"<error: ...>"`` string
        explaining why the resource did not answer.

    """
    rm = _get_resource_manager(get_visa_backend() if backend is None else backend)
    try:
        available = tuple(rm.list_resources())
    except Exception as exc:
        raise InstrumentError(f"Could not list VISA resources: {exc}") from exc

    results: List[Tuple[str, str]] = []
    for resource in available:
        instr = None
        try:
            instr = rm.open_resource(resource)
            instr.timeout = timeout_ms
            instr.read_termination = read_termination
            instr.write_termination = write_termination
            results.append((resource, str(instr.query("*IDN?")).strip()))
        except Exception as exc:
            results.append((resource, f"<error: {type(exc).__name__}: {exc}>"))
        finally:
            if instr is not None:
                try:
                    instr.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
    return results


class VisaInstrument:
    """Base class for SCPI instruments reached over VISA.

    Subclasses set :attr:`IDN_KEYWORDS` (and optionally :attr:`RESOURCE_HINTS`) so the
    instrument can be found without a hardcoded resource string, and override
    :meth:`safe_state` to define what "off" means for that hardware.

    The connection is opened in ``__init__``, so an instrument is usable directly in a
    notebook cell::

        fgen = Agilent33220A()
        fgen.constant(0.5)
        ...
        fgen.close()

    Prefer the context-manager form when running unattended, which guarantees the output is
    turned off even if the body raises::

        with Agilent33220A() as fgen:
            fgen.constant(0.5)

    :param resource: Explicit VISA resource string. When ``None``, falls back to the
        subclass's environment variable and then to autodiscovery.
    :param timeout_ms: VISA I/O timeout in milliseconds.
    :param backend: PyVISA backend spec. Defaults to ``DAQ_VISA_BACKEND``.
    :param transcript_path: Optional path to a text file receiving every SCPI write, query and
        error response, for debugging a misbehaving instrument.

    """

    IDN_KEYWORDS: Tuple[str, ...] = ()
    """Substrings identifying this model in its ``*IDN?`` response (case-insensitive)."""
    RESOURCE_HINTS: Tuple[str, ...] = ()
    """Substrings expected in the VISA resource name, used to skip unrelated resources."""
    READ_TERMINATION: str = "\n"
    WRITE_TERMINATION: str = "\n"
    ERROR_QUERY: str = "SYST:ERR?"
    """Query returning the oldest queued error. Set to ``""`` to disable error checking."""
    MAX_ERROR_READS: int = 50
    """Cap on the error-queue drain loop, so a permanently faulted instrument cannot hang."""
    PROBE_TIMEOUT_MS: int = 5000
    """I/O timeout when asking an unknown resource for ``*IDN?`` during autodiscovery.

    Generous on purpose: an instrument that answers more slowly than this is silently skipped
    and reported as "not found", which is a confusing way to fail. Raise it further if a slow
    GPIB adapter is being missed.

    """

    def __init__(
        self,
        resource: Optional[str] = None,
        *,
        timeout_ms: int = 5000,
        backend: Optional[str] = None,
        transcript_path: Optional[str] = None,
    ) -> None:
        self._backend = get_visa_backend() if backend is None else backend
        self._rm = _get_resource_manager(self._backend)
        self._transcript = open(transcript_path, "a", encoding="utf-8") if transcript_path else None
        self._instr = None
        self._closed = False

        self.resource = self._resolve_resource(resource)
        try:
            self._instr = self._rm.open_resource(self.resource)
            self._instr.timeout = timeout_ms
            self._instr.read_termination = self.READ_TERMINATION
            self._instr.write_termination = self.WRITE_TERMINATION
        except Exception as exc:
            self._close_transcript()
            raise InstrumentError(f"Could not open VISA resource {self.resource!r}: {exc}") from exc

        self._log(f"OPEN   {self.resource}")
        self._drain_errors()
        self.idn = self.query("*IDN?")
        self._log(f"IDN    {self.idn}")

    # ------------------------------------------------------------------ discovery

    @classmethod
    def env_resource(cls) -> Optional[str]:
        """Return the configured resource string for this instrument, or ``None``.

        Subclasses override this to read their own ``DAQ_*_RESOURCE`` setting.

        """
        return None

    def _resolve_resource(self, resource: Optional[str]) -> str:
        """Resolve the VISA resource to open.

        Resolution order is explicit argument, then the subclass's environment variable, then
        autodiscovery by ``*IDN?``.

        :param resource: Explicit resource string, or ``None`` to fall through.
        :returns: The resolved VISA resource string.

        """
        if resource:
            return resource
        configured = self.env_resource()
        if configured:
            return configured
        return self._discover()

    def _backend_note(self) -> str:
        """Return a parenthetical describing the loaded VISA library, for error messages.

        :returns: A short description, or an empty string if it cannot be determined.

        """
        try:
            return f"\nVISA backend in use: {visa_backend_info(self._backend)}"
        except Exception:  # pragma: no cover - diagnostics must never mask the real error
            return ""

    def _discover(self) -> str:
        """Find the one connected resource whose ``*IDN?`` matches :attr:`IDN_KEYWORDS`.

        Every candidate is opened briefly and asked to identify itself. This deliberately
        refuses to guess: with zero or several matches it raises rather than picking the first
        entry, because that failure mode is silent and produces plausible-looking data from
        the wrong instrument.

        :raises InstrumentError: If no resource matches, or if several do.
        :returns: The matching VISA resource string.

        """
        name = type(self).__name__
        try:
            available: Tuple[str, ...] = tuple(self._rm.list_resources())
        except Exception as exc:
            raise InstrumentError(f"Could not list VISA resources: {exc}") from exc

        if not available:
            raise InstrumentError(
                f"No VISA resources are visible at all, so {name} cannot be found. Check that "
                "the instrument is powered on and connected, and that its USB/GPIB driver is "
                "installed (on Windows, confirm it appears in NI MAX)." + self._backend_note()
            )

        candidates = available
        if self.RESOURCE_HINTS:
            hinted = tuple(
                r for r in available if any(h.lower() in r.lower() for h in self.RESOURCE_HINTS)
            )
            if hinted:
                candidates = hinted

        matches: List[str] = []
        # Keep why each candidate was rejected -- a bare "not found" leaves the user with no
        # way to tell a timeout from a busy resource from a genuine model mismatch.
        report: List[str] = []
        for candidate in candidates:
            idn, error = self._probe_idn(candidate)
            if idn is None:
                report.append(f"  {candidate}: no *IDN? response ({error})")
                continue
            if not self.IDN_KEYWORDS or any(k.lower() in idn.lower() for k in self.IDN_KEYWORDS):
                matches.append(candidate)
                report.append(f"  {candidate}: {idn}  <-- matches")
            else:
                report.append(f"  {candidate}: {idn}")

        if len(matches) == 1:
            logger.info("%s autodiscovered at %s", name, matches[0])
            return matches[0]

        env_var = getattr(self, "ENV_VAR", "the resource environment variable")
        listing = "\n".join(report)
        if not matches:
            # No USB or GPIB resource visible at all is a strong signal that the VISA layer
            # is not enumerating instrument buses, rather than that this model is absent.
            buses = [r for r in available if r.upper().startswith(("USB", "GPIB"))]
            if not buses:
                bus_hint = (
                    f"\nNOTE: no USB or GPIB instrument is visible at all -- only "
                    f"{list(available)}, which is a serial/loopback port rather than an "
                    "instrument. Check the obvious thing first: **is the instrument plugged "
                    "in and powered on?** An unplugged USB cable looks exactly like this. If "
                    "it is definitely connected, confirm it appears in NI MAX / Keysight "
                    "Connection Expert, run `python -m pyvisa.info`, and note that VISA can "
                    "fail to enumerate a USB instrument it will still open by explicit "
                    f"address -- so passing resource='...' or setting {env_var} is worth "
                    "trying before blaming the driver."
                )
            else:
                bus_hint = ""
            raise InstrumentError(
                f"No connected instrument identified as {name} -- looked for "
                f"{list(self.IDN_KEYWORDS)} in the *IDN? response of each visible resource:\n"
                f"{listing}\n"
                f"If the instrument is listed above but its *IDN? did not match, pass "
                f"resource='...' explicitly (or set {env_var}) to skip the check. If it is "
                f"listed but did not answer, close any other program holding it (NI MAX, "
                f"another kernel) or raise {name}.PROBE_TIMEOUT_MS. If it is not listed at "
                f"all, the likeliest cause is simply that it is not plugged in or not powered "
                f"on -- check the cable and the front panel before anything else; failing "
                f"that, it is a driver problem rather than a daq one."
                + bus_hint
                + self._backend_note()
            )
        raise InstrumentError(
            f"Found {len(matches)} instruments identifying as {name}:\n{listing}\n"
            f"Pass resource='...' explicitly or set {env_var} to choose one."
        )

    def _probe_idn(self, resource: str) -> Tuple[Optional[str], Optional[str]]:
        """Open *resource* briefly and ask it to identify itself.

        :param resource: VISA resource string to probe.
        :returns: ``(idn, None)`` on success, or ``(None, reason)`` when the resource could not
            be opened or did not answer -- because it belongs to another instrument, is held by
            another process, or is slower to respond than
            :attr:`PROBE_TIMEOUT_MS` allows.

        """
        instr = None
        try:
            instr = self._rm.open_resource(resource)
            instr.timeout = self.PROBE_TIMEOUT_MS
            instr.read_termination = self.READ_TERMINATION
            instr.write_termination = self.WRITE_TERMINATION
            return str(instr.query("*IDN?")).strip(), None
        except Exception as exc:
            logger.debug("Probing %s failed: %s", resource, exc)
            return None, f"{type(exc).__name__}: {exc}"
        finally:
            if instr is not None:
                try:
                    instr.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass

    # ------------------------------------------------------------------ SCPI I/O

    def _require_open(self) -> None:
        """Raise if the instrument has already been closed.

        :raises InstrumentError: If the connection is closed.

        """
        if self._closed or self._instr is None:
            raise InstrumentError(
                f"{type(self).__name__} connection is closed; construct a new instance."
            )

    def _log(self, message: str) -> None:
        """Append *message* to the SCPI transcript, when one is open.

        :param message: Line to record.

        """
        if self._transcript is not None:
            self._transcript.write(message + "\n")
            self._transcript.flush()

    def _raw_query(self, command: str) -> str:
        """Send *command* and return the stripped response without error checking.

        :param command: SCPI query to send.
        :returns: The stripped response text.

        """
        self._require_open()
        return str(self._instr.query(command)).strip()

    @staticmethod
    def _is_no_error(response: str) -> bool:
        """Return whether a ``SYST:ERR?`` response means "no error".

        :param response: Raw error-queue response, e.g. ``'+0,"No error"'``.
        :returns: ``True`` when the response carries error code 0.

        """
        text = response.strip()
        return text.startswith("+0,") or text.startswith("0,") or text in ("+0", "0")

    def _drain_errors(self) -> List[str]:
        """Empty the instrument's error queue so stale faults are not blamed on new commands.

        :returns: The error strings that were cleared.

        """
        if not self.ERROR_QUERY:
            return []
        cleared: List[str] = []
        for _ in range(self.MAX_ERROR_READS):
            try:
                response = self._raw_query(self.ERROR_QUERY)
            except Exception as exc:
                logger.debug("Error-queue read failed: %s", exc)
                break
            if self._is_no_error(response):
                break
            cleared.append(response)
        if cleared:
            self._log(f"DRAIN  {cleared}")
            logger.debug("Cleared stale errors from %s: %s", type(self).__name__, cleared)
        return cleared

    def write(self, command: str) -> None:
        """Send *command* and raise if the instrument rejects it.

        The error queue is drained first so a pre-existing fault is not misattributed to this
        command, then re-read afterwards to catch this command's own error.

        :param command: SCPI command to send.
        :raises InstrumentError: If the instrument queues an error in response.

        """
        self._require_open()
        self._drain_errors()
        self._instr.write(command)
        self._log(f"WRITE  {command}")
        if not self.ERROR_QUERY:
            return
        response = self._raw_query(self.ERROR_QUERY)
        if not self._is_no_error(response):
            self._log(f"  ERROR {response}")
            raise InstrumentError(f"{type(self).__name__} rejected {command!r}: {response}")

    def query(self, command: str) -> str:
        """Send *command* and return its stripped response.

        :param command: SCPI query to send.
        :raises InstrumentError: If the query fails.
        :returns: The stripped response text.

        """
        self._require_open()
        try:
            response = self._raw_query(command)
        except Exception as exc:
            self._log(f"QUERY  {command}\n  EXCEPTION {type(exc).__name__}: {exc}")
            raise InstrumentError(f"{type(self).__name__} query {command!r} failed: {exc}") from exc
        self._log(f"QUERY  {command}\n  -> {response}")
        return response

    def query_float(self, command: str) -> float:
        """Send *command* and return its response parsed as a float.

        :param command: SCPI query to send.
        :raises InstrumentError: If the response is not numeric.
        :returns: The parsed value.

        """
        response = self.query(command)
        try:
            return float(response)
        except ValueError as exc:
            raise InstrumentError(
                f"{type(self).__name__} returned non-numeric response to {command!r}: {response!r}"
            ) from exc

    # ------------------------------------------------------------------ lifecycle

    def safe_state(self) -> None:
        """Put the instrument into its inert state.

        Called by :meth:`close` and on context-manager exit. Subclasses override this to turn
        their output off; the default does nothing.

        """

    def settings(self) -> Dict[str, Any]:
        """Return the instrument's current state as a flat dict of scalars.

        Values are read back from the instrument rather than echoed from what was last
        written, so the record reflects what the hardware actually did. Intended for
        :meth:`daq._base.Base.attach`, which folds this into the HDF5 attributes and MongoDB
        document of a measurement.

        :returns: Flat mapping of setting name to scalar value.

        """
        return {"resource": self.resource, "idn": self.idn}

    def _close_transcript(self) -> None:
        """Close the SCPI transcript file, if one is open."""
        if self._transcript is not None:
            try:
                self._transcript.close()
            finally:
                self._transcript = None

    def close(self) -> None:
        """Force the instrument into its safe state and release the VISA resource.

        Idempotent, and never raises: a failure to reach the instrument during cleanup is
        logged rather than propagated, so it cannot mask the original error when called from
        an exception path.

        """
        if self._closed:
            return
        instr = self._instr
        # safe_state() talks to the instrument, so it has to run while the connection is
        # still considered open -- marking the object closed first would make _require_open()
        # reject the very write that turns the output off.
        if instr is not None:
            try:
                self.safe_state()
            except Exception as exc:
                logger.warning("%s safe_state() failed on close: %s", type(self).__name__, exc)
        self._closed = True
        self._instr = None
        if instr is not None:
            try:
                instr.close()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("%s resource close failed: %s", type(self).__name__, exc)
        self._log("CLOSE")
        self._close_transcript()

    def __enter__(self) -> "VisaInstrument":
        """Return self for use as a context manager.

        :returns: This instrument.

        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Force the safe state and close on context exit, including on exception."""
        self.close()

    def __repr__(self) -> str:
        """Return a debugging representation naming the resource.

        :returns: Representation string.

        """
        state = "closed" if self._closed else "open"
        return f"<{type(self).__name__} {self.resource!r} ({state})>"
