"""Software-started DC2200 pulses shared by time streams and QC traces."""

from contextlib import contextmanager
import logging
import sys
import time
from typing import Any, Callable, Dict, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..instruments import DC2200


class SoftwareLEDMixin:
    """Opt-in live LED attachment; ordinary ``Base.attach`` remains a snapshot."""

    def attach_led(self, led: Optional["DC2200"]) -> None:
        """Attach a configured, disabled DC2200 pulse engine to each acquisition.

        Configure the caller-owned driver with ``configure_pulse(..., output=False)``
        first. Each run refreshes its settings, starts it immediately before
        ``get_pixels()``, and disables it before saving, including on failure. The
        VISA session stays open. Repeated runs restart the pulse train.

        This is software synchronization: USB latency separates the start command
        from sample zero. Saved timestamps describe host calls, not measured light
        emission or Presto sample times. Locate pulses in each raw record before
        aligning them. Consider ``discard_start_ms=0`` to retain the start region.

        ``attach(led=led)`` still only records settings, including for TTL workflows.
        This method does not change digital trigger routing or configure pulse times.

        :param led: Open DC2200 in PULS mode with output disabled, or ``None`` to
            detach it and clear the LED metadata. Loading a file restores metadata
            only; explicitly reattach a driver (or detach) before acquiring again.
        :raises ValueError: If the driver is not in pulse mode or is already enabled.
        """
        settings = None if led is None else led.settings()
        if settings is not None:
            self._validate_led(settings)
        self._clear_led_metadata()
        self._software_led = led
        if settings is not None:
            self.attach(led={**settings, "synchronization": "software"})

    @staticmethod
    def _validate_led(settings: Dict[str, Any]) -> None:
        if not str(settings.get("mode", "")).upper().startswith("PULS"):
            raise ValueError("attach_led requires DC2200 pulse mode; use configure_pulse first")
        if settings.get("output", True):
            raise ValueError("attach_led requires the LED output disabled (output=False)")

    def _clear_led_metadata(self) -> None:
        for key in list(self.__dict__):
            if key.startswith("led_"):
                del self.__dict__[key]
        self.__dict__.get("_attached_keys", {}).pop("led", None)

    def _get_software_led(self) -> Optional["DC2200"]:
        led = getattr(self, "_software_led", None)
        if led is None and getattr(self, "led_synchronization", None) == "software":
            raise ValueError("Reattach a live LED with attach_led(led), or call attach_led(None)")
        return led

    @contextmanager
    def _software_led_acquisition(self) -> Iterator[Optional[Callable[[], None]]]:
        led = self._get_software_led()
        if led is None:
            yield None
            return

        try:
            # Do this before Presto setup and snapshot queries, on every run.
            led.output = False
            settings = led.settings()
            self._validate_led(settings)
            self._clear_led_metadata()
            self.attach(led={**settings, "synchronization": "software"})

            def start() -> None:
                self.led_start_command_unix = time.time()
                started = time.monotonic()
                led.output = True
                self.led_start_completed_unix = time.time()
                self.led_start_command_duration_s = time.monotonic() - started

            yield start
        finally:
            failing = sys.exc_info()[0] is not None
            try:
                led.output = False
            except Exception:
                if not failing:
                    raise
                logging.getLogger(__name__).exception("Failed to disable LED during cleanup")
