import time

from ert_shared.tracker.base import BaseTracker


class BlockingTracker(BaseTracker):
    """The BlockingTracker provide tracking for non-qt consumers."""

    def __init__(
        self,
        model,
        general_interval,
        detailed_interval,
        ee_monitor_connection_details=None,
    ):
        """See create_tracker for details."""
        super().__init__(model, ee_monitor_connection_details)
        self._general_interval = general_interval
        self._detailed_interval = detailed_interval

    def track(self):
        """Tracks the model in a blocking manner. This method is a generator
        and will yield events at the appropriate times."""
        tick = 0
        while not self.is_finished():
            if self._general_interval and tick % self._general_interval == 0:
                yield self._general_event()
            if self._detailed_interval and tick % self._detailed_interval == 0:
                yield self._detailed_event()

            tick += 1
            time.sleep(1)

        # Simulation done, emit final updates
        if self._general_interval > 0:
            yield self._general_event()

        if self._detailed_interval > 0:
            yield self._detailed_event()

        yield self._end_event()

    def stop(self):
        raise NotImplementedError("cannot stop BlockingTracker")
