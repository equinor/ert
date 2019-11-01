import time

from ert_shared.tracker.base import BaseTracker


class BlockingTracker(BaseTracker):
    """The BlockingTracker provide tracking for non-qt consumers."""

    def __init__(self, model, tick_interval, general_interval, detailed_interval):
        """See create_tracker for details."""
        super(BlockingTracker, self).__init__(model)
        self._tick_interval = tick_interval
        self._general_interval = general_interval
        self._detailed_interval = detailed_interval

    def track(self):
        """Tracks the model in a blocking manner. This method is a generator
        and will yield events at the appropriate times."""
        tick = 0
        while not self._model.isFinished():
            if self._tick_interval and tick % self._tick_interval == 0:
                yield self._tick_event()
            if self._general_interval and tick % self._general_interval == 0:
                yield self._general_event()
            if self._detailed_interval and tick % self._detailed_interval == 0:
                yield self._detailed_event()

            tick += 1
            time.sleep(1)

        # Simulation done, emit final updates
        if self._tick_interval > 0:
            yield self._tick_event()

        if self._general_interval > 0:
            yield self._general_event()

        if self._detailed_interval > 0:
            yield self._detailed_event()

        yield self._end_event()

    def stop(self):
        raise NotImplementedError("cannot stop BlockingTracker")
