from ert_shared.tracker.base import BaseTracker


class QTimerTracker(BaseTracker):
    """The QTimerTracker provide tracking for Qt-based consumers using
    QTimer."""

    def __init__(
        self,
        model,
        qtimer_cls,
        general_interval,
        detailed_interval,
        event_handler,
        ee_monitor_connection_details=None,
    ):
        """See create_tracker for details."""
        super().__init__(model, ee_monitor_connection_details)
        self._qtimers = []
        self._event_handler = event_handler

        if general_interval > 0:
            timer = qtimer_cls()
            timer.setInterval(general_interval * 1000)
            timer.timeout.connect(self._general)
            self._qtimers.append(timer)

        if detailed_interval > 0:
            timer = qtimer_cls()
            timer.setInterval(detailed_interval * 1000)
            timer.timeout.connect(self._detailed)
            self._qtimers.append(timer)

    def _general(self):
        self._event_handler(self._general_event())

        # Check for completion. If Complete, emit all events including a final
        # EndEvent. All timers stop after that.
        if self.is_finished():
            self._detailed()
            self._end()

            self.stop()

    def _detailed(self):
        self._event_handler(self._detailed_event())

    def _end(self):
        self._event_handler(self._end_event())

    def track(self):
        for timer in self._qtimers:
            timer.start()

    def stop(self):
        for timer in self._qtimers:
            timer.stop()
