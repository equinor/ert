from ert_shared.tracker.base import BaseTracker


class QTimerTracker(BaseTracker):
    """The QTimerTracker provide tracking for Qt-based consumers using
    QTimer."""

    def __init__(
        self,
        model,
        qtimer_cls,
        tick_interval,
        general_interval,
        detailed_interval,
        event_handler,
    ):
        """See create_tracker for details."""
        super(QTimerTracker, self).__init__(model)
        self._qtimers = []
        self._event_handler = event_handler

        if tick_interval <= 0:
            raise ValueError(
                "the qt driven tracker requires ticks in order "
                + "to check for completion"
            )

        timer = qtimer_cls()
        timer.setInterval(tick_interval * 1000)
        timer.timeout.connect(self._tick)
        self._qtimers.append(timer)

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

    def _tick(self):
        self._event_handler(self._tick_event())

        # Check for completion. If Complete, emit all events including a final
        # EndEvent. All timers stop after that.
        if self._model.isFinished():
            self._general()
            self._detailed()
            self._end()

            self.stop()

    def _general(self):
        self._event_handler(self._general_event())

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
