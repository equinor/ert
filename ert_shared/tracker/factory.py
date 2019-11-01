from ert_shared.tracker.blocking import BlockingTracker
from ert_shared.tracker.qt import QTimerTracker
from ert_shared.tracker.utils import scale_intervals


def create_tracker(
    model,
    tick_interval=1,
    general_interval=5,
    detailed_interval=10,
    qtimer_cls=None,
    event_handler=None,
    num_realizations=None,
):
    """Creates a tracker tracking a @model. The provided model
    is updated in three tiers: @tick_interval,
    @general_interval, @detailed_interval. Setting any
    interval to <=0 disables update.

    Should a @qtimer_cls be defined, the Qt event loop will be used for
    tracking. @event_handler must then be defined.

    If @num_realizations is defined, then the intervals are scaled
    according to some affine transformation such that it is tractable to
    do tracking.
    """
    if num_realizations is not None:
        general_interval, detailed_interval = scale_intervals(num_realizations)

    if qtimer_cls:
        if not event_handler:
            raise ValueError(
                "event_handler must be defined if" + "qtimer_cls is defined"
            )
        tracker = QTimerTracker(
            model,
            qtimer_cls,
            tick_interval,
            general_interval,
            detailed_interval,
            event_handler,
        )
    else:
        tracker = BlockingTracker(
            model, tick_interval, general_interval, detailed_interval
        )
    return tracker
