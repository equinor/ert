from ert.ensemble_evaluator import EvaluatorTracker


def create_tracker(
    model,
    ee_con_info,
):
    """Creates a tracker tracking a @model. The provided model
    is updated purely event-driven.

    If @ee_host_port_tuple then the factory will produce something that can
    track an ensemble evaluator and emit events appropriately.
    """

    return EvaluatorTracker(
        model,
        ee_con_info,
    )
