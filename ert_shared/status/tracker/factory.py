from ert_shared.status.tracker.evaluator import EvaluatorTracker


def create_tracker(
    model,
    ee_config=None,
):
    """Creates a tracker tracking a @model. The provided model
    is updated purely event-driven.

    If @ee_host_port_tuple then the factory will produce something that can
    track an ensemble evaluator and emit events appropriately.
    """

    return EvaluatorTracker(
        model,
        ee_config.host,
        ee_config.port,
        token=ee_config.token,
        cert=ee_config.cert,
    )
