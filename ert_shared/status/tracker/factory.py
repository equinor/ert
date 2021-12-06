from typing import Optional
from ert_shared.status.tracker.legacy import LegacyTracker
from ert_shared.status.tracker.evaluator import EvaluatorTracker
from ert_shared.status.utils import scale_intervals
from ert_shared.feature_toggling import FeatureToggling


def create_tracker(
    model,
    general_interval: int = 5,
    detailed_interval: int = 10,
    num_realizations: Optional[int] = None,
    ee_host: Optional[str] = None,
    ee_port: Optional[int] = None,
    ee_token: Optional[str] = None,
    ee_cert: Optional[str] = None,
):
    """Creates a tracker tracking a @model. The provided model
    is updated either purely event-driven, or in two tiers: @general_interval,
    @detailed_interval. Whether updates are continuous or periodic depends on
    invocation. In the case of periodic updates, setting an interval to <=0
    disables update.

    If @num_realizations is defined, then the intervals are scaled
    according to some affine transformation such that it is tractable to
    do tracking. This only applies to periodic updates.

    If @ee_host_port_tuple then the factory will produce something that can
    track an ensemble evaluator and emit events appropriately.
    """
    if num_realizations is not None:
        general_interval, detailed_interval = scale_intervals(num_realizations)

    if FeatureToggling.is_enabled("ensemble-evaluator"):
        return EvaluatorTracker(
            model,
            host=ee_host,
            port=ee_port,
            general_interval=general_interval,
            detailed_interval=detailed_interval,
            token=ee_token,
            cert=ee_cert,
        )
    return LegacyTracker(
        model,
        general_interval,
        detailed_interval,
    )
