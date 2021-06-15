import re

import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
import ert_shared.status.entity.state as state
from ert_shared.ensemble_evaluator.entity import serialization
from ert_shared.ensemble_evaluator.narratives.narrative import (
    Consumer,
    EventDescription,
    Provider,
    ReMatch,
)


monitor_failing_evaluation = (
    Consumer("Monitor")
    .forms_narrative_with(
        Provider("Ensemble Evaluator"),
    )
    .given("a failing evaluation")
    .responds_with("start then failure")
    .cloudevents_in_order(
        [
            EventDescription(
                type_=identifiers.EVTYPE_EE_SNAPSHOT,
                source=ReMatch(re.compile(r"/ert/ee/ee."), "/ert/ee/ee-0"),
            ),
            EventDescription(
                type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                source=ReMatch(re.compile(r"/ert/ee/ee."), "/ert/ee/ee-0"),
                data={identifiers.STATUS: state.ENSEMBLE_STATE_STARTED},
            ),
            EventDescription(
                type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                source=ReMatch(re.compile(r"/ert/ee/ee."), "/ert/ee/ee-0"),
                data={identifiers.STATUS: state.ENSEMBLE_STATE_FAILED},
            ),
        ]
    )
    .receives("done")
    .cloudevents_in_order(
        [
            EventDescription(
                type_=identifiers.EVTYPE_EE_USER_DONE,
                source=ReMatch(re.compile(r"/ert/monitor/."), "/ert/monitor/007"),
            ),
        ]
    )
    .responds_with("termination")
    .cloudevents_in_order(
        [
            EventDescription(
                type_=identifiers.EVTYPE_EE_TERMINATED,
                source=ReMatch(re.compile(r"/ert/ee/ee."), "/ert/ee/ee-0"),
            ),
        ]
    )
    .with_marshaller("application/json", serialization.evaluator_marshaller)
    .with_unmarshaller("application/json", serialization.evaluator_unmarshaller)
    .with_name("Monitor Failed Evaluation")
)
