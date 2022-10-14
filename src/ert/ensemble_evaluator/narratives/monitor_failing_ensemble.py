import re

from ert import serialization
from ert.ensemble_evaluator import identifiers, state

from .narrative import Consumer, EventDescription, Provider, ReMatch


def monitor_failing_ensemble():
    return (
        Consumer("Monitor")
        .forms_narrative_with(Provider("Ensemble Evaluator"))
        .given(
            "Ensemble with 2 reals, with 2 steps each, "
            "with 2 jobs each, job 1 in real 1 fails"
        )
        .responds_with("Snapshot")
        .cloudevents_in_order(
            [
                EventDescription(
                    type_=identifiers.EVTYPE_EE_SNAPSHOT,
                    source=ReMatch(re.compile(".*"), ""),
                )
            ]
        )
        .responds_with("Starting")
        .cloudevents_in_order(
            [
                EventDescription(
                    type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                    source=ReMatch(re.compile(".*"), ""),
                    data={identifiers.STATUS: state.ENSEMBLE_STATE_STARTED},
                )
            ]
        )
        .responds_with("Failure")
        .repeating_unordered_events(
            [],
            terminator=EventDescription(
                type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                source=ReMatch(re.compile(".*"), ""),
                data={
                    identifiers.REALS: {
                        "1": {
                            identifiers.STEPS: {
                                "0": {
                                    identifiers.JOBS: {
                                        "1": {
                                            identifiers.STATUS: state.JOB_STATE_FAILURE
                                        }
                                    }
                                }
                            },
                            identifiers.STATUS: state.REALIZATION_STATE_FAILED,
                        },
                    }
                },
            ),
        )
        .responds_with("Stopped")
        .cloudevents_in_order(
            [
                EventDescription(
                    type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                    source=ReMatch(re.compile(".*"), ""),
                    data={identifiers.STATUS: state.ENSEMBLE_STATE_STOPPED},
                )
            ]
        )
        .receives("Monitor done")
        .cloudevents_in_order(
            [
                EventDescription(
                    type_=identifiers.EVTYPE_EE_USER_DONE,
                    source=ReMatch(re.compile(".*"), ""),
                )
            ]
        )
        .responds_with("Termination")
        .cloudevents_in_order(
            [
                EventDescription(
                    type_=identifiers.EVTYPE_EE_TERMINATED,
                    source=ReMatch(re.compile(".*"), ""),
                )
            ]
        )
        .with_unmarshaller("application/json", serialization.evaluator_unmarshaller)
        .with_marshaller("application/json", serialization.evaluator_marshaller)
        .with_name("Monitor Failing Ensemble")
    )
