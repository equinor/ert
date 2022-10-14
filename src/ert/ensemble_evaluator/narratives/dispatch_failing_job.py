from ert.ensemble_evaluator import identifiers

from .narrative import Consumer, EventDescription, Provider


def dispatch_failing_job():
    return (
        Consumer("Dispatch")
        .forms_narrative_with(Provider("Ensemble Evaluator"))
        .given("small ensemble")
        .receives("a job eventually fails")
        .cloudevents_in_order(
            [
                EventDescription(
                    type_=identifiers.EVTYPE_ENSEMBLE_STARTED,
                    source="/ert/ee/0",
                ),
                EventDescription(
                    type_=identifiers.EVTYPE_FM_STEP_RUNNING,
                    source="/ert/ee/0/real/0/stage/0/step/0",
                ),
                EventDescription(
                    type_=identifiers.EVTYPE_FM_JOB_RUNNING,
                    source="/ert/ee/0/real/0/stage/0/step/0/job/0",
                    data={identifiers.CURRENT_MEMORY_USAGE: 1000},
                ),
                EventDescription(
                    type_=identifiers.EVTYPE_FM_STEP_RUNNING,
                    source="/ert/ee/0/real/1/stage/0/step/0",
                ),
                EventDescription(
                    type_=identifiers.EVTYPE_FM_JOB_RUNNING,
                    source="/ert/ee/0/real/1/stage/0/step/0/job/0",
                    data={identifiers.CURRENT_MEMORY_USAGE: 2000},
                ),
                EventDescription(
                    type_=identifiers.EVTYPE_FM_JOB_SUCCESS,
                    source="/ert/ee/0/real/0/stage/0/step/0/job/0",
                    data={identifiers.CURRENT_MEMORY_USAGE: 2000},
                ),
                EventDescription(
                    type_=identifiers.EVTYPE_FM_JOB_FAILURE,
                    source="/ert/ee/0/real/1/stage/0/step/0/job/0",
                    data={identifiers.ERROR_MSG: "error"},
                ),
                EventDescription(
                    type_=identifiers.EVTYPE_FM_STEP_FAILURE,
                    source="/ert/ee/0/real/1/stage/0/step/0",
                ),
                EventDescription(
                    type_=identifiers.EVTYPE_FM_STEP_SUCCESS,
                    source="/ert/ee/0/real/0/stage/0/step/0",
                ),
                EventDescription(
                    type_=identifiers.EVTYPE_ENSEMBLE_STOPPED,
                    source="/ert/ee/0",
                ),
            ]
        )
        .with_name("Dispatch Failing Job")
    )
