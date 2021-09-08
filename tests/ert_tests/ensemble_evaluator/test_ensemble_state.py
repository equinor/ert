import datetime
import json
from collections import deque
from typing import Generator, Union

import ert_shared.ensemble_evaluator.entity.identifiers as ids
import ert_shared.status.entity.state as state
import pytest
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.ensemble.state import (
    _ENSEMBLE_STARTED,
    _ENSEMBLE_STOPPED,
    _ENSEMBLE_UNKNOWN,
    _FSM,
    _FSM_UNKNOWN,
    _JOB_FAILURE,
    _JOB_FINISHED,
    _JOB_START,
    _REALIZATION_FINISHED,
    _STEP_PENDING,
    _STEP_SUCCESS,
    _STEP_UNKNOWN,
    _STEP_RUNNING,
    EnsembleFSM,
    IllegalTransition,
    RealizationFSM,
    _Encoder,
    _State,
    _StateChange,
    _Transition,
)

UNKNOWN = _State("UNKNOWN")
RUNNING = _State("RUNNING")
SUCCESS = _State("SUCCESS")
FAILURE = _State("FAILURE")


def test_transition():
    n: _FSM = _FSM("", "/0")
    transition = _Transition(_FSM_UNKNOWN, RUNNING)
    n.add_transition(transition)

    change = next(n.transition(RUNNING))
    assert isinstance(change, _StateChange)
    assert change.transition == transition
    assert change.src == "/0"


def test_orphaned_illegal_transition():
    n: _FSM = _FSM("/", "0")
    e = next(n.transition(RUNNING))
    assert e.transition == _Transition(_FSM_UNKNOWN, RUNNING)
    assert e.node == n


# def test_handled_illegal_transition():
#     tgf = """Ensemble /0
#     Realization /0/0
#     Step /0/0/0
#     Job /0/0/0/0
#     """
#     ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)

#     # UNKNOWN -> SUCCESS is handled by the step
#     gen = ens.dispatch(
#         CloudEvent(
#             {
#                 "type": ids.EVTYPE_FM_JOB_SUCCESS,
#                 "source": "/0/0/0/0",
#             }
#         )
#     )

#     illegal_transition = next(gen)

#     assert isinstance(illegal_transition, IllegalTransition)
#     assert illegal_transition.node == ens.children[0].children[0].children[0]
#     assert illegal_transition.transition.comparable_to(_STEP_PENDING, _STEP_SUCCESS)

#     change = next(gen)
#     assert isinstance(change, _StateChange)
#     assert change.node == ens.children[0].children[0].children[0]
#     assert change.transition.from_state == _JOB_START
#     assert change.transition.to_state == _JOB_FINISHED

#     # will cause a step to transition illegally
#     # TODO: maybe just handle this in the step

#     illegal_transition = next(gen)
#     assert isinstance(illegal_transition, IllegalTransition)
#     assert illegal_transition.node == ens.children[0].children[0]
#     assert illegal_transition.transition.from_state == _STEP_UNKNOWN
#     assert illegal_transition.transition.to_state == _STEP_SUCCESS

#     with pytest.raises(StopIteration):
#         next(gen)


def test_unhandled_illegal_transition():
    tgf = """Ensemble /0
    Realization /0/0
    Step /0/0/0
    Job /0/0/0/0
    """
    ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)

    # move job to RUNNING
    deque(
        ens.dispatch(
            CloudEvent(
                {
                    "type": ids.EVTYPE_FM_JOB_START,
                    "source": "/0/0/0/0",
                }
            )
        ),
        maxlen=0,
    )
    deque(
        ens.dispatch(
            CloudEvent(
                {
                    "type": ids.EVTYPE_FM_JOB_SUCCESS,
                    "source": "/0/0/0/0",
                }
            )
        ),
        maxlen=0,
    )

    # SUCCESS -> FAILURE is not handled by anyone
    gen = ens.dispatch(
        CloudEvent(
            {
                "type": ids.EVTYPE_FM_JOB_FAILURE,
                "source": "/0/0/0/0",
            }
        )
    )

    illegal_transition = next(gen)
    assert isinstance(illegal_transition, IllegalTransition)
    assert illegal_transition.node == ens.children[0].children[0].children[0]
    assert illegal_transition.transition.comparable_to(_JOB_FINISHED, _JOB_FAILURE)

    with pytest.raises(IllegalTransition):
        next(gen)


# def test_cascading_changes():
#     # test that completion of job, completes step, …, completes ens
#     tgf = """Ensemble /0
#     Realization /0/0
#     Step /0/0/0
#     Job /0/0/0/0
#     """
#     ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)
#     all_changes = list(
#         ens.dispatch(
#             CloudEvent(
#                 {
#                     "source": "/0/0/0/0",
#                     "type": ids.EVTYPE_FM_JOB_START,
#                 }
#             )
#         )
#     )
#     all_changes += list(
#         ens.dispatch(
#             CloudEvent(
#                 {
#                     "source": "/0/0/0/0",
#                     "type": ids.EVTYPE_FM_JOB_SUCCESS,
#                 }
#             )
#         )
#     )
#     partial = json.dumps(
#         _StateChange.changeset_to_partial(all_changes),
#         sort_keys=True,
#         indent=4,
#         cls=_Encoder,
#     )

#     assert (
#         partial
#         == f"""{{
#     "reals": {{
#         "0": {{
#             "status": "{_REALIZATION_FINISHED.name}",
#             "steps": {{
#                 "0": {{
#                     "jobs": {{
#                         "0": {{
#                             "status": "{_JOB_FINISHED.name}"
#                         }}
#                     }},
#                     "status": "{_STEP_SUCCESS.name}"
#                 }}
#             }}
#         }}
#     }},
#     "status": "{_ENSEMBLE_STOPPED.name}"
# }}"""
#     )
#     assert partial == json.dumps(
#         ens.snapshot_dict(),
#         sort_keys=True,
#         indent=4,
#         cls=_Encoder,
#     )


@pytest.mark.parametrize(
    "a,b,equal",
    [
        (_State("foo"), _State("bar"), False),
        (_State("foo"), _State("foo"), True),
        (_State("bar"), _State("foo"), False),
        (_State("foo", {}), _State("foo", {}), True),
        (_State("bar", {}), _State("foo", {}), False),
        (_State("foo", {"a": 0}), _State("foo", {"a": 0}), True),
        (_State("foo", {}), _State("foo", {"a": 0}), False),
        (_State("foo", {"a": 0}), _State("foo", {}), False),
    ],
)
def test_state_equality(a, b, equal):
    assert (a == b) == equal


def test_equal_states_yield_no_change():
    tgf = """Ensemble /0
    Realization /0/0
    Step /0/0/0
    """
    ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)
    gen = ens.dispatch(
        CloudEvent({"source": "/0", "type": ids.EVTYPE_ENSEMBLE_STARTED})
    )
    ens_running = next(gen)

    assert ens_running.node == ens
    assert ens_running.transition.from_state == _ENSEMBLE_UNKNOWN
    assert ens_running.transition.to_state == _ENSEMBLE_STARTED

    with pytest.raises(StopIteration):
        next(gen)

    for change in ens.dispatch(
        CloudEvent({"source": "/0/0/0", "type": ids.EVTYPE_FM_STEP_RUNNING})
    ):
        assert (
            change.transition.from_state != change.transition.to_state
        ), "transitioned to same state"


def test_equal_states_but_with_data_changes_yields_change():
    tgf = """Ensemble /0
    """
    ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)
    next(
        ens.dispatch(CloudEvent({"source": "/0", "type": ids.EVTYPE_ENSEMBLE_STARTED}))
    )

    gen = ens.dispatch(
        CloudEvent(
            {"source": "/0", "type": ids.EVTYPE_ENSEMBLE_STARTED}, {"foo": "bar"}
        )
    )
    ens_running = next(gen)

    assert ens_running.node == ens
    assert ens_running.transition.from_state == _ENSEMBLE_STARTED
    assert ens_running.transition.to_state == _ENSEMBLE_STARTED.with_data(
        {"foo": "bar"}
    )

    with pytest.raises(StopIteration):
        next(gen)


def test_timeout_step():
    tgf = """Ensemble /0
    Realization /0/0
    Step /0/0/0
    Job /0/0/0/0
    Job /0/0/0/1
    """
    ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)

    # timeout doesn't mutate state currently, which is a bit weird maybe
    gen = ens.dispatch(
        CloudEvent({"source": "/0/0/0", "type": ids.EVTYPE_FM_STEP_TIMEOUT})
    )

    for _ in range(2):
        job = next(gen)
        assert isinstance(job, _StateChange)
        assert job.transition.comparable_to(_JOB_START, _JOB_FAILURE)

    with pytest.raises(StopIteration):
        next(gen)


"""test/implement list
    - never transition away from realization/ens/step failure automatically
    - implement all_step_finished in Realization, etc
    - enough stuff such that PartialSnapshot can become immutable (migrate to pydantic)
    - Snapshot should become immutable (migrate to pydantic)
    - ens.snapshot+partial should be immutable
"""


def test_ensemble_metadata():
    ens = EnsembleFSM("/0")
    ens.metadata["iter"] = 1

    snapshot = ens.snapshot_dict()

    assert snapshot["metadata"]["iter"] == 1


def test_job_start_time():
    tgf = """Ensemble /0
    Realization /0/0
    Step /0/0/0
    Job /0/0/0/0
    """
    ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)
    ts = datetime.datetime.now()
    list(
        ens.dispatch(
            CloudEvent(
                {
                    "type": ids.EVTYPE_FM_JOB_START,
                    "source": "/0/0/0/0",
                    "time": ts.isoformat(),
                }
            )
        )
    )
    assert ens.children[0].children[0].children[0].state.data[ids.START_TIME] == ts


def test_job_end_time():
    tgf = """Ensemble /0
    Realization /0/0
    Step /0/0/0
    Job /0/0/0/0
    """
    ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)
    ts = datetime.datetime.now()
    list(
        ens.dispatch(
            CloudEvent(
                {
                    "type": ids.EVTYPE_FM_JOB_SUCCESS,
                    "source": "/0/0/0/0",
                    "time": ts.isoformat(),
                }
            )
        )
    )
    assert ens.children[0].children[0].children[0].state.data[ids.END_TIME] == ts


def test_step_start_time():
    tgf = """Ensemble /0
    Realization /0/0
    Step /0/0/0
    """
    ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)
    ts = datetime.datetime.now()
    list(
        ens.dispatch(
            CloudEvent(
                {
                    "type": ids.EVTYPE_FM_STEP_RUNNING,
                    "source": "/0/0/0",
                    "time": ts.isoformat(),
                }
            )
        )
    )
    assert ens.children[0].children[0].state.data[ids.START_TIME] == ts


def test_step_start_time():
    tgf = """Ensemble /0
    Realization /0/0
    Step /0/0/0
    """
    ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)
    ts = datetime.datetime.now()
    ens.children[0].children[0].state = _STEP_RUNNING
    next(
        ens.dispatch(
            CloudEvent(
                {
                    "type": ids.EVTYPE_FM_STEP_SUCCESS,
                    "source": "/0/0/0",
                    "time": ts.isoformat(),
                }
            )
        )
    )
    assert ens.children[0].children[0].state.data[ids.END_TIME] == ts


@pytest.mark.parametrize(
    "tgf",
    [
        (
            """Ensemble /0
    Realization /0/0
    """
        )
    ],
)
def test_tgf(tgf):
    ens = EnsembleFSM.from_ert_trivial_graph_format(tgf)
    assert isinstance(ens.children[0], RealizationFSM)
