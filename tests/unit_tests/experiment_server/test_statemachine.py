import pytest

import _ert_com_protocol

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_update_job():
    state_machine = _ert_com_protocol.ExperimentStateMachine()
    id_job = _ert_com_protocol.JobId(
        index=0,
        step=_ert_com_protocol.StepId(
            step=0,
            realization=_ert_com_protocol.RealizationId(
                realization=0,
                ensemble=_ert_com_protocol.EnsembleId(
                    id="ee_id",
                    experiment=_ert_com_protocol.ExperimentId(id="experiment_id"),
                ),
            ),
        ),
    )

    pjob = _ert_com_protocol.Job(id=id_job)
    pjob.status = _ert_com_protocol.JOB_START
    pjob.name = "job1"
    await state_machine.update(pjob)
    assert "ee_id" in state_machine.state.ensembles
    assert (
        state_machine.state.ensembles["ee_id"].realizations[0].steps[0].jobs[0].status
        == _ert_com_protocol.JOB_START
    )
    assert (
        state_machine.state.ensembles["ee_id"].realizations[0].steps[0].jobs[0].name
        == "job1"
    )
    pjob.status = _ert_com_protocol.JOB_FAILURE
    pjob.error = "Failed dramatically"
    await state_machine.update(pjob)
    assert (
        state_machine.state.ensembles["ee_id"].realizations[0].steps[0].jobs[0].status
        == _ert_com_protocol.JOB_FAILURE
    )
    assert (
        state_machine.state.ensembles["ee_id"].realizations[0].steps[0].jobs[0].error
        == "Failed dramatically"
    )


async def test_update_job_different_experiments():
    state_machine = _ert_com_protocol.ExperimentStateMachine()
    id_job = _ert_com_protocol.JobId(
        index=0,
        step=_ert_com_protocol.StepId(
            step=0,
            realization=_ert_com_protocol.RealizationId(
                realization=0,
                ensemble=_ert_com_protocol.EnsembleId(
                    id="ee_id",
                    experiment=_ert_com_protocol.ExperimentId(id="experiment_id"),
                ),
            ),
        ),
    )

    pjob = _ert_com_protocol.Job(id=id_job)
    pjob.status = _ert_com_protocol.JOB_START
    pjob.name = "job1"
    await state_machine.update(pjob)
    assert "ee_id" in state_machine.state.ensembles
    assert (
        state_machine.state.ensembles["ee_id"].realizations[0].steps[0].jobs[0].status
        == _ert_com_protocol.JOB_START
    )

    id_job_2 = _ert_com_protocol.JobId(
        index=0,
        step=_ert_com_protocol.StepId(
            step=0,
            realization=_ert_com_protocol.RealizationId(
                realization=0,
                ensemble=_ert_com_protocol.EnsembleId(
                    id="ee_id",
                    experiment=_ert_com_protocol.ExperimentId(id="another_id"),
                ),
            ),
        ),
    )
    pjob = _ert_com_protocol.Job(id=id_job_2)
    pjob.status = _ert_com_protocol.JOB_START
    with pytest.raises(_ert_com_protocol.ExperimentStateMachine.IllegalStateUpdate):
        await state_machine.update(pjob)


@pytest.mark.parametrize(
    "id_args, expected_type",
    [
        (["EXPERIMENT_STARTED", "exid", None, None, None, None], "experiment"),
        (["ENSEMBLE_STARTED", "exid", "ensid", None, None, None], "ensemble"),
        (["STEP_WAITING", "exid", "ensid", 0, None, None], "realization"),
        (["STEP_WAITING", "exid", "ensid", 0, 0, None], "step"),
        (["JOB_START", "exid", "ensid", 0, 0, 0], "job"),
    ],
)
async def test_node_builder(id_args, expected_type):
    msg: _ert_com_protocol.DispatcherMessage = _ert_com_protocol.node_status_builder(
        status=id_args[0],
        experiment_id=id_args[1],
        ensemble_id=id_args[2],
        realization_id=id_args[3],
        step_id=id_args[4],
        job_id=id_args[5],
    )
    assert msg.WhichOneof("object") == expected_type


async def test_node_builder_wrong_state():
    with pytest.raises(ValueError):
        _ert_com_protocol.node_status_builder(
            status="SOME_WEIRD_STATUS", experiment_id="exp_id"
        )


async def test_get_successful_realizations():
    state_machine = _ert_com_protocol.ExperimentStateMachine()
    for real_id in range(5):
        job_node: _ert_com_protocol.Job = _ert_com_protocol.node_status_builder(
            status="JOB_START",
            experiment_id="exp_id",
            ensemble_id="ens_id",
            realization_id=real_id,
            step_id=0,
            job_id=0,
        ).job
        await state_machine.update(job_node)
    assert state_machine.successful_realizations("ens_id") == 0

    step_node: _ert_com_protocol.Step = _ert_com_protocol.node_status_builder(
        status="STEP_SUCCESS",
        experiment_id="exp_id",
        ensemble_id="ens_id",
        realization_id=3,
        step_id=0,
    ).step
    await state_machine.update(step_node)
    assert state_machine.successful_realizations("ens_id") == 1
