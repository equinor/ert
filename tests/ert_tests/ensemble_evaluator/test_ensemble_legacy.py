import os
from unittest.mock import patch

import pytest
from typing import Tuple

import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.evaluator import (
    EnsembleEvaluator,
    EnsembleEvaluatorSession,
)
from ert_shared.status.entity import state


@pytest.mark.timeout(60)
def test_run_legacy_ensemble(
    tmpdir,
    make_ensemble_builder,
    evaluator_experiment_session: Tuple[EnsembleEvaluatorSession, str],
):
    evaluator, experiment_id = evaluator_experiment_session
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        evaluation_id = evaluator.submit_ensemble(
            ensemble=ensemble, iter_=0, experiment_id=experiment_id
        )
        with evaluator.run(evaluation_id) as monitor:
            for e in monitor.track():
                if e["type"] in (
                    identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                    identifiers.EVTYPE_EE_SNAPSHOT,
                ) and e.data.get(identifiers.STATUS) in [
                    state.ENSEMBLE_STATE_FAILED,
                    state.ENSEMBLE_STATE_STOPPED,
                ]:
                    monitor.signal_done()
        assert (
            evaluator._evaluator.ensemble_evaluations[
                evaluation_id
            ]._ensemble.get_status()
            == state.ENSEMBLE_STATE_STOPPED
        )
        assert (
            evaluator._evaluator.ensemble_evaluations[
                evaluation_id
            ]._ensemble.get_successful_realizations()
            == num_reals
        )

        # realisations should finish, each creating a status-file
        for i in range(num_reals):
            assert os.path.isfile(f"real_{i}/status.txt")


@pytest.mark.timeout(60)
def test_run_and_cancel_legacy_ensemble(
    tmpdir,
    make_ensemble_builder,
    evaluator_experiment_session: Tuple[EnsembleEvaluatorSession, str],
):
    evaluator, experiment_id = evaluator_experiment_session
    num_reals = 10
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2, job_sleep=5).build()

        evaluation_id = evaluator.submit_ensemble(
            ensemble=ensemble, iter_=0, experiment_id=experiment_id
        )
        with evaluator.run(evaluation_id) as mon:
            cancel = True
            for _ in mon.track():
                if cancel:
                    mon.signal_cancel()
                    cancel = False

        assert (
            evaluator._evaluator.ensemble_evaluations[
                evaluation_id
            ]._ensemble.get_status()
            == state.ENSEMBLE_STATE_CANCELLED
        )

        # realisations should not finish, thus not creating a status-file
        for i in range(num_reals):
            assert not os.path.isfile(f"real_{i}/status.txt")


@pytest.mark.timeout(60)
def test_run_legacy_ensemble_exception(
    tmpdir,
    make_ensemble_builder,
    evaluator_experiment_session: Tuple[EnsembleEvaluatorSession, str],
):
    evaluator, experiment_id = evaluator_experiment_session
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        evaluation_id = evaluator.submit_ensemble(
            ensemble=ensemble, iter_=0, experiment_id=experiment_id
        )

        with patch.object(ensemble, "get_active_reals", side_effect=RuntimeError()):
            with evaluator.run(evaluation_id) as monitor:
                for e in monitor.track():
                    if e.data is not None and e.data.get(identifiers.STATUS) in [
                        state.ENSEMBLE_STATE_FAILED,
                        state.ENSEMBLE_STATE_STOPPED,
                    ]:
                        monitor.signal_done()
            assert (
                evaluator._evaluator.ensemble_evaluations[
                    evaluation_id
                ]._ensemble.get_status()
                == state.ENSEMBLE_STATE_FAILED
            )
