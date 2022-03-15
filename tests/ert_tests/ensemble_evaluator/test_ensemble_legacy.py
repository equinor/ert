from unittest.mock import patch

import pytest

import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.status.entity import state


@pytest.mark.timeout(60)
def test_run_legacy_ensemble(tmpdir, make_ensemble_builder):
    num_reals = 2
    custom_port_range = range(1024, 65535)
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range, custom_host="127.0.0.1"
        )
        evaluator = EnsembleEvaluator(ensemble, config, 0, ee_id="1")
        with evaluator.run() as monitor:
            for e in monitor.track():
                if e["type"] in (
                    identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                    identifiers.EVTYPE_EE_SNAPSHOT,
                ) and e.data.get(identifiers.STATUS) in [
                    state.ENSEMBLE_STATE_FAILED,
                    state.ENSEMBLE_STATE_STOPPED,
                ]:
                    monitor.signal_done()
        assert evaluator._ensemble.get_status() == state.ENSEMBLE_STATE_STOPPED
        assert evaluator._ensemble.get_successful_realizations() == num_reals


@pytest.mark.timeout(60)
def test_run_and_cancel_legacy_ensemble(tmpdir, make_ensemble_builder):
    num_reals = 10
    custom_port_range = range(1024, 65535)
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2, job_sleep=5).build()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range, custom_host="127.0.0.1"
        )

        evaluator = EnsembleEvaluator(ensemble, config, 0, ee_id="1")

        with evaluator.run() as mon:
            cancel = True
            for _ in mon.track():
                if cancel:
                    mon.signal_cancel()
                    cancel = False

        assert evaluator._ensemble.get_status() == state.ENSEMBLE_STATE_CANCELLED


@pytest.mark.timeout(60)
def test_run_legacy_ensemble_exception(tmpdir, make_ensemble_builder):
    num_reals = 2
    custom_port_range = range(1024, 65535)
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range, custom_host="127.0.0.1"
        )
        evaluator = EnsembleEvaluator(ensemble, config, 0, ee_id="1")

        with patch.object(ensemble, "get_active_reals", side_effect=RuntimeError()):
            with evaluator.run() as monitor:
                for e in monitor.track():
                    if e.data is not None and e.data.get(identifiers.STATUS) in [
                        state.ENSEMBLE_STATE_FAILED,
                        state.ENSEMBLE_STATE_STOPPED,
                    ]:
                        monitor.signal_done()
            assert evaluator._ensemble.get_status() == state.ENSEMBLE_STATE_FAILED
