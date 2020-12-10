from unittest.mock import patch
import pytest
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.status.entity import state
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator


@pytest.mark.timeout(60)
def test_run_legacy_ensemble(tmpdir, unused_tcp_port, make_ensemble_builder):
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(unused_tcp_port)
        evaluator = EnsembleEvaluator(ensemble, config, 0, ee_id="1")
        with evaluator.run() as monitor:
            for e in monitor.track():
                if (
                    e["type"]
                    in (
                        identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                        identifiers.EVTYPE_EE_SNAPSHOT,
                    )
                    and e.data.get(identifiers.STATUS)
                    in [state.ENSEMBLE_STATE_FAILED, state.ENSEMBLE_STATE_STOPPED]
                ):
                    monitor.signal_done()
        assert evaluator._snapshot.get_status() == state.ENSEMBLE_STATE_STOPPED
        assert evaluator.get_successful_realizations() == num_reals


@pytest.mark.timeout(60)
def test_run_and_cancel_legacy_ensemble(tmpdir, unused_tcp_port, make_ensemble_builder):
    num_reals = 10
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(unused_tcp_port)

        evaluator = EnsembleEvaluator(ensemble, config, 0, ee_id="1")

        with evaluator.run() as mon:
            cancel = True
            for _ in mon.track():
                if cancel:
                    mon.signal_cancel()
                    cancel = False

        assert evaluator._snapshot.get_status() == state.ENSEMBLE_STATE_CANCELLED


@pytest.mark.timeout(60)
def test_run_legacy_ensemble_exception(tmpdir, unused_tcp_port, make_ensemble_builder):
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(unused_tcp_port)
        evaluator = EnsembleEvaluator(ensemble, config, 0, ee_id="1")

        with patch.object(ensemble, "_run_path_list", side_effect=RuntimeError()):
            with evaluator.run() as monitor:
                for e in monitor.track():
                    if e.data is not None and e.data.get(identifiers.STATUS) in [
                        state.ENSEMBLE_STATE_FAILED,
                        state.ENSEMBLE_STATE_STOPPED,
                    ]:
                        monitor.signal_done()
            assert evaluator._snapshot.get_status() == state.ENSEMBLE_STATE_FAILED
