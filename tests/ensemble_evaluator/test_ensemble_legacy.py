from unittest.mock import patch
import pytest
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator


@pytest.mark.timeout(60)
def test_run_legacy_ensemble(tmpdir, unused_tcp_port, make_ensemble_builder):
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(unused_tcp_port)
        evaluator = EnsembleEvaluator(ensemble, config, ee_id="1")
        monitor = evaluator.run()
        for e in monitor.track():
            if (
                e["type"]
                in (
                    identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                    identifiers.EVTYPE_EE_SNAPSHOT,
                )
                and e.data.get("status") in ["Failed", "Stopped"]
            ):
                monitor.signal_done()
        assert evaluator._snapshot.get_status() == "Stopped"
        assert evaluator.get_successful_realizations() == num_reals


@pytest.mark.timeout(60)
def test_run_and_cancel_legacy_ensemble(tmpdir, unused_tcp_port, make_ensemble_builder):
    num_reals = 10
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(unused_tcp_port)

        evaluator = EnsembleEvaluator(ensemble, config, ee_id="1")

        mon = evaluator.run()
        cancel = True
        for _ in mon.track():
            if cancel:
                mon.signal_cancel()
                cancel = False

        assert evaluator._snapshot.get_status() == "Cancelled"


@pytest.mark.timeout(60)
def test_run_legacy_ensemble_exception(tmpdir, unused_tcp_port, make_ensemble_builder):
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(unused_tcp_port)
        evaluator = EnsembleEvaluator(ensemble, config, ee_id="1")

        with patch.object(ensemble, "_run_path_list", side_effect=RuntimeError()):
            monitor = evaluator.run()
            for e in monitor.track():
                if (
                    e["type"]
                    in (
                        identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                        identifiers.EVTYPE_EE_SNAPSHOT,
                    )
                    and e.data.get("status") in ["Failed", "Stopped"]
                ):
                    monitor.signal_done()
            assert evaluator._snapshot.get_status() == "Failed"
