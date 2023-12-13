import contextlib
import os
import shutil
from unittest.mock import patch

import pytest
from websockets.exceptions import ConnectionClosed

from ert.ensemble_evaluator import identifiers, state
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert.ensemble_evaluator.monitor import Monitor
from ert.shared.feature_toggling import FeatureToggling


@pytest.mark.timeout(60)
def test_run_legacy_ensemble(tmpdir, make_ensemble_builder, monkeypatch):
    _test_run_legacy_ensemble(tmpdir, make_ensemble_builder, monkeypatch)
    monkeypatch.setattr(FeatureToggling._conf["scheduler"], "is_enabled", True)
    shutil.rmtree(tmpdir)
    tmpdir.mkdir()
    _test_run_legacy_ensemble(tmpdir, make_ensemble_builder, monkeypatch)


def _test_run_legacy_ensemble(tmpdir, make_ensemble_builder, monkeypatch):
    num_reals = 2
    custom_port_range = range(1024, 65535)
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(monkeypatch, tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range,
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )
        evaluator = EnsembleEvaluator(ensemble, config, 0)
        evaluator._start_running()
        with Monitor(config) as monitor:
            for e in monitor.track():
                if e["type"] in (
                    identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                    identifiers.EVTYPE_EE_SNAPSHOT,
                ) and e.data.get(identifiers.STATUS) in [
                    state.ENSEMBLE_STATE_FAILED,
                    state.ENSEMBLE_STATE_STOPPED,
                ]:
                    monitor.signal_done()
        assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
        assert len(evaluator._ensemble.get_successful_realizations()) == num_reals

        # realisations should finish, each creating a status-file
        for i in range(num_reals):
            assert os.path.isfile(f"real_{i}/status.txt")


@pytest.mark.timeout(60)
def test_run_and_cancel_legacy_ensemble(tmpdir, make_ensemble_builder, monkeypatch):
    _test_run_and_cancel_legacy_ensemble(tmpdir, make_ensemble_builder, monkeypatch)
    monkeypatch.setattr(FeatureToggling._conf["scheduler"], "is_enabled", True)
    shutil.rmtree(tmpdir)
    tmpdir.mkdir()
    _test_run_and_cancel_legacy_ensemble(tmpdir, make_ensemble_builder, monkeypatch)


def _test_run_and_cancel_legacy_ensemble(tmpdir, make_ensemble_builder, monkeypatch):
    num_reals = 2
    custom_port_range = range(1024, 65535)
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(
            monkeypatch, tmpdir, num_reals, 2, job_sleep=40
        ).build()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range,
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )

        evaluator = EnsembleEvaluator(ensemble, config, 0)

        evaluator._start_running()
        with Monitor(config) as mon:
            cancel = True
            with contextlib.suppress(
                ConnectionClosed
            ):  # monitor throws some variant of CC if dispatcher dies
                for _ in mon.track():
                    # Cancel the ensemble upon the arrival of the first event
                    if cancel:
                        mon.signal_cancel()
                        cancel = False

        assert evaluator._ensemble.status == state.ENSEMBLE_STATE_CANCELLED

        # realisations should not finish, thus not creating a status-file
        for i in range(num_reals):
            assert not os.path.isfile(f"real_{i}/status.txt")


@pytest.mark.timeout(10)
def test_run_legacy_ensemble_exception(tmpdir, make_ensemble_builder, monkeypatch):
    num_reals = 2
    custom_port_range = range(1024, 65535)
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(monkeypatch, tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range,
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )
        evaluator = EnsembleEvaluator(ensemble, config, 0)

        with patch.object(ensemble._job_queue, "add_realization") as faulty_queue:
            faulty_queue.side_effect = RuntimeError()
            evaluator._start_running()
            with Monitor(config) as monitor:
                for e in monitor.track():
                    if e.data is not None and e.data.get(identifiers.STATUS) in [
                        state.ENSEMBLE_STATE_FAILED,
                        state.ENSEMBLE_STATE_STOPPED,
                    ]:
                        monitor.signal_done()
            assert evaluator._ensemble.status == state.ENSEMBLE_STATE_FAILED

        # realisations should not finish, thus not creating a status-file
        for i in range(num_reals):
            assert not os.path.isfile(f"real_{i}/status.txt")
