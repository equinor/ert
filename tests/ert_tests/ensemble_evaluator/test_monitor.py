import pytest
from ensemble_evaluator_utils import TestEnsemble

from ert.ensemble_evaluator import identifiers
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator, ee_monitor
from ert_shared.ensemble_evaluator.narratives import (
    monitor_failing_ensemble,
    monitor_failing_evaluation,
    monitor_successful_ensemble,
)
from ert_shared.ensemble_evaluator.narratives.proxy import NarrativeProxy
from ert.ensemble_evaluator.state import ENSEMBLE_STATE_FAILED, ENSEMBLE_STATE_STOPPED


@pytest.mark.consumer_driven_contract_test
def test_monitor_successful_ensemble(make_ee_config):
    ensemble = TestEnsemble(iter=1, reals=2, steps=2, jobs=2)
    ensemble.addFailJob(real=1, step=0, job=1)
    ee_config = make_ee_config(use_token=False, generate_cert=False)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )

    ee.run()
    with NarrativeProxy(monitor_successful_ensemble()).proxy(ee_config.url):
        with ee_monitor.create(ee_config.get_connection_info()) as monitor:
            for event in monitor.track():
                if event["type"] == identifiers.EVTYPE_EE_SNAPSHOT:
                    ensemble.start()
                if (
                    event.data
                    and event.data.get(identifiers.STATUS) == ENSEMBLE_STATE_STOPPED
                ):
                    monitor.signal_done()

    ensemble.join()


@pytest.mark.consumer_driven_contract_test
def test_monitor_failing_evaluation(make_ee_config):
    ee_config = make_ee_config(use_token=False, generate_cert=False)
    ensemble = TestEnsemble(iter=1, reals=1, steps=1, jobs=1)
    ensemble.with_failure()
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    ee.run()
    with NarrativeProxy(
        monitor_failing_evaluation().on_uri(f"ws://localhost:{ee_config.port}")
    ).proxy(ee_config.url):
        with ee_monitor.create(ee_config.get_connection_info()) as monitor:
            for event in monitor.track():
                if event["type"] == identifiers.EVTYPE_EE_SNAPSHOT:
                    ensemble.start()
                if (
                    event.data
                    and event.data.get(identifiers.STATUS) == ENSEMBLE_STATE_FAILED
                ):
                    monitor.signal_done()

    ensemble.join()


@pytest.mark.consumer_driven_contract_test
def test_monitor_failing_ensemble(make_ee_config):
    ensemble = TestEnsemble(iter=1, reals=2, steps=2, jobs=2)
    ensemble.addFailJob(real=1, step=0, job=1)
    ee_config = make_ee_config(use_token=False, generate_cert=False)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    with ee.run():
        pass
    with NarrativeProxy(
        monitor_failing_ensemble().on_uri(f"ws://localhost:{ee_config.port}")
    ).proxy(ee_config.url):
        with ee_monitor.create(ee_config.get_connection_info()) as monitor:
            for event in monitor.track():
                if event["type"] == identifiers.EVTYPE_EE_SNAPSHOT:
                    ensemble.start()
                if (
                    event.data
                    and event.data.get(identifiers.STATUS) == ENSEMBLE_STATE_STOPPED
                ):
                    monitor.signal_done()

    ensemble.join()
