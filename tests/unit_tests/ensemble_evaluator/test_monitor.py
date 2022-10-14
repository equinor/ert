import pytest

from ert.ensemble_evaluator import Monitor, identifiers
from ert.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert.ensemble_evaluator.narratives import (
    monitor_failing_ensemble,
    monitor_failing_evaluation,
    monitor_successful_ensemble,
)
from ert.ensemble_evaluator.narratives.proxy import NarrativeProxy
from ert.ensemble_evaluator.state import ENSEMBLE_STATE_FAILED, ENSEMBLE_STATE_STOPPED

from .ensemble_evaluator_utils import TestEnsemble


@pytest.mark.consumer_driven_contract_test
def test_monitor_successful_ensemble(make_ee_config):
    ensemble = TestEnsemble(_iter=1, reals=2, steps=2, jobs=2, id_="0")
    ensemble.addFailJob(real=1, step=0, job=1)
    ee_config = make_ee_config(use_token=False, generate_cert=False)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
    )

    ee.run()
    with NarrativeProxy(monitor_successful_ensemble()).proxy(ee_config.url):
        with Monitor(ee_config.get_connection_info()) as monitor:
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
    ensemble = TestEnsemble(_iter=1, reals=1, steps=1, jobs=1, id_="0")
    ensemble.with_failure()
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
    )
    ee.run()
    with NarrativeProxy(
        monitor_failing_evaluation().on_uri(f"ws://localhost:{ee_config.port}")
    ).proxy(ee_config.url):
        with Monitor(ee_config.get_connection_info()) as monitor:
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
    ensemble = TestEnsemble(_iter=1, reals=2, steps=2, jobs=2, id_="0")
    ensemble.addFailJob(real=1, step=0, job=1)
    ee_config = make_ee_config(use_token=False, generate_cert=False)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
    )
    with ee.run():
        pass
    with NarrativeProxy(
        monitor_failing_ensemble().on_uri(f"ws://localhost:{ee_config.port}")
    ).proxy(ee_config.url):
        with Monitor(ee_config.get_connection_info()) as monitor:
            for event in monitor.track():
                if event["type"] == identifiers.EVTYPE_EE_SNAPSHOT:
                    ensemble.start()
                if (
                    event.data
                    and event.data.get(identifiers.STATUS) == ENSEMBLE_STATE_STOPPED
                ):
                    monitor.signal_done()

    ensemble.join()
