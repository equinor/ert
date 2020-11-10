import pytest
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from pathlib import Path
from ert_shared.ensemble_evaluator.config import CONFIG_FILE, load_config
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator


@pytest.mark.timeout(60)
def test_run_legacy_ensemble(tmpdir, unused_tcp_port, make_ensemble_builder):
    num_reals = 2
    conf_file = Path(tmpdir / CONFIG_FILE)
    with tmpdir.as_cwd():
        with open(conf_file, "w") as f:
            f.write(f'port: "{unused_tcp_port}"\n')

        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = load_config(conf_file)
        evaluator = EnsembleEvaluator(ensemble, config, ee_id="1")
        monitor = evaluator.run()
        for e in monitor.track():
            if (
                e["type"]
                in (
                    identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                    identifiers.EVTYPE_EE_SNAPSHOT,
                )
                and e.data.get("status") == "Stopped"
            ):
                monitor.signal_done()
        assert evaluator.get_successful_realizations() == num_reals


@pytest.mark.timeout(60)
def test_run_and_cancel_legacy_ensemble(tmpdir, unused_tcp_port, make_ensemble_builder):
    num_reals = 10
    conf_file = Path(tmpdir / CONFIG_FILE)

    with tmpdir.as_cwd():
        with open(conf_file, "w") as f:
            f.write(f'port: "{unused_tcp_port}"\n')

        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = load_config(conf_file)

        evaluator = EnsembleEvaluator(ensemble, config, ee_id="1")

        mon = evaluator.run()
        cancel = True
        for _ in mon.track():
            if cancel:
                mon.signal_cancel()
                cancel = False

        assert evaluator._snapshot.get_status() == "Cancelled"
