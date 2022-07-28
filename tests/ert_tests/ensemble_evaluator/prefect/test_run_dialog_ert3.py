from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ert3.evaluator._evaluator import ERT3RunModel
from ert.gui.simulation.run_dialog import RunDialog


def test_success_ert3(qtbot, poly_ensemble):
    ee_config = EvaluatorServerConfig(
        custom_port_range=range(1024, 65535), custom_host="127.0.0.1"
    )

    evaluator = EnsembleEvaluator(poly_ensemble, ee_config, 0, ee_id="1")
    run_model = ERT3RunModel()
    widget = RunDialog("poly_example", run_model)
    widget.show()
    qtbot.addWidget(widget)

    widget.kill_button.setHidden(True)
    future = widget.startSimulationErt3(evaluator)
    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.waitForWindowShown(widget)
        qtbot.waitUntil(run_model.isFinished, timeout=10000)
        qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100)
        qtbot.waitUntil(widget.done_button.isVisible, timeout=10000)

    results = future.result()
    assert 0 in results
    assert 1 in results
