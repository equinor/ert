from pathlib import Path
from queue import SimpleQueue
from textwrap import dedent

import polars as pl
import pytest
from PyQt6.QtWidgets import QComboBox
from pytestqt.qtbot import QtBot

from ert.config import DataSource, ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.experiments import ExperimentPanel
from ert.run_models import (
    EnsembleExperiment,
    EnsembleInformationFilter,
    EnsembleSmoother,
    MultipleDataAssimilation,
    RunModel,
    SingleTestRun,
    create_model,
)
from tests.ert.conftest import _create_design_matrix

from .conftest import MockStorage


@pytest.mark.filterwarnings("ignore:Parameters \\['b'\\] are also defined")
@pytest.mark.parametrize(
    "experiment_type",
    [
        SingleTestRun,
        EnsembleExperiment,
        EnsembleSmoother,
        EnsembleInformationFilter,
        MultipleDataAssimilation,
    ],
)
def test_that_gui_experiments_with_sampled_priority_only_sample_gen_kw_parameters(
    qtbot: QtBot, copy_case, experiment_type: type[RunModel]
):
    copy_case("poly_example")
    _create_design_matrix(
        "poly_design.xlsx",
        pl.DataFrame({"REAL": [0, 1], "b": [10.0, 11.0], "c": [20.0, 21.0]}),
    )
    Path("coeff_priors").write_text("a UNIFORM 0 1\nb UNIFORM 0 2\n", encoding="utf-8")
    Path("poly.ert").write_text(
        dedent(
            """\
            NUM_REALIZATIONS 2
            OBS_CONFIG observations
            GEN_KW COEFFS coeff_priors
            GEN_DATA POLY_RES RESULT_FILE:poly.out
            DESIGN_MATRIX poly_design.xlsx PRIORITY:sampled
            """
        ),
        encoding="utf-8",
    )
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    experiment_panel = ExperimentPanel(config, notifier, "poly.ert")
    qtbot.addWidget(experiment_panel)
    experiment_type_combo = experiment_panel.findChild(QComboBox, "experiment_type")
    assert experiment_type_combo is not None
    experiment_type_combo.setCurrentText(experiment_type.display_name())

    model = create_model(
        config, experiment_panel.get_experiment_arguments(), SimpleQueue()
    )

    assert isinstance(model, experiment_type)
    assert {p.name: p.input_source for p in model.parameter_configuration} == {
        "a": DataSource.SAMPLED,
        "b": DataSource.SAMPLED,
        "c": DataSource.DESIGN_MATRIX,
    }
