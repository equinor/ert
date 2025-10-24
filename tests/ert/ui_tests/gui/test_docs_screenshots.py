import os.path
import shutil
from pathlib import Path
from textwrap import dedent

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QToolButton, QWidget
from skimage import io, transform
from skimage.metrics import structural_similarity as ssim

from ert.gui.ertwidgets import CopyableLabel
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.single_test_run_panel import SingleTestRunPanel
from ert.gui.simulation.view import RealizationWidget
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_ensemble_selection_widget import EnsembleSelectionWidget
from ert.run_models import EnsembleExperiment, EnsembleSmoother
from ert.services import StorageService
from ert.storage import open_storage
from tests.ert.ui_tests.gui.conftest import open_gui_with_config
from tests.ert.unit_tests.gui.simulation.test_run_path_dialog import (
    handle_run_path_dialog,
)

from .conftest import wait_for_child

# List of png files under docs that are either:
#  - not screenshots of the gui
# or:
#  - screenshots of the gui tied to a specific version of ert that will not change
# and are therefore not applicable for generation.
# Not currently used, but left here as a convenience for future work on these tests
# and in case we want to verify that all pngs under docs are tested for change unless
# they are listed as not applicable
PNGS_NOT_APPLICABLE_FOR_GENERATION = [
    "docs/ert/theory/images/posterior_path.png",
    "docs/ert/about/v9_auto_scale.png",
    "docs/ert/about/v10_manage_experiments.png",
    "docs/ert/about/log_scale_button.png",
    "docs/ert/about/license-retry.png",
    "docs/ert/about/click-on-stderr.png",
    "docs/ert/about/click-show-details.png",
    "docs/ert/about/v9_update_param.png",
    "docs/ert/img/logo.png",
    "docs/ert/reference/configuration/fig/errf_symmetric_uniform.png",
    "docs/ert/reference/configuration/fig/truncated_ok.png",
    "docs/ert/reference/configuration/fig/const.png",
    "docs/ert/reference/configuration/fig/lognormal.png",
    "docs/ert/reference/configuration/fig/derrf_symmetric_uniform.png",
    "docs/ert/reference/configuration/fig/dunif.png",
    "docs/ert/reference/configuration/fig/loguniform.png",
    "docs/ert/reference/configuration/fig/triangular.png",
    "docs/ert/reference/configuration/fig/derrf_right_skewed.png",
    "docs/ert/reference/configuration/fig/normal.png",
    "docs/ert/reference/configuration/fig/uniform.png",
    "docs/ert/reference/configuration/fig/errf_right_skewed_unimodal.png",
    "docs/ert/getting_started/configuration/poly_new/minimal/warning.png",
    "docs/ert/getting_started/configuration/poly_new/minimal/startdialog.png",
    "docs/ert/getting_started/updating_parameters/fig/prior_response.png",
    "docs/ert/getting_started/updating_parameters/fig/prior_params.png",
    "docs/everest/images/deter_vs_robust.png",
    "docs/everest/images/architecture_design.png",
    "docs/everest/images/everest_wf.png",
    "docs/everest/images/enopt_objfunc.png",
    "docs/everest/images/Everest_vs_Ert_03.png",
    "docs/everest/images/Everest_vs_Ert_02.png",
    "docs/everest/images/Everest_vs_Ert_01.png",
]


# List of png files under docs that could be tested for change
# and generated, but that has not yet been added to a test.
TODOS = [
    "docs/ert/getting_started/updating_parameters/fig/update_report.png",
    "docs/ert/getting_started/howto/restart-es-mda.png",
    "docs/ert/getting_started/howto/ert_screenshot_adaptive_loc.png",
]

SKIP_MESSAGE = """Skipping this test because no fonts could be found by qt.

This is a problem on the github runners using ubuntu-latest.
It has been verified that fonts are present in the system by calling fc-list,
but qt are not able to detect these fonts for some reason.
"""

FIXED_RANDOM_SEED = 11223344

COEFF_A_PNG_THRESHOLD = 0.9
COEFF_B_PNG_THRESHOLD = 0.9
COEFF_C_PNG_THRESHOLD = 0.9
ERT_PNG_THRESHOLD = 0.9
PLOT_OBS_PNG_THRESHOLD = 0.9
PLOTS_PNG_THRESHOLD = 0.9
POLY_PLOT_PNG_THRESHOLD = 0.9
SIMULATIONS_PNG_THRESHOLD = 0.9


def run_experiment(qtbot, experiment_mode, gui, click_done=True):
    # Select correct experiment in the simulation panel
    experiment_panel = gui.findChild(ExperimentPanel)
    assert isinstance(experiment_panel, ExperimentPanel)

    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)

    simulation_mode_combo.setCurrentText(experiment_mode.display_name())

    # Click start simulation and agree to the message
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")

    def handle_dialog():
        QTimer.singleShot(
            500,
            lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=False),
        )

    if experiment_mode.name() not in {"Ensemble experiment", "Evaluate ensemble"}:
        QTimer.singleShot(500, handle_dialog)

    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

    if click_done:
        # The Run dialog opens, click show details and wait until done appears
        # then click it
        run_dialog = wait_for_child(gui, qtbot, RunDialog, timeout=10000)
        qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=600000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

        # Assert that the number of boxes in the detailed view is
        # equal to the number of realizations
        realization_widget = run_dialog._tab_widget.currentWidget()
        assert isinstance(realization_widget, RealizationWidget)

        list_model = realization_widget._real_view.model()
        assert (
            list_model.rowCount()
            == experiment_panel.config.runpath_config.num_realizations
        )


# Checks if the current Python script is running within a GitHub Actions environment.
IS_RUNNING_IN_GITHUB_ACTIONS = (
    os.getenv("CI") == "true" and os.getenv("GITHUB_ACTIONS") == "true"
)


class GuiEvaluator:
    def __init__(self, source_root, example_folder, gui, qtbot) -> None:
        self.gui_changed: list[str] = []
        self.source_root = source_root
        self.example_folder = example_folder
        self.gui = gui
        self.qtbot = qtbot

    def compare_img_with_gui(self, img_name, threshold=0.99):
        temp_image_path = self.qtbot.screenshot(self.gui)
        new_img = io.imread(temp_image_path, as_gray=True)

        CI_CMP_PREFIX = "ci_cmp_"

        name = (
            f"{CI_CMP_PREFIX}{img_name}" if IS_RUNNING_IN_GITHUB_ACTIONS else img_name
        )

        image_path = os.path.join(self.example_folder, name)
        full_image_path = os.path.join(self.source_root, image_path)

        # The ssim_score is a decimal value between -1 and 1, where:
        #   1 indicates perfect similarity,
        #   0 indicates no similarity,
        #   and -1 indicates perfect anti-correlation.
        ssim_score = (
            self._get_ssim_score(new_img, io.imread(full_image_path, as_gray=True))
            if os.path.isfile(full_image_path)
            else 0
        )

        if ssim_score < threshold:
            if IS_RUNNING_IN_GITHUB_ACTIONS:
                # Copy the new image in temp storage for artifact upload
                tmp_img_storage = os.path.join(
                    "/tmp/test_docs_screenshots", self.example_folder
                )
                os.makedirs(tmp_img_storage, exist_ok=True)
                full_image_path = os.path.join(tmp_img_storage, f"{name}")

            shutil.copy(temp_image_path, full_image_path)
            self.gui_changed.append(
                f"{image_path} SSIM:{ssim_score} < Threshold:{threshold}"
            )

        os.remove(temp_image_path)

    def gui_change_detected(self):
        return len(self.gui_changed) > 0

    @staticmethod
    def _get_ssim_score(img1, img2):
        # Images of different shape cannot be compared with ssim
        if img1.shape != img2.shape:
            img2 = transform.resize(img2, img1.shape, anti_aliasing=True)
        return ssim(img1, img2, data_range=img1.max() - img1.min())

    def change_report(self):
        if not self.gui_change_detected():
            return "No gui changes detected"

        newline = "\n            - "
        return dedent(
            f"""
            One or more auto generated images differed from the image used in the docs

            The image(s):
            - {newline.join(self.gui_changed)}
            {
                "has been added to the test-images artifact."
                if IS_RUNNING_IN_GITHUB_ACTIONS
                else "has been overwritten with the new version."
            }
            If the new and old image looks the same and both are correct, the test might
            simply need to lower the similarity threshold.

            If the updated image reflects an intended gui change it should be kept.
            If the updated image seems incorrect the underlying code generating
            the gui might need corrections or this test needs to be fixed.
            """
        )


def set_data_type_selection_index(data_type_widget, index):
    data_type_widget.data_type_keys_widget.setCurrentIndex(
        data_type_widget.filter_model.index(index, 0)
    )


def clean_up_diplayed_runpath(gui: QWidget):
    experiment_panel = gui.findChild(ExperimentPanel)
    assert isinstance(experiment_panel, ExperimentPanel)

    single_test_run_panel = experiment_panel.findChild(SingleTestRunPanel)
    assert isinstance(single_test_run_panel, SingleTestRunPanel)

    runpath_label = single_test_run_panel.findChild(CopyableLabel)
    current_directory = os.getcwd()
    label_text = runpath_label.label.text()
    runpath_label.label.setText(label_text.replace(current_directory, "&lt;cwd&gt;"))


def open_gui_with_docs_example(
    tmp_path, source_root, example_folder, config_file, random_seed=None
):
    def ignore_pngs(src, files):
        return [f for f in files if f.endswith(".png")]

    shutil.copytree(
        os.path.join(source_root, example_folder),
        tmp_path,
        ignore=ignore_pngs,
        dirs_exist_ok=True,
    )

    if random_seed is not None:
        original_content = Path(config_file).read_text(encoding="utf-8")

        combined_content = (
            f"RANDOM_SEED {random_seed}\n" + original_content
        )  # Add fixed random seed to make the plots reproducible

        Path(config_file).write_text(combined_content, encoding="utf-8")

    gui_generator = open_gui_with_config(tmp_path / config_file)
    return next(gui_generator)


@pytest.mark.skip_mac_ci
def test_that_poly_new_minimal_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)

    example_folder = "docs/ert/getting_started/configuration/poly_new/minimal"
    gui = open_gui_with_docs_example(tmp_path, source_root, example_folder, "poly.ert")

    if not gui.available_fonts:
        pytest.skip(SKIP_MESSAGE)

    gui_evaluator = GuiEvaluator(source_root, example_folder, gui, qtbot)
    gui_evaluator.compare_img_with_gui("ert.png", ERT_PNG_THRESHOLD)

    run_experiment(qtbot, EnsembleExperiment, gui)

    gui_evaluator.compare_img_with_gui("simulations.png", SIMULATIONS_PNG_THRESHOLD)

    assert not gui_evaluator.gui_change_detected(), gui_evaluator.change_report()


@pytest.mark.skip_mac_ci
def test_that_poly_new_with_simple_script_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)

    example_folder = (
        "docs/ert/getting_started/configuration/poly_new/with_simple_script"
    )

    gui = open_gui_with_docs_example(tmp_path, source_root, example_folder, "poly.ert")

    if not gui.available_fonts:
        pytest.skip(SKIP_MESSAGE)

    clean_up_diplayed_runpath(gui)

    gui_evaluator = GuiEvaluator(source_root, example_folder, gui, qtbot)
    gui_evaluator.compare_img_with_gui("ert.png", ERT_PNG_THRESHOLD)

    assert not gui_evaluator.gui_change_detected(), gui_evaluator.change_report()


@pytest.mark.skip_mac_ci
def test_that_poly_new_with_results_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)

    example_folder = "docs/ert/getting_started/configuration/poly_new/with_results"

    gui = open_gui_with_docs_example(
        tmp_path, source_root, example_folder, "poly.ert", FIXED_RANDOM_SEED
    )

    if not gui.available_fonts:
        pytest.skip(SKIP_MESSAGE)

    run_experiment(qtbot, EnsembleExperiment, gui)
    open_storage(gui.ert_config.ens_path, mode="w")
    gui_evaluator = GuiEvaluator(source_root, example_folder, gui, qtbot)

    with StorageService.init_service(
        project=os.path.abspath(gui.ert_config.ens_path),
    ):
        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)

        gui_evaluator.compare_img_with_gui("poly_plot.png", POLY_PLOT_PNG_THRESHOLD)

        data_type_widget = wait_for_child(gui, qtbot, DataTypeKeysWidget, "Data types")
        set_data_type_selection_index(data_type_widget, 1)

        gui_evaluator.compare_img_with_gui("plots.png", PLOTS_PNG_THRESHOLD)

    assert not gui_evaluator.gui_change_detected(), gui_evaluator.change_report()


@pytest.mark.skip_mac_ci
def test_that_poly_new_with_observations_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)

    example_folder = "docs/ert/getting_started/configuration/poly_new/with_observations"

    gui = open_gui_with_docs_example(
        tmp_path, source_root, example_folder, "poly_final.ert", FIXED_RANDOM_SEED
    )

    if not gui.available_fonts:
        pytest.skip(SKIP_MESSAGE)

    run_experiment(qtbot, EnsembleSmoother, gui)
    open_storage(gui.ert_config.ens_path, mode="w")
    gui_evaluator = GuiEvaluator(source_root, example_folder, gui, qtbot)

    with StorageService.init_service(
        project=os.path.abspath(gui.ert_config.ens_path),
    ):
        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)

        ensemble_selector_widget = wait_for_child(
            gui, qtbot, EnsembleSelectionWidget, "Plot ensemble"
        )

        dndlist = ensemble_selector_widget._EnsembleSelectionWidget__dndlist

        for index in range(dndlist.count()):
            item = dndlist.item(index)
            if not item.data(Qt.ItemDataRole.CheckStateRole):
                dndlist.slot_toggle_plot(item)
        dndlist.ensembleSelectionListChanged.emit()

        gui_evaluator.compare_img_with_gui("plot_obs.png", PLOT_OBS_PNG_THRESHOLD)

        data_type_widget = wait_for_child(gui, qtbot, DataTypeKeysWidget, "Data types")

        for index, img, threshold in [
            (1, "coeff_a.png", COEFF_A_PNG_THRESHOLD),
            (2, "coeff_b.png", COEFF_B_PNG_THRESHOLD),
            (3, "coeff_c.png", COEFF_C_PNG_THRESHOLD),
        ]:
            set_data_type_selection_index(data_type_widget, index)
            gui_evaluator.compare_img_with_gui(img, threshold)

    assert not gui_evaluator.gui_change_detected(), gui_evaluator.change_report()


@pytest.mark.skip_mac_ci
def test_that_poly_new_with_more_observations_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)

    example_folder = (
        "docs/ert/getting_started/configuration/poly_new/with_more_observations"
    )

    gui = open_gui_with_docs_example(
        tmp_path, source_root, example_folder, "poly_final.ert", FIXED_RANDOM_SEED
    )

    if not gui.available_fonts:
        pytest.skip(SKIP_MESSAGE)

    run_experiment(qtbot, EnsembleSmoother, gui)
    open_storage(gui.ert_config.ens_path, mode="w")
    gui_evaluator = GuiEvaluator(source_root, example_folder, gui, qtbot)

    with StorageService.init_service(
        project=os.path.abspath(gui.ert_config.ens_path),
    ):
        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)

        ensemble_selector_widget = wait_for_child(
            gui, qtbot, EnsembleSelectionWidget, "Plot ensemble"
        )

        dndlist = ensemble_selector_widget._EnsembleSelectionWidget__dndlist
        for index in range(dndlist.count()):
            item = dndlist.item(index)
            if not item.data(Qt.ItemDataRole.CheckStateRole):
                dndlist.slot_toggle_plot(item)
        dndlist.ensembleSelectionListChanged.emit()

        data_type_widget = wait_for_child(gui, qtbot, DataTypeKeysWidget, "Data types")
        set_data_type_selection_index(data_type_widget, 2)

        gui_evaluator.compare_img_with_gui("coeff_b.png", COEFF_B_PNG_THRESHOLD)

    assert not gui_evaluator.gui_change_detected(), gui_evaluator.change_report()
