import os.path
import shutil
from textwrap import dedent

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QToolButton, QWidget
from skimage import io
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

PNGS_NOT_APPLICABLE_FOR_GENERATION = [
    "docs/ert/theory/images/posterior_path.png",
    "docs/ert/about/v9_auto_scale.png",
    "docs/ert/about/v10_manage_experiments.png",
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

TODOS = [
    "docs/ert/getting_started/updating_parameters/fig/update_report.png",
    "docs/ert/getting_started/howto/restart-es-mda.png",
    "docs/ert/getting_started/howto/ert_screenshot_adaptive_loc.png",
]


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

    if experiment_mode.name() not in {
        "Ensemble experiment",
        "Evaluate ensemble",
    }:
        QTimer.singleShot(500, handle_dialog)
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

    if click_done:
        # The Run dialog opens, click show details and wait until done appears
        # then click it
        run_dialog = wait_for_child(gui, qtbot, RunDialog, timeout=10000)
        qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=200000)
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


def compare_img_with_gui(gui_changed, example_folder, image_name, gui, qtbot):
    current_image_path = os.path.join(example_folder, image_name)
    if gui_has_significant_change(gui, qtbot, current_image_path):
        gui_changed.append(current_image_path)


def set_data_type_selection_index(data_type_widget, index):
    data_type_widget.data_type_keys_widget.setCurrentIndex(
        data_type_widget.filter_model.index(index, 0)
    )


def gui_has_significant_change(gui, qtbot, current_image_path, threshold=0.99):
    temp_image_path = qtbot.screenshot(gui)
    new_image = io.imread(temp_image_path, as_gray=True)
    current_image = io.imread(current_image_path, as_gray=True)

    significant_change = True

    if new_image.shape == current_image.shape:
        similarity_index = ssim(
            new_image, current_image, data_range=new_image.max() - new_image.min()
        )

        significant_change = (
            similarity_index <= threshold
        )  # This needs to be tuned. Minor changes like temp path is expected.)

    if significant_change:
        shutil.move(temp_image_path, current_image_path)

    return significant_change


def assert_error_message(gui_changed):
    newline = "\n         - "
    return dedent(
        f"""
        One or more auto generated images differed from the image used in the docs

        The image(s):
         - {newline.join(gui_changed)}
        has been overwritten with the new version.
        If the new version seems correct it should be kept.
        If the autogenerated image seems incorrect the underlying code generating the
        gui might need corrections or this test needs to be fixed.
        """
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


def open_gui_with_docs_example(tmp_path, example_folder, config_file, random_seed=None):
    def ignore_pngs(src, files):
        return [f for f in files if f.endswith(".png")]

    shutil.copytree(
        os.path.join(
            example_folder,
        ),
        tmp_path,
        ignore=ignore_pngs,
        dirs_exist_ok=True,
    )

    if random_seed is not None:
        with open(config_file, encoding="utf-8") as f:
            original_content = f.read()

        combined_content = (
            f"RANDOM_SEED {random_seed}\n" + original_content
        )  # Add fixed random seed to make the plots reproducible

        with open(config_file, "w", encoding="utf-8") as f:
            f.write(combined_content)

    gui_generator = open_gui_with_config(tmp_path / config_file)
    return next(gui_generator)


def test_that_poly_new_minimal_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)
    gui_changed: list[str] = []

    example_folder = os.path.join(
        source_root, "docs/ert/getting_started/configuration/poly_new/minimal"
    )

    gui = open_gui_with_docs_example(tmp_path, example_folder, "poly.ert")

    compare_img_with_gui(gui_changed, example_folder, "ert.png", gui, qtbot)

    run_experiment(qtbot, EnsembleExperiment, gui)

    compare_img_with_gui(gui_changed, example_folder, "simulations.png", gui, qtbot)

    assert not gui_changed, assert_error_message(gui_changed)


def test_that_poly_new_with_simple_script_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)

    example_folder = os.path.join(
        source_root,
        "docs/ert/getting_started/configuration/poly_new/with_simple_script",
    )

    current_image_path = os.path.join(example_folder, "ert.png")

    gui = open_gui_with_docs_example(tmp_path, example_folder, "poly.ert")

    clean_up_diplayed_runpath(gui)

    assert not gui_has_significant_change(gui, qtbot, current_image_path), (
        assert_error_message([current_image_path])
    )


def test_that_poly_new_with_results_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)
    gui_changed: list[str] = []

    example_folder = os.path.join(
        source_root, "docs/ert/getting_started/configuration/poly_new/with_results"
    )

    gui = open_gui_with_docs_example(tmp_path, example_folder, "poly.ert", 11223344)

    run_experiment(qtbot, EnsembleExperiment, gui)
    open_storage(gui.ert_config.ens_path, mode="w")
    with StorageService.init_service(
        project=os.path.abspath(gui.ert_config.ens_path),
    ):
        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)

        compare_img_with_gui(gui_changed, example_folder, "poly_plot.png", gui, qtbot)

        data_type_widget = wait_for_child(gui, qtbot, DataTypeKeysWidget, "Data types")
        set_data_type_selection_index(data_type_widget, 1)

        compare_img_with_gui(gui_changed, example_folder, "plots.png", gui, qtbot)

    assert not gui_changed, assert_error_message(gui_changed)


def test_that_poly_new_with_observations_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)
    gui_changed: list[str] = []

    example_folder = os.path.join(
        source_root, "docs/ert/getting_started/configuration/poly_new/with_observations"
    )

    gui = open_gui_with_docs_example(
        tmp_path, example_folder, "poly_final.ert", 11223344
    )

    run_experiment(qtbot, EnsembleSmoother, gui)
    open_storage(gui.ert_config.ens_path, mode="w")
    with StorageService.init_service(
        project=os.path.abspath(gui.ert_config.ens_path),
    ):
        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)

        ensamble_selector_widget = wait_for_child(
            gui, qtbot, EnsembleSelectionWidget, "Plot ensemble"
        )

        for index in range(
            ensamble_selector_widget._EnsembleSelectionWidget__dndlist.count()
        ):
            item = ensamble_selector_widget._EnsembleSelectionWidget__dndlist.item(
                index
            )
            item.setData(Qt.ItemDataRole.CheckStateRole, True)
        ensamble_selector_widget._EnsembleSelectionWidget__dndlist.ensembleSelectionListChanged.emit()

        compare_img_with_gui(gui_changed, example_folder, "plot_obs.png", gui, qtbot)

        data_type_widget = wait_for_child(gui, qtbot, DataTypeKeysWidget, "Data types")
        set_data_type_selection_index(data_type_widget, 1)
        compare_img_with_gui(gui_changed, example_folder, "coeff_a.png", gui, qtbot)

        set_data_type_selection_index(data_type_widget, 2)
        compare_img_with_gui(gui_changed, example_folder, "coeff_b.png", gui, qtbot)

        set_data_type_selection_index(data_type_widget, 3)
        compare_img_with_gui(gui_changed, example_folder, "coeff_c.png", gui, qtbot)

    assert not gui_changed, assert_error_message(gui_changed)


def test_that_poly_new_with_more_observations_screenshots_are_up_to_date(
    tmp_path,
    monkeypatch,
    qtbot,
    source_root,
):
    monkeypatch.chdir(tmp_path)
    gui_changed: list[str] = []

    example_folder = os.path.join(
        source_root,
        "docs/ert/getting_started/configuration/poly_new/with_more_observations",
    )

    gui = open_gui_with_docs_example(
        tmp_path, example_folder, "poly_final.ert", 11223344
    )

    run_experiment(qtbot, EnsembleSmoother, gui)
    open_storage(gui.ert_config.ens_path, mode="w")
    with StorageService.init_service(
        project=os.path.abspath(gui.ert_config.ens_path),
    ):
        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)

        ensamble_selector_widget = wait_for_child(
            gui, qtbot, EnsembleSelectionWidget, "Plot ensemble"
        )

        for index in range(
            ensamble_selector_widget._EnsembleSelectionWidget__dndlist.count()
        ):
            item = ensamble_selector_widget._EnsembleSelectionWidget__dndlist.item(
                index
            )
            item.setData(Qt.ItemDataRole.CheckStateRole, True)
        ensamble_selector_widget._EnsembleSelectionWidget__dndlist.ensembleSelectionListChanged.emit()

        data_type_widget = wait_for_child(gui, qtbot, DataTypeKeysWidget, "Data types")
        set_data_type_selection_index(data_type_widget, 2)

        compare_img_with_gui(gui_changed, example_folder, "coeff_b.png", gui, qtbot)

    assert not gui_changed, assert_error_message(gui_changed)
