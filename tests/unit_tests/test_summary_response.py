import os
import shutil
from pathlib import Path
from textwrap import dedent

from ert import LibresFacade
from ert.config import ErtConfig
from ert.enkf_main import EnKFMain, create_run_path, ensemble_context


def test_load_summary_response_restart_not_zero(tmpdir, snapshot, request, storage):
    """
    This is a regression test for summary responses where the index map
    was not correctly loaded, this is relevant for restart cases from eclipse.
    The summary file can not be easily created programatically because the
    report steps do not start from 1 as they usually do.
    """
    test_path = Path(request.module.__file__).parent / "summary_response"
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 1
        ECLBASE PRED_RUN
        SUMMARY *
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        cwd = Path().absolute()
        sim_path = Path("simulations") / "realization-0" / "iter-0"
        ert_config = ErtConfig.from_file("config.ert")
        ert = EnKFMain(ert_config)

        experiment_id = storage.create_experiment(
            responses=ert_config.ensemble_config.response_configuration
        )
        ensemble = storage.create_ensemble(
            experiment_id, name="prior", ensemble_size=ert.getEnsembleSize()
        )
        prior = ensemble_context(
            ensemble,
            [True],
            0,
            None,
            "",
            ert_config.model_config.runpath_format_string,
            "name",
        )

        create_run_path(prior, ert_config.substitution_list, ert_config)
        os.chdir(sim_path)
        shutil.copy(test_path / "PRED_RUN.SMSPEC", "PRED_RUN.SMSPEC")
        shutil.copy(test_path / "PRED_RUN.UNSMRY", "PRED_RUN.UNSMRY")
        os.chdir(cwd)

        facade = LibresFacade(ert)
        facade.load_from_forward_model(ensemble, [True], 0)

        snapshot.assert_match(
            facade.load_all_summary_data(ensemble).dropna().iloc[:, :15].to_csv(),
            "summary_restart",
        )
