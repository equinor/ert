import shutil
from pathlib import Path
from textwrap import dedent

from ert import LibresFacade
from ert.config import ErtConfig
from ert.enkf_main import create_run_path


def test_load_summary_response_restart_not_zero(
    tmpdir, snapshot, request, storage, run_paths, run_args
):
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
        sim_path = Path("simulations") / "realization-0" / "iter-0"
        ert_config = ErtConfig.from_file("config.ert")

        experiment_id = storage.create_experiment(
            responses=ert_config.ensemble_config.response_configuration
        )
        ensemble = storage.create_ensemble(
            experiment_id,
            name="prior",
            ensemble_size=ert_config.model_config.num_realizations,
        )

        create_run_path(
            run_args=run_args(ert_config, ensemble),
            ensemble=ensemble,
            user_config_file=ert_config.user_config_file,
            forward_model_steps=ert_config.forward_model_steps,
            env_vars=ert_config.env_vars,
            substitutions=ert_config.substitutions,
            templates=ert_config.ert_templates,
            model_config=ert_config.model_config,
            runpaths=run_paths(ert_config),
        )
        shutil.copy(test_path / "PRED_RUN.SMSPEC", sim_path / "PRED_RUN.SMSPEC")
        shutil.copy(test_path / "PRED_RUN.UNSMRY", sim_path / "PRED_RUN.UNSMRY")

        facade = LibresFacade.from_config_file("config.ert")
        facade.load_from_forward_model(ensemble, [True], 0)

        df = ensemble.load_responses("summary", (0,))
        df = df.pivot(on="response_key", values="values")
        df = df[df.columns[:17]]
        df = df.rename({"time": "Date", "realization": "Realization"})

        snapshot.assert_match(
            df.to_pandas().to_csv(index=False),
            "summary_restart",
        )
