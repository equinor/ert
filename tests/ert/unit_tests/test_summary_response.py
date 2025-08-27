import shutil
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from packaging import version

from ert.config import ErtConfig
from ert.run_models._create_run_path import create_run_path
from ert.runpaths import Runpaths
from ert.storage.local_ensemble import load_parameters_and_responses_from_runpath


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_load_summary_response_restart_not_zero(
    tmpdir, snapshot, request, storage, run_args
):
    """
    This is a regression test for summary responses where the index map
    was not correctly loaded, this is relevant for restart cases from eclipse.
    The summary file can not be easily created programatically because the
    report steps do not start from 1 as they usually do.
    """
    test_path = Path(request.module.__file__).parent / "summary_response"
    # Numpy 2.3 changed the criteria for displaying float16 and float32 in
    # scientific notation see https://github.com/numpy/numpy/releases/tag/v2.3.0
    # for backwards compatibility in the test we set the legacy print option
    legacy = "2.2" if version.parse(np.__version__) >= version.parse("2.3") else False
    with tmpdir.as_cwd(), np.printoptions(legacy=legacy):
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
            ensemble_size=ert_config.runpath_config.num_realizations,
        )

        create_run_path(
            run_args=run_args(ert_config, ensemble),
            ensemble=ensemble,
            user_config_file=ert_config.user_config_file,
            forward_model_steps=ert_config.forward_model_steps,
            env_vars=ert_config.env_vars,
            env_pr_fm_step=ert_config.env_pr_fm_step,
            substitutions=ert_config.substitutions,
            parameters_file="parameters",
            runpaths=Runpaths.from_config(ert_config),
        )
        shutil.copy(test_path / "PRED_RUN.SMSPEC", sim_path / "PRED_RUN.SMSPEC")
        shutil.copy(test_path / "PRED_RUN.UNSMRY", sim_path / "PRED_RUN.UNSMRY")

        load_parameters_and_responses_from_runpath(
            ert_config.runpath_config.runpath_format_string, ensemble, [0]
        )

        df = ensemble.load_responses("summary", (0,))
        df = df.pivot(on="response_key", values="values")
        df = df[df.columns[:17]]
        df = df.rename({"time": "Date", "realization": "Realization"})

        snapshot.assert_match(
            df.to_pandas().to_csv(index=False),
            "summary_restart",
        )
