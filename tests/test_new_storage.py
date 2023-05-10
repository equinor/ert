import json
import os
from pathlib import Path
from textwrap import dedent

import numpy as np
from ert._c_wrappers.enkf.enums import RealizationStateEnum

from ert import LibresFacade
from ert._c_wrappers.enkf import ErtConfig, EnKFMain
from ert.analysis import ESUpdate


def _load_coeffs(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def _evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


def test_running_storage(tmpdir, storage):
    ensemble_size = 500
    nr_responses = 100
    with tmpdir.as_cwd():
        config = dedent(
            f"""
NUM_REALIZATIONS {ensemble_size}
QUEUE_OPTION LOCAL MAX_RUNNING 50
OBS_CONFIG observations
GEN_KW COEFFS coeff.tmpl coeffs.json coeff_priors
GEN_DATA POLY_RES RESULT_FILE:poly_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
TIME_MAP time_map
        """
        )

        with open("observations", "w", encoding="utf-8") as fout:
            fout.write(
                dedent(
                    """
            GENERAL_OBSERVATION MY_OBS {
                DATA       = POLY_RES;
                RESTART    = 0;
                OBS_FILE   = obs.txt;
            };"""
                )
            )

        with open("coeff_priors", "w", encoding="utf-8") as fout:
            fout.write(
                dedent(
                    """
                    COEFF_A UNIFORM 0 1
                    COEFF_B UNIFORM 0 2
                    COEFF_C UNIFORM 0 5
                """
                )
            )

        with open("coeff.tmpl", "w", encoding="utf-8") as fout:
            fout.write(
                dedent(
                    """
                    {
                      "a": <COEFF_A>,
                      "b": <COEFF_B>,
                      "c": <COEFF_C>
                    }
                """
                )
            )

        with open("obs.txt", "w", encoding="utf-8") as fobs:
            rng = np.random.default_rng(12345)

            def p(x):
                return 0.5 * x**2 + x + 3

            obs = [
                f"{p(x) + rng.normal(loc=0, scale=0.2 * p(x))} {0.2 * p(x)}"
                for x in range(nr_responses)
            ]
            fobs.write(f"\n".join(obs))

        with open("time_map", "w", encoding="utf-8") as fobs:
            fobs.write("2014-09-10")

        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        ert_config = ErtConfig.from_file("config.ert")
        ert = EnKFMain(ert_config)
        experiment_id = storage.create_experiment(
            parameters=ert_config.ensemble_config.parameter_configuration
        )
        prior_ensemble = storage.create_ensemble(
            experiment_id, name="prior", ensemble_size=ensemble_size
        )
        run_context = ert.ensemble_context(
            prior_ensemble, [True] * ensemble_size, iteration=0
        )
        new_ensemble = storage.create_ensemble(
            experiment_id, name="posterior", ensemble_size=ensemble_size
        )
        import time

        start_time = time.time()
        ert.sample_prior(prior_ensemble, list(range(ensemble_size)))
        print(f"Time for sampling: {time.time() - start_time}")
        start_time = time.time()
        ert.createRunPath(run_context)
        print(f"Time for creating run_path: {time.time() - start_time}")

        cwd = os.getcwd()
        for real in range(ensemble_size):
            run_path = Path(cwd) / f"simulations/realization-{real}/iter-0/"
            os.chdir(run_path)
            coeffs = _load_coeffs("coeffs.json")
            output = [_evaluate(coeffs, x) for x in range(nr_responses)]
            with open("poly_0.out", "w", encoding="utf-8") as f:
                f.write("\n".join(map(str, output)))
        os.chdir(cwd)

        facade = LibresFacade(ert)
        start_time = time.time()
        facade.load_from_forward_model(prior_ensemble, [True] * ensemble_size, 0)
        print(f"Time for loading: {time.time() - start_time}")
        es_update = ESUpdate(ert)
        start_time = time.time()
        es_update.smootherUpdate(prior_ensemble, new_ensemble, "prior.run_id")
        print(f"Time for update: {time.time() - start_time}")
