import os
import stat
import tempfile
from multiprocessing import Process
from pathlib import Path
from textwrap import dedent

import memray
import numpy as np
import polars
import py
import pytest
import xtgeo

from ert.analysis import smoother_update
from ert.config import ErtConfig
from ert.enkf_main import sample_prior
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage
from tests.ert.performance_tests.performance_utils import make_poly_example
from tests.ert.ui_tests.cli.run_cli import run_cli


@pytest.fixture
def poly_template(monkeypatch):
    folder = py.path.local(tempfile.mkdtemp())
    script_path = Path(__file__).parent.resolve()
    folder = make_poly_example(
        folder,
        f"{script_path}/../../../test-data/ert/poly_template",
        gen_data_count=34,
        gen_data_entries=15,
        summary_data_entries=100,
        reals=2,
        summary_data_count=4000,
        sum_obs_count=450,
        gen_obs_count=34,
        sum_obs_every=10,
        gen_obs_every=1,
        parameter_entries=12,
        parameter_count=8,
        update_steps=1,
    )
    monkeypatch.chdir(folder)
    yield folder


@pytest.mark.integration_test
def test_memory_smoothing(poly_template):
    ert_config = ErtConfig.from_file("poly.ert")
    fill_storage_with_data(poly_template, ert_config)
    with open_storage(poly_template / "ensembles", mode="w") as storage:
        experiment = storage.get_experiment_by_name("test-experiment")
        prior_ens = experiment.get_ensemble_by_name("prior")
        posterior_ens = storage.create_ensemble(
            prior_ens.experiment_id,
            ensemble_size=prior_ens.ensemble_size,
            iteration=1,
            name="posterior",
            prior_ensemble=prior_ens,
        )
        with memray.Tracker(poly_template / "memray.bin"):
            smoother_update(
                prior_ens,
                posterior_ens,
                list(experiment.observation_keys),
                list(ert_config.ensemble_config.parameters),
            )

    stats = memray._memray.compute_statistics(str(poly_template / "memray.bin"))
    assert stats.peak_memory_allocated < 1024**2 * 300


def fill_storage_with_data(poly_template: Path, ert_config: ErtConfig) -> None:
    path = Path(poly_template) / "ensembles"
    with open_storage(path, mode="w") as storage:
        ens_config = ert_config.ensemble_config
        experiment_id = storage.create_experiment(
            parameters=ens_config.parameter_configuration,
            responses=ens_config.response_configuration,
            observations=ert_config.observations,
            name="test-experiment",
        )
        source = storage.create_ensemble(experiment_id, name="prior", ensemble_size=100)

        realizations = list(range(ert_config.model_config.num_realizations))
        for real in realizations:
            gendatas = []
            gen_obs = ert_config.observations["gen_data"]
            for response_key, df in gen_obs.group_by("response_key"):
                gendata_df = make_gen_data(df["index"].max() + 1)
                gendata_df = gendata_df.insert_column(
                    0,
                    polars.Series(np.full(len(gendata_df), response_key)).alias(
                        "response_key"
                    ),
                )
                gendatas.append(gendata_df)

            source.save_response("gen_data", polars.concat(gendatas), real)

            obs_time_list = ens_config.refcase.all_dates

            summary_keys = ert_config.observations["summary"]["response_key"].unique(
                maintain_order=True
            )

            source.save_response(
                "summary",
                make_summary_data(summary_keys, obs_time_list),
                real,
            )

        sample_prior(source, realizations, ens_config.parameters)

        storage.create_ensemble(
            source.experiment_id,
            ensemble_size=source.ensemble_size,
            iteration=1,
            name="target_ens",
            prior_ensemble=source,
        )


def make_gen_data(obs: int, min_val: float = 0, max_val: float = 5) -> polars.DataFrame:
    data = np.random.default_rng().uniform(min_val, max_val, obs)
    return polars.DataFrame(
        {
            "report_step": polars.Series(np.full(len(data), 0), dtype=polars.UInt16),
            "index": polars.Series(range(len(data)), dtype=polars.UInt16),
            "values": data,
        }
    )


def make_summary_data(
    obs_keys: list[str],
    dates,
    min_val: float = 0,
    max_val: float = 5,
) -> polars.DataFrame:
    data = np.random.default_rng().uniform(min_val, max_val, len(obs_keys) * len(dates))

    return polars.DataFrame(
        {
            "response_key": np.repeat(obs_keys, len(dates)),
            "time": polars.Series(
                np.tile(dates, len(obs_keys)).tolist()
            ).dt.cast_time_unit("ms"),
            "values": data,
        }
    )


@pytest.mark.limit_memory("130 MB")
@pytest.mark.flaky(reruns=5)
def test_field_param_memory(tmpdir):
    with tmpdir.as_cwd():
        # Setup is done in a subprocess so that memray does not pick up the allocations
        p = Process(target=create_poly_with_field, args=((2000, 1000, 1), 2))
        p.start()
        p.join()  # this blocks until the process terminates

        run_poly()


@pytest.mark.usefixtures("set_site_config")
def create_poly_with_field(field_dim: tuple[int, int, int], realisations: int):
    """
    This replicates the poly example, only it uses FIELD parameter
    """
    grid_size = field_dim[0] * field_dim[1] * field_dim[2]
    config = dedent(
        f"""
            NUM_REALIZATIONS {realisations}
            OBS_CONFIG observations

            FIELD MY_PARAM PARAMETER my_param.bgrdecl INIT_FILES:my_param.bgrdecl FORWARD_INIT:True
            GRID MY_EGRID.EGRID

            GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
            INSTALL_JOB poly_eval POLY_EVAL
            FORWARD_MODEL poly_eval
            """
    )
    with open("config.ert", "w", encoding="utf-8") as fh:
        fh.writelines(config)

    grid = xtgeo.create_box_grid(dimension=field_dim)
    grid.to_file("MY_EGRID.EGRID", "egrid")
    del grid

    with open("forward_model", "w", encoding="utf-8") as f:
        f.write(
            f"""#!/usr/bin/env python
import numpy as np
import os
import resfo

if __name__ == "__main__":
    if not os.path.exists("my_param.bgrdecl"):
        values = np.random.standard_normal({grid_size})
        resfo.write("my_param.bgrdecl", [("MY_PARAM", values)])
    datas = resfo.read("my_param.bgrdecl")
    assert datas[0][0] == "MY_PARAM"
    a,b,c,*_ = datas[0][1]

    output = [float(a) * x**2 + float(b) * x + float(c) for x in range(10)]
    with open("gen_data_0.out", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, output)))
            """
        )
    os.chmod(
        "forward_model",
        os.stat("forward_model").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )
    with open("POLY_EVAL", "w", encoding="utf-8") as fout:
        fout.write("EXECUTABLE forward_model")
    with open("observations", "w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                """
        GENERAL_OBSERVATION MY_OBS {
            DATA       = MY_RESPONSE;
            INDEX_LIST = 0,2,4,6,8;
            RESTART    = 0;
            OBS_FILE   = obs.txt;
        };"""
            )
        )

    with open("obs.txt", "w", encoding="utf-8") as fobs:
        fobs.write(
            dedent(
                """
        2.1457049781272213 0.6
        8.769219841380755 1.4
        12.388014786122742 3.0
        25.600464531354252 5.4
        42.35204755970952 8.6"""
            )
        )


def run_poly():
    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitor",
        "config.ert",
    )
