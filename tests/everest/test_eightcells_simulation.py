import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ert.config import ErtConfig
from ert.config.parsing import ConfigKeys as ErtConfigKeys
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict
from tests.everest.utils import (
    everest_default_jobs,
    skipif_no_everest_models,
)

CONFIG_FILE = "everest/model/config.yml"
NUM_REALIZATIONS = 3  # tied to the specified config.yml defined in CONFIG_FILE
SUM_KEYS_NO_OPM = [
    "YEAR",
    "YEARS",
    "TCPU",
    "TCPUDAY",
    "MONTH",
    "DAY",
    "FOPR",
    "FOPT",
    "FOIR",
    "FOIT",
    "FWPR",
    "FWPT",
    "FWIR",
    "FWIT",
    "FGPR",
    "FGPT",
    "FGIR",
    "FGIT",
    "FVPR",
    "FVPT",
    "FVIR",
    "FVIT",
    "FWCT",
    "FGOR",
    "FOIP",
    "FOIPL",
    "FOIPG",
    "FWIP",
    "FGIP",
    "FGIPL",
    "FGIPG",
    "FPR",
    "FAQR",
    "FAQRG",
    "FAQT",
    "FAQTG",
    "FWGR",
] + [
    ":".join(tup)
    for tup in itertools.product(
        [
            "WBHP",
            "WGIR",
            "WGIRT",
            "WGIT",
            "WGOR",
            "WGPR",
            "WGPRT",
            "WGPT",
            "WOIR",
            "WOIRT",
            "WOIT",
            "WOPR",
            "WOPRT",
            "WOPT",
            "WPI",
            "WTHP",
            "WVIR",
            "WVIRT",
            "WVIT",
            "WVPR",
            "WVPRT",
            "WVPT",
            "WWCT",
            "WWGR",
            "WWGRT",
            "WWIR",
            "WWIRT",
            "WWIT",
            "WWPR",
            "WWPRT",
            "WWPT",
        ],
        ["OP1", "WI1"],
    )
]

SUM_KEYS = [
    ["*"]
    + SUM_KEYS_NO_OPM
    + [
        vector + ":" + group
        for vector in [
            "GGIR",
            "GGIT",
            "GGOR",
            "GGPR",
            "GGPT",
            "GOIR",
            "GOIT",
            "GOPR",
            "GOPT",
            "GVIR",
            "GVIT",
            "GVPR",
            "GVPT",
            "GWCT",
            "GWGR",
            "GWIR",
            "GWIT",
            "GWPR",
            "GWPT",
        ]
        for group in ["FIELD", "INJECT", "PRODUC"]
    ]
]


def sort_res_summary(ert_config):
    ert_config[ErtConfigKeys.SUMMARY][0] = sorted(ert_config[ErtConfigKeys.SUMMARY][0])


def _generate_exp_ert_config(config_path, output_dir, config_file):
    return {
        ErtConfigKeys.DEFINE: [
            ("<CONFIG_PATH>", config_path),
            ("<CONFIG_FILE>", Path(config_file).stem),
        ],
        ErtConfigKeys.INSTALL_JOB: everest_default_jobs(output_dir),
        ErtConfigKeys.NUM_REALIZATIONS: NUM_REALIZATIONS,
        ErtConfigKeys.RUNPATH: os.path.join(
            output_dir,
            "eightcells_simulations/batch_<ITER>/geo_realization_<GEO_ID>/simulation_<IENS>",
        ),
        ErtConfigKeys.RUNPATH_FILE: os.path.join(
            os.path.realpath("everest/model"),
            "everest_output/.res_runpath_list",
        ),
        ErtConfigKeys.MAX_SUBMIT: 2,
        ErtConfigKeys.FORWARD_MODEL: [
            [
                "copy_directory",
                [
                    f"{config_path}/../../eclipse/include/realizations/realization-<GEO_ID>/eclipse",
                    "eclipse",
                ],
            ],
            [
                "symlink",
                [f"{config_path}/../input/files", "files"],
            ],
            [
                "copy_file",
                [
                    os.path.realpath(
                        "everest/model/everest_output/.internal_data/wells.json"
                    ),
                    "wells.json",
                ],
            ],
            [
                "well_constraints",
                [
                    "-i",
                    "files/well_readydate.json",
                    "-c",
                    "files/wc_config.yml",
                    "-rc",
                    "well_rate.json",
                    "-o",
                    "wc_wells.json",
                ],
            ],
            [
                "add_templates",
                [
                    "-i",
                    "wc_wells.json",
                    "-c",
                    "files/at_config.yml",
                    "-o",
                    "at_wells.json",
                ],
            ],
            [
                "schmerge",
                [
                    "-s",
                    "eclipse/include/schedule/schedule.tmpl",
                    "-i",
                    "at_wells.json",
                    "-o",
                    "eclipse/include/schedule/schedule.sch",
                ],
            ],
            [
                "flow",
                ["flow", "eclipse/model/EIGHTCELLS", "--enable-tuning=true"],
            ],
            ["rf", ["-s", "eclipse/model/EIGHTCELLS", "-o", "rf"]],
        ],
        ErtConfigKeys.ENSPATH: os.path.join(
            os.path.realpath("everest/model"),
            "everest_output/simulation_results",
        ),
        ErtConfigKeys.EVEREST_OBJECTIVES: [{"input_file": "rf", "name": "rf"}],
        ErtConfigKeys.ECLBASE: "eclipse/model/EIGHTCELLS",
        ErtConfigKeys.RANDOM_SEED: 123456,
        ErtConfigKeys.SUMMARY: SUM_KEYS,
        ErtConfigKeys.GEN_DATA: [],
    }


@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_conversion_of_eightcells_everestmodel_to_ertmodel(
    copy_eightcells_test_data_to_tmp,
):
    config = EverestConfig.load_file(CONFIG_FILE)
    ert_config = _everest_to_ert_config_dict(config)

    # configpath isn't specified in config_file so it should be inferred
    # to be at the directory of the config file.
    output_dir = config.output_dir
    config_path = os.path.dirname(os.path.abspath(CONFIG_FILE))
    exp_ert_config = _generate_exp_ert_config(config_path, output_dir, CONFIG_FILE)
    exp_ert_config[ErtConfigKeys.SUMMARY][0] = ["*", *SUM_KEYS_NO_OPM]
    sort_res_summary(exp_ert_config)
    sort_res_summary(ert_config)
    assert exp_ert_config == ert_config


@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_opm_fail_default_summary_keys(copy_eightcells_test_data_to_tmp):
    pytest.importorskip("everest_models")

    config = EverestConfig.load_file(CONFIG_FILE)
    # The Everest config file will fail to load as an Eclipse data file
    ert_config = _everest_to_ert_config_dict(config)

    # configpath isn't specified in config_file so it should be inferred
    # to be at the directory of the config file.
    output_dir = config.output_dir
    config_path = os.path.dirname(os.path.abspath(CONFIG_FILE))
    exp_ert_config = _generate_exp_ert_config(config_path, output_dir, CONFIG_FILE)
    exp_ert_config[ErtConfigKeys.SUMMARY][0] = filter(
        lambda key: not key.startswith("G"), exp_ert_config[ErtConfigKeys.SUMMARY][0]
    )
    sort_res_summary(exp_ert_config)
    sort_res_summary(ert_config)
    assert exp_ert_config == ert_config


@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_opm_fail_explicit_summary_keys(copy_eightcells_test_data_to_tmp):
    extra_sum_keys = [
        "GOIR:PRODUC",
        "GOIT:INJECT",
        "GOIT:PRODUC",
        "GWPR:INJECT",
        "GWPR:PRODUC",
        "GWPT:INJECT",
        "GWPT:PRODUC",
        "GWIR:INJECT",
    ]

    config = EverestConfig.load_file(CONFIG_FILE)
    # The Everest config file will fail to load as an Eclipse data file
    config.export.keywords = extra_sum_keys

    ert_config = _everest_to_ert_config_dict(config)

    # configpath isn't specified in config_file so it should be inferred
    # to be at the directory of the config file.
    output_dir = config.output_dir
    config_path = os.path.dirname(os.path.abspath(CONFIG_FILE))
    exp_ert_config = _generate_exp_ert_config(config_path, output_dir, CONFIG_FILE)
    exp_ert_config[ErtConfigKeys.SUMMARY] = [
        list(
            filter(
                lambda key: not key.startswith("G"),
                exp_ert_config[ErtConfigKeys.SUMMARY][0],
            )
        )
        + extra_sum_keys
    ]
    sort_res_summary(exp_ert_config)
    sort_res_summary(ert_config)
    assert exp_ert_config == ert_config


@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.integration_test
def test_init_eightcells_model(copy_eightcells_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    ert_config = _everest_to_ert_config_dict(
        config, site_config=ErtConfig.read_site_config()
    )
    ErtConfig.with_plugins().from_dict(config_dict=ert_config)


@pytest.mark.integration_test
@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_eightcells_model_wells_json_output_no_none(copy_eightcells_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    _ = _everest_to_ert_config_dict(config)

    with open(
        os.path.join(config.output_dir, ".internal_data", "wells.json"),
        encoding="utf-8",
    ) as f:
        data = json.load(f)

        assert data
        assert data[0]["name"] == "OP1"
        assert all(v for i in data for v in i.values())


@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.requires_eclipse
@pytest.mark.timeout(0)
def test_eightcells_snapshot(snapshot, copy_eightcells_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)

    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    best_batch = [b for b in run_model._ever_storage.data.batches if b.is_improvement][
        -1
    ]

    best_controls = best_batch.realization_controls
    best_objectives_csv = best_batch.perturbation_objectives
    best_objective_gradients_csv = best_batch.batch_objective_gradient

    def _is_close(data, snapshot_name):
        data = data.to_pandas()
        snapshot_data = pd.read_csv(snapshot.snapshot_dir / snapshot_name)
        if data.shape != snapshot_data.shape or not all(
            data.columns == snapshot_data.columns
        ):
            raise ValueError(
                f"Dataframes have different structures for {snapshot_name}"
                f"{data}\n\n{snapshot_data}"
            )
        tolerance = 1

        comparison = data.select_dtypes(include=[float, int]).apply(
            lambda col: np.isclose(col, snapshot_data[col.name], atol=tolerance)
        )

        # Check if all values match within the tolerance
        assert comparison.all().all(), (
            f"Values do not match for {snapshot_name} \n{data}\n\n{snapshot_data}"
        )

    _is_close(best_controls, "best_controls")
    _is_close(best_objectives_csv, "best_objectives_csv")
    _is_close(best_objective_gradients_csv, "best_objective_gradients_csv")
