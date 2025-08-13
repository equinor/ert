import json
import os

import numpy as np
import pandas as pd
import pytest

from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict
from tests.everest.utils import (
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
    "WOPR:PROD4",
    "WOPR:PROD3",
    "WOPR:INJECT8",
    "WOPR:INJECT6",
    "WOPR:INJECT7",
    "WOPR:INJECT4",
    "WOPR:INJECT5",
    "WOPR:INJECT2",
    "WOPR:INJECT3",
    "WOPR:INJECT1",
    "WOPR:PROD1",
    "WOPR:PROD2",
    "WOPRT:PROD4",
    "WOPRT:PROD3",
    "WOPRT:INJECT8",
    "WOPRT:INJECT6",
    "WOPRT:INJECT7",
    "WOPRT:INJECT4",
    "WOPRT:INJECT5",
    "WOPRT:INJECT2",
    "WOPRT:INJECT3",
    "WOPRT:INJECT1",
    "WOPRT:PROD1",
    "WOPRT:PROD2",
    "WOPT:PROD4",
    "WOPT:PROD3",
    "WOPT:INJECT8",
    "WOPT:INJECT6",
    "WOPT:INJECT7",
    "WOPT:INJECT4",
    "WOPT:INJECT5",
    "WOPT:INJECT2",
    "WOPT:INJECT3",
    "WOPT:INJECT1",
    "WOPT:PROD1",
    "WOPT:PROD2",
    "WOIR:PROD4",
    "WOIR:PROD3",
    "WOIR:INJECT8",
    "WOIR:INJECT6",
    "WOIR:INJECT7",
    "WOIR:INJECT4",
    "WOIR:INJECT5",
    "WOIR:INJECT2",
    "WOIR:INJECT3",
    "WOIR:INJECT1",
    "WOIR:PROD1",
    "WOIR:PROD2",
    "WOIRT:PROD4",
    "WOIRT:PROD3",
    "WOIRT:INJECT8",
    "WOIRT:INJECT6",
    "WOIRT:INJECT7",
    "WOIRT:INJECT4",
    "WOIRT:INJECT5",
    "WOIRT:INJECT2",
    "WOIRT:INJECT3",
    "WOIRT:INJECT1",
    "WOIRT:PROD1",
    "WOIRT:PROD2",
    "WOIT:PROD4",
    "WOIT:PROD3",
    "WOIT:INJECT8",
    "WOIT:INJECT6",
    "WOIT:INJECT7",
    "WOIT:INJECT4",
    "WOIT:INJECT5",
    "WOIT:INJECT2",
    "WOIT:INJECT3",
    "WOIT:INJECT1",
    "WOIT:PROD1",
    "WOIT:PROD2",
    "WWPR:PROD4",
    "WWPR:PROD3",
    "WWPR:INJECT8",
    "WWPR:INJECT6",
    "WWPR:INJECT7",
    "WWPR:INJECT4",
    "WWPR:INJECT5",
    "WWPR:INJECT2",
    "WWPR:INJECT3",
    "WWPR:INJECT1",
    "WWPR:PROD1",
    "WWPR:PROD2",
    "WWPRT:PROD4",
    "WWPRT:PROD3",
    "WWPRT:INJECT8",
    "WWPRT:INJECT6",
    "WWPRT:INJECT7",
    "WWPRT:INJECT4",
    "WWPRT:INJECT5",
    "WWPRT:INJECT2",
    "WWPRT:INJECT3",
    "WWPRT:INJECT1",
    "WWPRT:PROD1",
    "WWPRT:PROD2",
    "WWPT:PROD4",
    "WWPT:PROD3",
    "WWPT:INJECT8",
    "WWPT:INJECT6",
    "WWPT:INJECT7",
    "WWPT:INJECT4",
    "WWPT:INJECT5",
    "WWPT:INJECT2",
    "WWPT:INJECT3",
    "WWPT:INJECT1",
    "WWPT:PROD1",
    "WWPT:PROD2",
    "WWIR:PROD4",
    "WWIR:PROD3",
    "WWIR:INJECT8",
    "WWIR:INJECT6",
    "WWIR:INJECT7",
    "WWIR:INJECT4",
    "WWIR:INJECT5",
    "WWIR:INJECT2",
    "WWIR:INJECT3",
    "WWIR:INJECT1",
    "WWIR:PROD1",
    "WWIR:PROD2",
    "WWIRT:PROD4",
    "WWIRT:PROD3",
    "WWIRT:INJECT8",
    "WWIRT:INJECT6",
    "WWIRT:INJECT7",
    "WWIRT:INJECT4",
    "WWIRT:INJECT5",
    "WWIRT:INJECT2",
    "WWIRT:INJECT3",
    "WWIRT:INJECT1",
    "WWIRT:PROD1",
    "WWIRT:PROD2",
    "WWIT:PROD4",
    "WWIT:PROD3",
    "WWIT:INJECT8",
    "WWIT:INJECT6",
    "WWIT:INJECT7",
    "WWIT:INJECT4",
    "WWIT:INJECT5",
    "WWIT:INJECT2",
    "WWIT:INJECT3",
    "WWIT:INJECT1",
    "WWIT:PROD1",
    "WWIT:PROD2",
    "WGPR:PROD4",
    "WGPR:PROD3",
    "WGPR:INJECT8",
    "WGPR:INJECT6",
    "WGPR:INJECT7",
    "WGPR:INJECT4",
    "WGPR:INJECT5",
    "WGPR:INJECT2",
    "WGPR:INJECT3",
    "WGPR:INJECT1",
    "WGPR:PROD1",
    "WGPR:PROD2",
    "WGPRT:PROD4",
    "WGPRT:PROD3",
    "WGPRT:INJECT8",
    "WGPRT:INJECT6",
    "WGPRT:INJECT7",
    "WGPRT:INJECT4",
    "WGPRT:INJECT5",
    "WGPRT:INJECT2",
    "WGPRT:INJECT3",
    "WGPRT:INJECT1",
    "WGPRT:PROD1",
    "WGPRT:PROD2",
    "WGPT:PROD4",
    "WGPT:PROD3",
    "WGPT:INJECT8",
    "WGPT:INJECT6",
    "WGPT:INJECT7",
    "WGPT:INJECT4",
    "WGPT:INJECT5",
    "WGPT:INJECT2",
    "WGPT:INJECT3",
    "WGPT:INJECT1",
    "WGPT:PROD1",
    "WGPT:PROD2",
    "WGIR:PROD4",
    "WGIR:PROD3",
    "WGIR:INJECT8",
    "WGIR:INJECT6",
    "WGIR:INJECT7",
    "WGIR:INJECT4",
    "WGIR:INJECT5",
    "WGIR:INJECT2",
    "WGIR:INJECT3",
    "WGIR:INJECT1",
    "WGIR:PROD1",
    "WGIR:PROD2",
    "WGIRT:PROD4",
    "WGIRT:PROD3",
    "WGIRT:INJECT8",
    "WGIRT:INJECT6",
    "WGIRT:INJECT7",
    "WGIRT:INJECT4",
    "WGIRT:INJECT5",
    "WGIRT:INJECT2",
    "WGIRT:INJECT3",
    "WGIRT:INJECT1",
    "WGIRT:PROD1",
    "WGIRT:PROD2",
    "WGIT:PROD4",
    "WGIT:PROD3",
    "WGIT:INJECT8",
    "WGIT:INJECT6",
    "WGIT:INJECT7",
    "WGIT:INJECT4",
    "WGIT:INJECT5",
    "WGIT:INJECT2",
    "WGIT:INJECT3",
    "WGIT:INJECT1",
    "WGIT:PROD1",
    "WGIT:PROD2",
    "WVPR:PROD4",
    "WVPR:PROD3",
    "WVPR:INJECT8",
    "WVPR:INJECT6",
    "WVPR:INJECT7",
    "WVPR:INJECT4",
    "WVPR:INJECT5",
    "WVPR:INJECT2",
    "WVPR:INJECT3",
    "WVPR:INJECT1",
    "WVPR:PROD1",
    "WVPR:PROD2",
    "WVPRT:PROD4",
    "WVPRT:PROD3",
    "WVPRT:INJECT8",
    "WVPRT:INJECT6",
    "WVPRT:INJECT7",
    "WVPRT:INJECT4",
    "WVPRT:INJECT5",
    "WVPRT:INJECT2",
    "WVPRT:INJECT3",
    "WVPRT:INJECT1",
    "WVPRT:PROD1",
    "WVPRT:PROD2",
    "WVPT:PROD4",
    "WVPT:PROD3",
    "WVPT:INJECT8",
    "WVPT:INJECT6",
    "WVPT:INJECT7",
    "WVPT:INJECT4",
    "WVPT:INJECT5",
    "WVPT:INJECT2",
    "WVPT:INJECT3",
    "WVPT:INJECT1",
    "WVPT:PROD1",
    "WVPT:PROD2",
    "WVIR:PROD4",
    "WVIR:PROD3",
    "WVIR:INJECT8",
    "WVIR:INJECT6",
    "WVIR:INJECT7",
    "WVIR:INJECT4",
    "WVIR:INJECT5",
    "WVIR:INJECT2",
    "WVIR:INJECT3",
    "WVIR:INJECT1",
    "WVIR:PROD1",
    "WVIR:PROD2",
    "WVIRT:PROD4",
    "WVIRT:PROD3",
    "WVIRT:INJECT8",
    "WVIRT:INJECT6",
    "WVIRT:INJECT7",
    "WVIRT:INJECT4",
    "WVIRT:INJECT5",
    "WVIRT:INJECT2",
    "WVIRT:INJECT3",
    "WVIRT:INJECT1",
    "WVIRT:PROD1",
    "WVIRT:PROD2",
    "WVIT:PROD4",
    "WVIT:PROD3",
    "WVIT:INJECT8",
    "WVIT:INJECT6",
    "WVIT:INJECT7",
    "WVIT:INJECT4",
    "WVIT:INJECT5",
    "WVIT:INJECT2",
    "WVIT:INJECT3",
    "WVIT:INJECT1",
    "WVIT:PROD1",
    "WVIT:PROD2",
    "WWCT:PROD4",
    "WWCT:PROD3",
    "WWCT:INJECT8",
    "WWCT:INJECT6",
    "WWCT:INJECT7",
    "WWCT:INJECT4",
    "WWCT:INJECT5",
    "WWCT:INJECT2",
    "WWCT:INJECT3",
    "WWCT:INJECT1",
    "WWCT:PROD1",
    "WWCT:PROD2",
    "WGOR:PROD4",
    "WGOR:PROD3",
    "WGOR:INJECT8",
    "WGOR:INJECT6",
    "WGOR:INJECT7",
    "WGOR:INJECT4",
    "WGOR:INJECT5",
    "WGOR:INJECT2",
    "WGOR:INJECT3",
    "WGOR:INJECT1",
    "WGOR:PROD1",
    "WGOR:PROD2",
    "WWGR:PROD4",
    "WWGR:PROD3",
    "WWGR:INJECT8",
    "WWGR:INJECT6",
    "WWGR:INJECT7",
    "WWGR:INJECT4",
    "WWGR:INJECT5",
    "WWGR:INJECT2",
    "WWGR:INJECT3",
    "WWGR:INJECT1",
    "WWGR:PROD1",
    "WWGR:PROD2",
    "WWGRT:PROD4",
    "WWGRT:PROD3",
    "WWGRT:INJECT8",
    "WWGRT:INJECT6",
    "WWGRT:INJECT7",
    "WWGRT:INJECT4",
    "WWGRT:INJECT5",
    "WWGRT:INJECT2",
    "WWGRT:INJECT3",
    "WWGRT:INJECT1",
    "WWGRT:PROD1",
    "WWGRT:PROD2",
    "WBHP:PROD4",
    "WBHP:PROD3",
    "WBHP:INJECT8",
    "WBHP:INJECT6",
    "WBHP:INJECT7",
    "WBHP:INJECT4",
    "WBHP:INJECT5",
    "WBHP:INJECT2",
    "WBHP:INJECT3",
    "WBHP:INJECT1",
    "WBHP:PROD1",
    "WBHP:PROD2",
    "WTHP:PROD4",
    "WTHP:PROD3",
    "WTHP:INJECT8",
    "WTHP:INJECT6",
    "WTHP:INJECT7",
    "WTHP:INJECT4",
    "WTHP:INJECT5",
    "WTHP:INJECT2",
    "WTHP:INJECT3",
    "WTHP:INJECT1",
    "WTHP:PROD1",
    "WTHP:PROD2",
    "WPI:PROD4",
    "WPI:PROD3",
    "WPI:INJECT8",
    "WPI:INJECT6",
    "WPI:INJECT7",
    "WPI:INJECT4",
    "WPI:INJECT5",
    "WPI:INJECT2",
    "WPI:INJECT3",
    "WPI:INJECT1",
    "WPI:PROD1",
    "WPI:PROD2",
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


@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_egg_model_convert_no_opm(copy_egg_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    run_model = EverestRunModel.create(config)

    smry_config = next(
        c for c in run_model.response_configuration if c.type == "summary"
    )

    expected_smry_keys = set(SUM_KEYS_NO_OPM).union({"*"})
    assert set(smry_config.keys) == expected_smry_keys


@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_opm_fail_default_summary_keys(copy_egg_test_data_to_tmp, snapshot):
    pytest.importorskip("everest_models")

    config = EverestConfig.load_file(CONFIG_FILE)
    run_model = EverestRunModel.create(config)

    smry_config = next(
        c for c in run_model.response_configuration if c.type == "summary"
    )

    expected_smry_keys = {k for k in SUM_KEYS[0] if not k.startswith("G")}
    assert set(smry_config.keys) == expected_smry_keys


@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_opm_fail_explicit_summary_keys(copy_egg_test_data_to_tmp):
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
    run_model = EverestRunModel.create(config)

    smry_config = next(
        c for c in run_model.response_configuration if c.type == "summary"
    )

    expected_smry_keys = {k for k in SUM_KEYS[0] if not k.startswith("G")}.union(
        set(extra_sum_keys)
    )
    assert set(smry_config.keys) == expected_smry_keys


@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.integration_test
def test_init_egg_model(copy_egg_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    ert_config = _everest_to_ert_config_dict(
        config, site_config=ErtConfig.read_site_config()
    )
    ErtConfig.with_plugins().from_dict(config_dict=ert_config)


@pytest.mark.integration_test
@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_egg_model_wells_json_output_no_none(copy_egg_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    _ = _everest_to_ert_config_dict(config)

    with open(
        os.path.join(config.output_dir, ".internal_data", "wells.json"),
        encoding="utf-8",
    ) as f:
        data = json.load(f)

        assert data
        assert data[0]["name"] == "PROD1"
        assert all(v for i in data for v in i.values())


@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.requires_eclipse
@pytest.mark.timeout(0)
def test_egg_snapshot(snapshot, copy_egg_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)

    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    best_batch = [
        b for b in run_model._ever_storage.data.everest_batches if b.is_improvement
    ][-1]

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
        tolerance = 1e-15
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
