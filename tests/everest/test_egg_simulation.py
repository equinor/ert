import json
import os

import pytest

import everest
from ert.config import ErtConfig, QueueSystem
from everest.config import EverestConfig
from everest.config.export_config import ExportConfig
from everest.config_keys import ConfigKeys
from everest.export import MetaDataColumnNames
from everest.plugins.site_config_env import PluginSiteConfigEnv
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict
from tests.everest.utils import (
    everest_default_jobs,
    hide_opm,
    skipif_no_everest_models,
    skipif_no_opm,
    skipif_no_simulator,
)

CONFIG_FILE = "everest/model/config.yml"
SUM_KEYS_NO_OPM = [
    "YEAR",
    "YEARSTCPU",
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
    SUM_KEYS_NO_OPM
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
    ert_config["SUMMARY"][0] = sorted(ert_config["SUMMARY"][0])


def _generate_exp_ert_config(config_path, output_dir):
    return {
        "DEFINE": [("<CONFIG_PATH>", config_path)],
        "INSTALL_JOB": everest_default_jobs(output_dir),
        "QUEUE_OPTION": [(QueueSystem.LOCAL, "MAX_RUNNING", 3)],
        "QUEUE_SYSTEM": QueueSystem.LOCAL,
        "NUM_REALIZATIONS": 10000,
        "RUNPATH": os.path.join(
            output_dir,
            "egg_simulations/<CASE_NAME>/geo_realization_<GEO_ID>/simulation_<IENS>",
        ),
        "RUNPATH_FILE": os.path.join(
            os.path.realpath("everest/model"),
            "everest_output/.res_runpath_list",
        ),
        "SIMULATION_JOB": [
            (
                "copy_directory",
                f"{config_path}/../../eclipse/include/"
                "realizations/realization-<GEO_ID>/eclipse",
                "eclipse",
            ),
            (
                "symlink",
                "{config_path}/../input/files".format(config_path=config_path),
                "files",
            ),
            (
                "copy_file",
                os.path.realpath(
                    "everest/model/everest_output/.internal_data/wells.json"
                ),
                "wells.json",
            ),
            (
                "well_constraints",
                "-i",
                "files/well_readydate.json",
                "-c",
                "files/wc_config.yml",
                "-rc",
                "well_rate.json",
                "-o",
                "wc_wells.json",
            ),
            (
                "add_templates",
                "-i",
                "wc_wells.json",
                "-c",
                "files/at_config.yml",
                "-o",
                "at_wells.json",
            ),
            (
                "schmerge",
                "-s",
                "eclipse/include/schedule/schedule.tmpl",
                "-i",
                "at_wells.json",
                "-o",
                "eclipse/include/schedule/schedule.sch",
            ),
            (
                "eclipse100",
                "eclipse/model/EGG.DATA",
                "--version",
                "2020.2",
            ),
            (
                "strip_dates",
                "--summary",
                "<ECLBASE>.UNSMRY",
                "--dates",
                "2014-05-30",
                "2014-08-28",
                "2014-11-26",
                "2015-02-24",
                "2015-05-25",
                "2015-08-23",
                "2015-11-21",
                "2016-02-19",
                "2016-05-19",
            ),
            ("rf", "-s", "eclipse/model/EGG", "-o", "rf"),
        ],
        "ENSPATH": os.path.join(
            os.path.realpath("everest/model"),
            "everest_output/simulation_results",
        ),
        "ECLBASE": "eclipse/model/EGG",
        "RANDOM_SEED": 123456,
        "SUMMARY": SUM_KEYS,
        "GEN_DATA": [("rf", "RESULT_FILE:rf")],
    }


@skipif_no_opm
def test_egg_model_convert(copy_egg_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    ert_config = _everest_to_ert_config_dict(config)

    # configpath isn't specified in config_file so it should be inferred
    # to be at the directory of the config file.
    output_dir = config.output_dir
    config_path = os.path.dirname(os.path.abspath(CONFIG_FILE))
    exp_ert_config = _generate_exp_ert_config(config_path, output_dir)
    sort_res_summary(exp_ert_config)
    sort_res_summary(ert_config)
    assert exp_ert_config == ert_config


@hide_opm
@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_egg_model_convert_no_opm(copy_egg_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    ert_config = _everest_to_ert_config_dict(config)

    # configpath isn't specified in config_file so it should be inferred
    # to be at the directory of the config file.
    output_dir = config.output_dir
    config_path = os.path.dirname(os.path.abspath(CONFIG_FILE))
    exp_ert_config = _generate_exp_ert_config(config_path, output_dir)
    exp_ert_config["SUMMARY"][0] = SUM_KEYS_NO_OPM
    sort_res_summary(exp_ert_config)
    sort_res_summary(ert_config)
    assert exp_ert_config == ert_config


@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_opm_fail_default_summary_keys(copy_egg_test_data_to_tmp):
    pytest.importorskip("everest_models")

    config = EverestConfig.load_file(CONFIG_FILE)
    # The Everest config file will fail to load as an Eclipse data file
    config.model.data_file = os.path.realpath(CONFIG_FILE)
    assert len(EverestConfig.lint_config_dict(config.to_dict())) == 0

    ert_config = _everest_to_ert_config_dict(config)

    # configpath isn't specified in config_file so it should be inferred
    # to be at the directory of the config file.
    output_dir = config.output_dir
    config_path = os.path.dirname(os.path.abspath(CONFIG_FILE))
    exp_ert_config = _generate_exp_ert_config(config_path, output_dir)
    exp_ert_config["SUMMARY"][0] = filter(
        lambda key: not key.startswith("G"), exp_ert_config["SUMMARY"][0]
    )
    sort_res_summary(exp_ert_config)
    sort_res_summary(ert_config)
    assert exp_ert_config == ert_config


@skipif_no_everest_models
@pytest.mark.everest_models_test
@skipif_no_opm
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
    config.model.data_file = os.path.realpath(CONFIG_FILE)
    if ConfigKeys.EXPORT not in config:
        config.export = ExportConfig()

    config.export.keywords = extra_sum_keys
    assert len(EverestConfig.lint_config_dict(config.to_dict())) == 0

    ert_config = _everest_to_ert_config_dict(config)

    # configpath isn't specified in config_file so it should be inferred
    # to be at the directory of the config file.
    output_dir = config.output_dir
    config_path = os.path.dirname(os.path.abspath(CONFIG_FILE))
    exp_ert_config = _generate_exp_ert_config(config_path, output_dir)
    exp_ert_config["SUMMARY"] = [
        list(
            filter(
                lambda key: not key.startswith("G"),
                exp_ert_config["SUMMARY"][0],
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
def test_init_egg_model(copy_egg_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    ert_config = _everest_to_ert_config_dict(
        config, site_config=ErtConfig.read_site_config()
    )
    ErtConfig.with_plugins().from_dict(config_dict=ert_config)


@skipif_no_everest_models
@pytest.mark.everest_models_test
@skipif_no_simulator
@pytest.mark.requires_eclipse
def test_run_egg_model(copy_egg_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)

    # test callback
    class CBTracker(object):
        def __init__(self):
            self.called = False

        def sweetcallbackofmine(self, *args, **kwargs):
            self.called = True

    cbtracker = CBTracker()
    workflow = everest.suite._EverestWorkflow(
        config=config, simulation_callback=cbtracker.sweetcallbackofmine
    )
    assert workflow is not None
    with PluginSiteConfigEnv():
        workflow.start_optimization()

    assert cbtracker.called
    # TODO: The comparison is currently disabled because we know it would
    # fail. 0.851423 is indeed the optimal value, but the underlying
    # optimization algorithm (newton) is unable to find the optimum for a
    # well drill problem. We believe this is because newton is gradient
    # based, so it works ok for continuous problems, but well drill is
    # highly discontinuous.
    # As soon as a solution for this problem is found, this comparison will
    # be enabled again; high delta for now.

    # self.assertAlmostEqual(result.total_objective, 0.851423, delta=0.5)

    # Test conversion to pandas DataFrame
    df = everest.export(config)

    # Check meta data export
    for meta_key in MetaDataColumnNames.get_all():
        assert meta_key in df

    # Check control export
    cgname = config.controls[0].name
    well_names = [well.name for well in config.wells]
    for wname in well_names:
        assert "{cg}_{w}-1".format(cg=cgname, w=wname) in df

    # Check objective export
    objective_names = [objf.name for objf in config.objective_functions]
    for objective_name in objective_names:
        assert objective_name in df

    exp_keywords = [
        "FOPT",
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
        "well_rate_INJECT5-1",
        "well_rate_INJECT4-1",
        "well_rate_INJECT7-1",
        "well_rate_INJECT6-1",
        "well_rate_INJECT1-1",
        "well_rate_INJECT3-1",
        "rf",
        "well_rate_INJECT2-1",
        "rf_norm",
        "well_rate_INJECT8-1",
        "well_rate_PROD3-1",
        "well_rate_PROD2-1",
        "well_rate_PROD1-1",
        "rf_weighted_norm",
        "well_rate_PROD4-1",
    ]

    # Check summary keys
    for summary_key in exp_keywords:
        assert summary_key in df

    # Check length
    num_dates = len(set(df[MetaDataColumnNames.SIMULATED_DATE]))
    assert num_dates > 0
    num_real = len(config.model.realizations)
    pert_num = config.optimization.perturbation_num or 5
    assert df.shape[0] >= num_dates * num_real * (1 + pert_num)

    # Check export filter
    config.export = ExportConfig(keywords=["*OPT*"])

    filtered_df = everest.export(config)

    exp_keywords += MetaDataColumnNames.get_all()
    columns = sorted(set(filtered_df.columns))
    for expected_key in sorted(set(exp_keywords)):
        assert expected_key in columns


@skipif_no_everest_models
@pytest.mark.everest_models_test
@skipif_no_opm
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
@skipif_no_simulator
@pytest.mark.requires_eclipse
@pytest.mark.timeout(0)
def test_egg_snapshot(snapshot, copy_egg_test_data_to_tmp):
    # shutil.copytree(relpath(ROOT), tmp_path, dirs_exist_ok=True)
    # monkeypatch.chdir(tmp_path)
    config = EverestConfig.load_file(CONFIG_FILE)

    class CBTracker(object):
        def __init__(self):
            self.called = False

        def sweetcallbackofmine(self, *args, **kwargs):
            self.called = True

    cbtracker = CBTracker()
    workflow = everest.suite._EverestWorkflow(
        config=config, simulation_callback=cbtracker.sweetcallbackofmine
    )
    assert workflow is not None
    with PluginSiteConfigEnv():
        workflow.start_optimization()

    assert cbtracker.called

    snapshot.assert_match(
        everest.export(config)
        .drop(columns=["TCPUDAY", "start_time", "end_time"], axis=1)
        .round(6)
        .to_csv(),
        "egg.csv",
    )
