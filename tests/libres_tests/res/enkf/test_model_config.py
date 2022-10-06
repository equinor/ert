import pytest

from ert._c_wrappers.enkf import ConfigKeys, ModelConfig, ResConfig
from ert._c_wrappers.sched import HistorySourceEnum


@pytest.mark.usefixtures("copy_minimum_case")
def test_eclbase_and_jobname():
    res_config = ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "JOBNAME%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 1,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
                "ECLBASE": "ECLBASE%d",
            },
        }
    )
    assert res_config.model_config.getJobnameFormat() == "JOBNAME%d"
    assert res_config.ecl_config.active()


@pytest.mark.usefixtures("copy_minimum_case")
def test_eclbase():
    res_config = ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "SIMULATION": {
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 1,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
                "ECLBASE": "ECLBASE%d",
            },
        }
    )

    assert res_config.ecl_config.active()
    assert res_config.model_config.getJobnameFormat() == "ECLBASE%d"


@pytest.mark.usefixtures("copy_minimum_case")
def test_jobname():
    res_config = ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "JOBNAME%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 1,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
            },
        }
    )
    assert not res_config.ecl_config.active()
    assert res_config.model_config.getJobnameFormat() == "JOBNAME%d"


def test_model_config_dict_constructor(setup_case):
    res_config = setup_case("configuration_tests", "model_config.ert")
    assert res_config.model_config == ModelConfig(
        data_root="",
        joblist=res_config.site_config.get_installed_jobs(),
        last_history_restart=res_config.ecl_config.getLastHistoryRestart(),
        refcase=res_config.ecl_config.getRefcase(),
        config_dict={
            ConfigKeys.MAX_RESAMPLE: 1,
            ConfigKeys.JOBNAME: "model_config_test",
            ConfigKeys.RUNPATH: "/tmp/simulations/run%d",
            ConfigKeys.NUM_REALIZATIONS: 10,
            ConfigKeys.ENSPATH: "Ensemble",
            ConfigKeys.TIME_MAP: "input/refcase/time_map.txt",
            ConfigKeys.OBS_CONFIG: ("input/observations/observations.txt"),
            ConfigKeys.DATAROOT: ".",
            ConfigKeys.HISTORY_SOURCE: HistorySourceEnum(1),
            ConfigKeys.GEN_KW_EXPORT_NAME: "parameter_test.json",
            ConfigKeys.FORWARD_MODEL: [
                {
                    ConfigKeys.NAME: "COPY_FILE",
                    ConfigKeys.ARGLIST: (
                        "<FROM>=input/schedule.sch, " "<TO>=output/schedule_copy.sch"
                    ),
                },
                {
                    ConfigKeys.NAME: "SNAKE_OIL_SIMULATOR",
                    ConfigKeys.ARGLIST: "",
                },
                {
                    ConfigKeys.NAME: "SNAKE_OIL_NPV",
                    ConfigKeys.ARGLIST: "",
                },
                {
                    ConfigKeys.NAME: "SNAKE_OIL_DIFF",
                    ConfigKeys.ARGLIST: "",
                },
            ],
        },
    )


def test_schedule_file_as_history_is_disallowed(setup_case):
    with pytest.raises(ValueError) as cm:
        setup_case("configuration_tests", "sched_file_as_history_source.ert")

        # Any assert should per the unittest documentation be outside the
        # scope of the assertRaises with-block.
        assert (
            f"{HistorySourceEnum.SCHEDULE} as "
            f"{ConfigKeys.HISTORY_SOURCE} is not supported"
        ) in str(cm.exception)
