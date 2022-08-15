from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import ResConfig, EnKFMain
from ert._c_wrappers.enkf.export import GenDataObservationCollector


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_report_steps():
    with open("config_file.ert", "w") as fout:
        # Write a minimal config file
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 1
        OBS_CONFIG observations
        TIME_MAP time_map
        GEN_DATA RESPONSE RESULT_FILE:result_%d.out REPORT_STEPS:0,1 INPUT_FORMAT:ASCII
        """
            )
        )
    with open("obs_data_0.txt", "w") as fout:
        fout.write("1.0 0.1")
    with open("obs_data_1.txt", "w") as fout:
        fout.write("2.0 0.1")

    with open("time_map", "w") as fout:
        fout.write("2014-09-10\n2017-02-05")

    with open("observations", "w") as fout:
        fout.write(
            dedent(
                """
        GENERAL_OBSERVATION OBS_0 {
            DATA       = RESPONSE;
            INDEX_LIST = 0;
            RESTART    = 0;
            OBS_FILE   = obs_data_0.txt;
        };

        GENERAL_OBSERVATION OBS_1 {
            DATA       = RESPONSE;
            INDEX_LIST = 0;
            RESTART    = 1;
            OBS_FILE   = obs_data_1.txt;
        };
        """
            )
        )

    res_config = ResConfig("config_file.ert")
    ert = EnKFMain(res_config)
    obs_key = GenDataObservationCollector.getObservationKeyForDataKey(
        ert, "RESPONSE", 0
    )
    assert obs_key == "OBS_0"

    obs_key = GenDataObservationCollector.getObservationKeyForDataKey(
        ert, "RESPONSE", 1
    )
    assert obs_key == "OBS_1"

    obs_key = GenDataObservationCollector.getObservationKeyForDataKey(
        ert, "RESPONSE", 2
    )
    assert obs_key is None

    obs_key = GenDataObservationCollector.getObservationKeyForDataKey(
        ert, "NOT_A_KEY", 0
    )
    assert obs_key is None
