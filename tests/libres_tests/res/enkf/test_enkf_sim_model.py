from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import EnKFMain, ResConfig


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "job, forward_model, expected_args",
    [
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            ARGLIST WORD_A
            """
            ),
            "FORWARD_MODEL job_name",
            ["WORD_A"],
            id="No argument",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            ARGLIST <ARGUMENT>
            """
            ),
            "FORWARD_MODEL job_name(<ARGUMENT>=yy)",
            ["yy"],
            id="With argument",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            DEFAULT <ARGUMENTA> DEFAULT_ARGA_VALUE
            ARGLIST <ARGUMENTA> <ARGUMENTB> <ARGUMENTC>
                """
            ),
            dedent(
                """
            DEFINE <TO_BE_DEFINED> <ARGUMENTB>
            FORWARD_MODEL job_name(<ARGUMENTA>=configured_argumentA, <TO_BE_DEFINED>=configured_argumentB)
            """  # noqa E501
            ),
            [
                "configured_argumentA",
                "configured_argumentB",
                "<ARGUMENTC>",
            ],
            id="Using DEFINE, substituting arg name, default argument C",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            DEFAULT <ARGUMENTA> DEFAULT_ARGA_VALUE
            ARGLIST <ARGUMENTA> <ARGUMENTB> <ARGUMENTC>
                """
            ),
            dedent(
                """
            DEFINE <ARGUMENTB> DEFINED_ARGUMENTB_VALUE
            FORWARD_MODEL job_name(<ARGUMENTB>=configured_argumentB)
            """
            ),
            ["DEFAULT_ARGA_VALUE", "DEFINED_ARGUMENTB_VALUE", "<ARGUMENTC>"],
            id="Resolved argument given by DEFINE, even though user specified value",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            DEFAULT <ARGUMENTA> DEFAULT_ARGA_VALUE
            ARGLIST <ARGUMENTA> <ARGUMENTB> <ARGUMENTC>
            """
            ),
            "FORWARD_MODEL job_name()",
            ["DEFAULT_ARGA_VALUE", "<ARGUMENTB>", "<ARGUMENTC>"],
            id="No args, parenthesis, gives default argument A",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            DEFAULT <ARGUMENTA> DEFAULT_ARGA_VALUE
            ARGLIST <ARGUMENTA> <ARGUMENTB> <ARGUMENTC>
            """
            ),
            "FORWARD_MODEL job_name",
            ["DEFAULT_ARGA_VALUE", "<ARGUMENTB>", "<ARGUMENTC>"],
            id="No args, gives default argument A",
        ),
    ],
)
def test_forward_model_job(job, forward_model, expected_args):
    with open("job_file", "w") as fout:
        fout.write(job)

    with open("config_file.ert", "w") as fout:
        # Write a minimal config file
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 1
        """
            )
        )
        fout.write("INSTALL_JOB job_name job_file\n")
        fout.write(forward_model)

    res_config = ResConfig("config_file.ert")
    ert = EnKFMain(res_config)

    model_config = ert.getModelConfig()
    forward_model = model_config.getForwardModel()
    assert forward_model.get_size() == 1
    assert forward_model.iget_job(0).get_argvalues() == expected_args


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "job, forward_model, expected_args",
    [
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            MIN_ARG    1
            MAX_ARG    6
            ARG_TYPE 0 STRING
            ARG_TYPE 1 BOOL
            ARG_TYPE 2 FLOAT
            ARG_TYPE 3 INT
            """
            ),
            "SIMULATION_JOB job_name Hello True 3.14 4",
            ["Hello", "True", "3.14", "4"],
            id="Not all args",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            MIN_ARG    1
            MAX_ARG    2
            ARG_TYPE 0 STRING
            ARG_TYPE 0 STRING
                    """
            ),
            "SIMULATION_JOB job_name word <ECLBASE>",
            ["word", "<ECLBASE>"],
            id="Some args",
        ),
    ],
)
def test_simulation_job(job, forward_model, expected_args):
    with open("job_file", "w") as fout:
        fout.write(job)

    with open("config_file.ert", "w") as fout:
        # Write a minimal config file
        fout.write("NUM_REALIZATIONS 1\n")
        fout.write("INSTALL_JOB job_name job_file\n")
        fout.write(forward_model)

    res_config = ResConfig("config_file.ert")
    ert = EnKFMain(res_config)

    model_config = ert.getModelConfig()
    forward_model = model_config.getForwardModel()
    forward_model_job = forward_model.iget_job(0)
    assert forward_model.get_size() == 1
    assert forward_model_job.get_argvalues() == expected_args
    assert forward_model_job.get_arglist() == expected_args
    assert forward_model_job.valid_args(
        forward_model_job.arg_types, forward_model_job.get_arglist()
    )
