import os
import re
from pathlib import Path
from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import EnKFMain, ErtConfig, GenKwConfig
from ert.parsing import ConfigValidationError


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_config():
    GenKwConfig(
        name="KEY",
        forward_init=False,
        template_file="",
        parameter_file="",
        transfer_function_definitions=["KEY  UNIFORM 0 1"],
        output_file="kw.txt",
    )
    conf = GenKwConfig(
        name="KEY",
        forward_init=False,
        template_file="",
        parameter_file="",
        transfer_function_definitions=[
            "KEY1  UNIFORM 0 1",
            "KEY2 UNIFORM 0 1",
            "KEY3 UNIFORM 0 1",
        ],
        output_file="kw.txt",
    )
    assert len(conf.transfer_functions) == 3


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_config_get_priors():
    parameter_file = "parameters.txt"
    template_file = "template.txt"

    with open(template_file, "w", encoding="utf-8") as f:
        f.write("Hello")

    with open(parameter_file, "w", encoding="utf-8") as f:
        f.write("KEY1  NORMAL 0 1\n")
        f.write("KEY2  LOGNORMAL 2 3\n")
        f.write("KEY3  TRUNCATED_NORMAL 4 5 6 7\n")
        f.write("KEY4  TRIANGULAR 0 1 2\n")
        f.write("KEY5  UNIFORM 2 3\n")
        f.write("KEY6  DUNIF 3 0 1\n")
        f.write("KEY7  ERRF 0 1 2 3\n")
        f.write("KEY8  DERRF 0 1 2 3 4\n")
        f.write("KEY9  LOGUNIF 0 1\n")
        f.write("KEY10  CONST 10\n")

    transfer_function_definitions = []
    with open(parameter_file, "r", encoding="utf-8") as file:
        for item in file:
            transfer_function_definitions.append(item)

    conf = GenKwConfig(
        name="KW_NAME",
        forward_init=False,
        template_file=template_file,
        parameter_file=parameter_file,
        transfer_function_definitions=transfer_function_definitions,
        output_file="param.txt",
    )
    priors = conf.get_priors()
    assert len(conf.transfer_functions) == 10

    assert {
        "key": "KEY1",
        "function": "NORMAL",
        "parameters": {"MEAN": 0, "STD": 1},
    } in priors

    assert {
        "key": "KEY2",
        "function": "LOGNORMAL",
        "parameters": {"MEAN": 2, "STD": 3},
    } in priors

    assert {
        "key": "KEY3",
        "function": "TRUNCATED_NORMAL",
        "parameters": {"MEAN": 4, "STD": 5, "MIN": 6, "MAX": 7},
    } in priors

    assert {
        "key": "KEY4",
        "function": "TRIANGULAR",
        "parameters": {"XMIN": 0, "XMODE": 1, "XMAX": 2},
    } in priors

    assert {
        "key": "KEY5",
        "function": "UNIFORM",
        "parameters": {"MIN": 2, "MAX": 3},
    } in priors

    assert {
        "key": "KEY6",
        "function": "DUNIF",
        "parameters": {"STEPS": 3, "MIN": 0, "MAX": 1},
    } in priors

    assert {
        "key": "KEY7",
        "function": "ERRF",
        "parameters": {"MIN": 0, "MAX": 1, "SKEWNESS": 2, "WIDTH": 3},
    } in priors

    assert {
        "key": "KEY8",
        "function": "DERRF",
        "parameters": {"STEPS": 0, "MIN": 1, "MAX": 2, "SKEWNESS": 3, "WIDTH": 4},
    } in priors

    assert {
        "key": "KEY9",
        "function": "LOGUNIF",
        "parameters": {"MIN": 0, "MAX": 1},
    } in priors

    assert {
        "key": "KEY10",
        "function": "CONST",
        "parameters": {"VALUE": 10},
    } in priors


number_regex = r"[-+]?(?:\d*\.\d+|\d+)"


@pytest.mark.parametrize(
    "distribution, expect_log, parameters_regex",
    [
        ("NORMAL 0 1", False, r"KW_NAME:MY_KEYWORD " + number_regex),
        (
            "LOGNORMAL 0 1",
            True,
            r"KW_NAME:MY_KEYWORD "
            + number_regex
            + r"\n"
            + r"LOG10_KW_NAME:MY_KEYWORD "
            + number_regex,
        ),
        ("UNIFORM 0 1", False, r"KW_NAME:MY_KEYWORD " + number_regex),
        (
            "TRUNCATED_NORMAL 1 0.25 0 10",
            False,
            r"KW_NAME:MY_KEYWORD " + number_regex,
        ),
        (
            "LOGUNIF 0.0001 1",
            True,
            r"KW_NAME:MY_KEYWORD "
            + number_regex
            + r"\n"
            + r"LOG10_KW_NAME:MY_KEYWORD "
            + number_regex,
        ),
        ("CONST 1.0", False, "KW_NAME:MY_KEYWORD 1\n"),
        ("DUNIF 5 1 5", False, r"KW_NAME:MY_KEYWORD " + number_regex),
        ("ERRF 1 2 0.1 0.1", False, r"KW_NAME:MY_KEYWORD " + number_regex),
        ("DERRF 10 1 2 0.1 0.1", False, r"KW_NAME:MY_KEYWORD " + number_regex),
        ("TRIANGULAR 0 0.5 1", False, r"KW_NAME:MY_KEYWORD " + number_regex),
    ],
)
def test_gen_kw_is_log_or_not(
    tmpdir, storage, distribution, expect_log, parameters_regex
):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", "w", encoding="utf-8") as fh:
            fh.writelines(f"MY_KEYWORD {distribution}")

        ert_config = ErtConfig.from_file("config.ert")
        ert = EnKFMain(ert_config)

        gen_kw_config = ert.ensembleConfig().getNode("KW_NAME")
        assert isinstance(gen_kw_config, GenKwConfig)
        assert gen_kw_config.shouldUseLogScale("MY_KEYWORD") is expect_log
        assert gen_kw_config.shouldUseLogScale("Non-existent-keyword") is False
        experiment_id = storage.create_experiment(
            parameters=ert_config.ensemble_config.parameter_configuration
        )
        prior_ensemble = storage.create_ensemble(
            experiment_id, name="prior", ensemble_size=1
        )
        prior = ert.ensemble_context(prior_ensemble, [True], 0)
        ert.sample_prior(prior.sim_fs, prior.active_realizations)
        ert.createRunPath(prior)
        assert re.match(
            parameters_regex,
            Path("simulations/realization-0/iter-0/parameters.txt").read_text(
                encoding="utf-8"
            ),
        ), distribution


@pytest.mark.parametrize(
    "distribution, mean, std, error",
    [
        ("LOGNORMAL", "0", "1", None),
        ("LOGNORMAL", "-1", "1", ["MEAN"]),
        ("LOGNORMAL", "0", "-1", ["STD"]),
        ("LOGNORMAL", "-1", "-1", ["MEAN", "STD"]),
        ("NORMAL", "0", "1", None),
        ("NORMAL", "-1", "1", None),
        ("NORMAL", "0", "-1", ["STD"]),
        ("TRUNCATED_NORMAL", "-1", "1", None),
        ("TRUNCATED_NORMAL", "0", "1", None),
        ("TRUNCATED_NORMAL", "0", "-1", ["STD"]),
    ],
)
def test_gen_kw_distribution_errors(tmpdir, distribution, mean, std, error):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", "w", encoding="utf-8") as fh:
            if distribution == "TRUNCATED_NORMAL":
                fh.writelines(f"MY_KEYWORD {distribution} {mean} {std} -1 1")
            else:
                fh.writelines(f"MY_KEYWORD {distribution} {mean} {std}")

        if error:
            for e in error:
                with pytest.raises(
                    ConfigValidationError,
                    match=f"Negative {e} {mean if e == 'MEAN' else std}",
                ):
                    ErtConfig.from_file("config.ert")
        else:
            ErtConfig.from_file("config.ert")


@pytest.mark.parametrize(
    "params, error",
    [
        ("MYNAME NORMAL 0 1", None),
        ("MYNAME LOGNORMAL 0 1", None),
        ("MYNAME TRUNCATED_NORMAL 0 1 2 3", None),
        ("MYNAME TRIANGULAR 0 1 2", None),
        ("MYNAME UNIFORM 0 1", None),
        ("MYNAME DUNIF 0 1 2", None),
        ("MYNAME ERRF 0 1 2 3", None),
        ("MYNAME DERRF 0 1 2 3 4", None),
        ("MYNAME LOGUNIF 0 1", None),
        ("MYNAME CONST 0", None),
        ("MYNAME RAW", None),
        ("MYNAME UNIFORM 0 1 2", "Incorrect number of values provided"),
        ("MYNAME", "Too few instructions provided in"),
        ("MYNAME RANDOM 0 1", "Unknown transfer function provided"),
        ("MYNAME DERRF -0 1.12345 -2.3 3.14 10E-5", None),
        ("MYNAME DERRF -0 -14 -2.544545 10E5 10E+5", None),
        ("MYNAME CONST no-number", "Unable to convert float number"),
        ("MYNAME      CONST    0", None),  # spaces
        ("MYNAME\t\t\tCONST\t\t0", None),  # tabs
    ],
)
def test_gen_kw_params_parsing(tmpdir, params, error):
    with tmpdir.as_cwd():
        if error:
            with pytest.raises(ConfigValidationError, match=error):
                GenKwConfig._parse_transfer_function(params)
        else:
            GenKwConfig._parse_transfer_function(params)


@pytest.mark.parametrize(
    "params, xinput, expected",
    [
        ("MYNAME TRIANGULAR 0 0.5 1.0", -1.0, 0.28165160565089725209),
        ("MYNAME TRIANGULAR 0 0.5 1.0", 0.0, 0.50000000000000000000),
        ("MYNAME TRIANGULAR 0 0.5 1.0", 0.3, 0.56291386557621880815),
        ("MYNAME TRIANGULAR 0 0.5 1.0", 0.7, 0.65217558149040699700),
        ("MYNAME TRIANGULAR 0 0.5 1.0", 1.0, 0.71834839434910269240),
        ("MYNAME TRIANGULAR 0 1.0 4.0", -1.0, 0.7966310411513150456286),
        ("MYNAME TRIANGULAR 0 1.0 4.0", 1.1, 2.72407181575270778882286),
        ("MYNAME UNIFORM 0 1", -1.0, 0.15865525393145707422),
        ("MYNAME UNIFORM 0 1", 0.0, 0.50000000000000000000),
        ("MYNAME UNIFORM 0 1", 0.3, 0.61791142218895256377),
        ("MYNAME UNIFORM 0 1", 0.7, 0.75803634777692696645),
        ("MYNAME UNIFORM 0 1", 1.0, 0.84134474606854292578),
        ("MYNAME DUNIF 5 1 5", -1.0, 1.00000000000000000000),
        ("MYNAME DUNIF 5 1 5", 0.0, 3.00000000000000000000),
        ("MYNAME DUNIF 5 1 5", 0.3, 4.00000000000000000000),
        ("MYNAME DUNIF 5 1 5", 0.7, 4.00000000000000000000),
        ("MYNAME DUNIF 5 1 5", 1.0, 5.00000000000000000000),
        ("MYNAME CONST 5", -1.0, 5.00000000000000000000),
        ("MYNAME CONST 5", 0.0, 5.00000000000000000000),
        ("MYNAME CONST 5", 0.3, 5.00000000000000000000),
        ("MYNAME CONST 5", 0.7, 5.00000000000000000000),
        ("MYNAME CONST 5", 1.0, 5.00000000000000000000),
        ("MYNAME RAW", -1.0, -1.00000000000000000000),
        ("MYNAME RAW", 0.0, 0.00000000000000000000),
        ("MYNAME RAW", 0.3, 0.29999999999999998890),
        ("MYNAME RAW", 0.7, 0.69999999999999995559),
        ("MYNAME RAW", 1.0, 1.00000000000000000000),
        ("MYNAME LOGUNIF 0.00001 1", -1.0, 0.00006212641160264609),
        ("MYNAME LOGUNIF 0.00001 1", 0.0, 0.00316227766016837896),
        ("MYNAME LOGUNIF 0.00001 1", 0.3, 0.01229014794851427186),
        ("MYNAME LOGUNIF 0.00001 1", 0.7, 0.06168530819028691242),
        ("MYNAME LOGUNIF 0.00001 1", 1.0, 0.16096213739108147789),
        ("MYNAME NORMAL 0 1", -1.0, -1.00000000000000000000),
        ("MYNAME NORMAL 0 1", 0.0, 0.00000000000000000000),
        ("MYNAME NORMAL 0 1", 0.3, 0.29999999999999998890),
        ("MYNAME NORMAL 0 1", 0.7, 0.69999999999999995559),
        ("MYNAME NORMAL 0 1", 1.0, 1.00000000000000000000),
        ("MYNAME LOGNORMAL 0 1", -1.0, 0.36787944117144233402),
        ("MYNAME LOGNORMAL 0 1", 0.0, 1.00000000000000000000),
        ("MYNAME LOGNORMAL 0 1", 0.3, 1.34985880757600318347),
        ("MYNAME LOGNORMAL 0 1", 0.7, 2.01375270747047663278),
        ("MYNAME LOGNORMAL 0 1", 1.0, 2.71828182845904509080),
        ("MYNAME TRUNCATED_NORMAL 1 0.25 0 10", -1.0, 0.75000000000000000000),
        ("MYNAME TRUNCATED_NORMAL 1 0.25 0 10", 0.0, 1.00000000000000000000),
        ("MYNAME TRUNCATED_NORMAL 1 0.25 0 10", 0.3, 1.07499999999999995559),
        ("MYNAME TRUNCATED_NORMAL 1 0.25 0 10", 0.7, 1.17500000000000004441),
        ("MYNAME TRUNCATED_NORMAL 1 0.25 0 10", 1.0, 1.25000000000000000000),
        ("MYNAME ERRF 1 2 0.1 0.1", -1.0, 1.00000000000000000000),
        ("MYNAME ERRF 1 2 0.1 0.1", 0.0, 1.84134474606854281475),
        ("MYNAME ERRF 1 2 0.1 0.1", 0.3, 1.99996832875816688002),
        ("MYNAME ERRF 1 2 0.1 0.1", 0.7, 1.99999999999999933387),
        ("MYNAME ERRF 1 2 0.1 0.1", 1.0, 2.00000000000000000000),
        ("MYNAME DERRF 10 1 2 0.1 0.1", -1.0, 1.00000000000000000000),
        ("MYNAME DERRF 10 1 2 0.1 0.1", 0.0, 1.00000000000000000000),
        ("MYNAME DERRF 10 1 2 0.1 0.1", 0.3, 2.00000000000000000000),
        ("MYNAME DERRF 10 1 2 0.1 0.1", 0.7, 2.00000000000000000000),
        ("MYNAME DERRF 10 1 2 0.1 0.1", 1.0, 2.00000000000000000000),
    ],
)
def test_gen_kw_trans_func(tmpdir, params, xinput, expected):
    """
    This test data was generated using c++ transfer functions, and is used solely
    to verify that implementation in python is equal to c++.
    """
    args = params.split()[2:]
    float_args = []
    for a in args:
        float_args.append(float(a))

    with tmpdir.as_cwd():
        tf = GenKwConfig._parse_transfer_function(params)
        assert abs(tf.calculate(xinput, float_args) - expected) < 10**-15


def test_gen_kw_objects_equal(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD UNIFORM 1 2")
        with open("empty.txt", "w", encoding="utf-8") as fh:
            fh.writelines("")

        ert_config = ErtConfig.from_file("config.ert")
        ert = EnKFMain(ert_config)

        g1 = ert.ensembleConfig()["KW_NAME"]
        assert g1.transfer_functions[0].name == "MY_KEYWORD"

        g2 = GenKwConfig(
            name="KW_NAME",
            forward_init=False,
            template_file="template.txt",
            transfer_function_definitions=["MY_KEYWORD UNIFORM 1 2\n"],
            parameter_file="prior.txt",
            output_file="kw.txt",
        )
        assert g1.name == g2.name
        assert os.path.abspath(g1.template_file) == os.path.abspath(g2.template_file)
        assert os.path.abspath(g1.parameter_file) == os.path.abspath(g2.parameter_file)
        assert g1.output_file == g2.output_file
        assert g1.forward_init_file == g2.forward_init_file

        g3 = GenKwConfig(
            name="KW_NAME2",
            forward_init=False,
            template_file="template.txt",
            transfer_function_definitions=["MY_KEYWORD UNIFORM 1 2\n"],
            parameter_file="prior.txt",
            output_file="kw.txt",
        )
        g4 = GenKwConfig(
            name="KW_NAME",
            forward_init=False,
            template_file="empty.txt",
            transfer_function_definitions=["MY_KEYWORD UNIFORM 1 2\n"],
            parameter_file="prior.txt",
            output_file="kw.txt",
        )
        g5 = GenKwConfig(
            name="KW_NAME",
            forward_init=False,
            template_file="template.txt",
            transfer_function_definitions=[],
            parameter_file="empty.txt",
            output_file="kw.txt",
        )
        g6 = GenKwConfig(
            name="KW_NAME",
            forward_init=False,
            template_file="template.txt",
            parameter_file="prior.txt",
            transfer_function_definitions=[],
            output_file="empty.txt",
        )

        assert g1 != g3
        assert g1 != g4
        assert g1 != g5
        assert g1 != g6
