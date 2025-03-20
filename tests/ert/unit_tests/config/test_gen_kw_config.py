import os
import re
from pathlib import Path
from textwrap import dedent

import pytest
from lark import Token

from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    ErtConfig,
    GenKwConfig,
)
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.config.parsing import ContextString
from ert.config.parsing.file_context_token import FileContextToken
from ert.enkf_main import create_run_path, sample_prior


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_config():
    conf = GenKwConfig(
        name="KEY",
        forward_init=False,
        template_file="",
        transform_function_definitions=[
            TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
            TransformFunctionDefinition("KEY2", "UNIFORM", [0, 1]),
            TransformFunctionDefinition("KEY3", "UNIFORM", [0, 1]),
        ],
        output_file="kw.txt",
        update=True,
    )
    assert len(conf.transform_functions) == 3


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_config_duplicate_keys_raises():
    with pytest.raises(
        ConfigValidationError,
        match="Duplicate GEN_KW keys 'KEY2' found, keys must be unique\\.",
    ):
        GenKwConfig(
            name="KEY",
            forward_init=False,
            template_file="",
            transform_function_definitions=[
                TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
                TransformFunctionDefinition("KEY2", "UNIFORM", [0, 1]),
                TransformFunctionDefinition("KEY2", "UNIFORM", [0, 1]),
                TransformFunctionDefinition("KEY3", "UNIFORM", [0, 1]),
            ],
            output_file="kw.txt",
            update=True,
        )


def test_short_definition_raises_config_error(tmp_path):
    parameter_file = tmp_path / "parameter.txt"
    parameter_file.write_text("incorrect", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Too few values"):
        GenKwConfig.from_config_list(
            [
                "GEN",
                str(parameter_file),
                {"INIT_FILES": "%dgen_init.txt"},
            ]
        )


def test_gen_kw_config_get_priors():
    conf = GenKwConfig(
        name="KW_NAME",
        forward_init=False,
        template_file="template.txt",
        transform_function_definitions=[
            TransformFunctionDefinition("KEY1", "NORMAL", ["0", "1"]),
            TransformFunctionDefinition("KEY2", "LOGNORMAL", ["2", "3"]),
            TransformFunctionDefinition(
                "KEY3", "TRUNCATED_NORMAL", ["4", "5", "6", "7"]
            ),
            TransformFunctionDefinition("KEY4", "TRIANGULAR", ["0", "1", "2"]),
            TransformFunctionDefinition("KEY5", "UNIFORM", ["2", "3"]),
            TransformFunctionDefinition("KEY6", "DUNIF", ["3", "0", "1"]),
            TransformFunctionDefinition("KEY7", "ERRF", ["0", "1", "2", "3"]),
            TransformFunctionDefinition("KEY8", "DERRF", ["1", "1", "2", "3", "4"]),
            TransformFunctionDefinition("KEY9", "LOGUNIF", ["0", "1"]),
            TransformFunctionDefinition("KEY10", "CONST", ["10"]),
        ],
        output_file="param.txt",
        update=True,
    )
    priors = conf.get_priors()
    assert len(conf.transform_functions) == 10

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
        "parameters": {"MIN": 0, "MODE": 1, "MAX": 2},
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
        "parameters": {"STEPS": 1, "MIN": 1, "MAX": 2, "SKEWNESS": 3, "WIDTH": 4},
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
    tmpdir, storage, distribution, expect_log, parameters_regex, run_args, run_paths
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

        gen_kw_config = ert_config.ensemble_config.parameter_configs["KW_NAME"]
        assert isinstance(gen_kw_config, GenKwConfig)
        assert gen_kw_config.shouldUseLogScale("MY_KEYWORD") is expect_log
        assert gen_kw_config.shouldUseLogScale("Non-existent-keyword") is False
        experiment_id = storage.create_experiment(
            parameters=ert_config.ensemble_config.parameter_configuration
        )
        prior_ensemble = storage.create_ensemble(
            experiment_id, name="prior", ensemble_size=1
        )
        sample_prior(prior_ensemble, [0])
        create_run_path(
            run_args=run_args(ert_config, prior_ensemble),
            ensemble=prior_ensemble,
            runpaths=run_paths(ert_config),
            user_config_file=ert_config.user_config_file,
            forward_model_steps=ert_config.forward_model_steps,
            env_vars=ert_config.env_vars,
            env_pr_fm_step=ert_config.env_pr_fm_step,
            substitutions=ert_config.substitutions,
            templates=ert_config.ert_templates,
            model_config=ert_config.model_config,
        )
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
        ("LOGNORMAL", "-1", "1", None),
        ("LOGNORMAL", "0", "-1", ["STD"]),
        ("LOGNORMAL", "-10000", "-1", ["STD"]),
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
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")

        if distribution == "TRUNCATED_NORMAL":
            distribution_line = f"MY_KEYWORD {distribution} {mean} {std} -1 1"
        else:
            distribution_line = f"MY_KEYWORD {distribution} {mean} {std}"

        config_list = [
            "KW_NAME",
            ("template.txt", "MY_KEYWORD <MY_KEYWORD>"),
            "kw.txt",
            ("prior.txt", distribution_line),
            {},
        ]

        if error:
            for e in error:
                with pytest.raises(
                    ConfigValidationError,
                    match=f"Negative {e} {mean if e == 'MEAN' else std}",
                ):
                    GenKwConfig.from_config_list(config_list)
        else:
            GenKwConfig.from_config_list(config_list)


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
        ("MYNAME DERRF 3 1 2 3 4", None),
        ("MYNAME LOGUNIF 0 1", None),
        ("MYNAME CONST 0", None),
        ("MYNAME RAW", None),
        (
            "MYNAME UNIFORM 0 1 2",
            "Incorrect number of values: \\['0', '1', '2'\\], provided for variable MYNAME with distribution UNIFORM.",
        ),
        ("MYNAME", "Too few instructions provided in"),
        (
            "MYNAME RANDOM 0 1",
            "Unknown distribution provided: RANDOM, for variable MYNAME",
        ),
        ("MYNAME DERRF 50 1.12345 2.3 3.14 10E-5", None),
        ("MYNAME DERRF 100 -14 -2.544545 10E5 10E+5", None),
        (
            "MYNAME CONST no-number",
            "Unable to convert 'no-number' to float number for variable MYNAME with distribution CONST.",
        ),
        ("MYNAME      CONST    0", None),  # spaces
        ("MYNAME\t\t\tCONST\t\t0", None),  # tabs
    ],
)
def test_gen_kw_params_parsing(tmpdir, params, error):
    with tmpdir.as_cwd():
        ss = params.split()
        if len(ss) == 1:
            tfd = TransformFunctionDefinition(
                name=ss[0],
                param_name=None,
                values=None,
            )
        else:
            tfd = TransformFunctionDefinition(
                name=ss[0],
                param_name=ss[1],
                values=ss[2:],
            )
        if error:
            with pytest.raises(ConfigValidationError, match=error):
                GenKwConfig(
                    name="MY_PARAM",
                    forward_init=False,
                    update=False,
                    template_file=None,
                    output_file=None,
                    transform_function_definitions=[tfd],
                )
        else:
            GenKwConfig(
                name="MY_PARAM",
                forward_init=False,
                update=False,
                template_file=None,
                output_file=None,
                transform_function_definitions=[tfd],
            )


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
        ("MYNAME ERRF 1 2 0.1 0.1", -1.0, 1.00000000000000000000),
        ("MYNAME ERRF 1 2 0.1 0.1", 0.0, 1.84134474606854281475),
        ("MYNAME ERRF 1 2 0.1 0.1", 0.3, 1.99996832875816688002),
        ("MYNAME ERRF 1 2 0.1 0.1", 0.7, 1.99999999999999933387),
        ("MYNAME ERRF 1 2 0.1 0.1", 1.0, 2.00000000000000000000),
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

    tfd = TransformFunctionDefinition(
        name=params.split()[0],
        param_name=params.split()[1],
        values=params.split()[2:],
    )

    with tmpdir.as_cwd():
        gkw = GenKwConfig(
            name="MY_PARAM",
            forward_init=False,
            update=False,
            template_file=None,
            output_file=None,
            transform_function_definitions=[tfd],
        )
        tf = gkw.transform_functions[0]
        assert abs(tf.calculate(xinput, float_args) - expected) < 10**-15


def test_gen_kw_objects_equal(tmpdir):
    with tmpdir.as_cwd():
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")

        g1 = GenKwConfig.from_config_list(
            [
                "KW_NAME",
                ("template.txt", "MY_KEYWORD <MY_KEYWORD>"),
                "kw.txt",
                ("prior.txt", "MY_KEYWORD UNIFORM 1 2"),
                {},
            ]
        )
        assert g1.transform_functions[0].name == "MY_KEYWORD"

        tfd = TransformFunctionDefinition(
            name="MY_KEYWORD", param_name="UNIFORM", values=["1", "2"]
        )

        g2 = GenKwConfig(
            name="KW_NAME",
            forward_init=False,
            template_file="template.txt",
            transform_function_definitions=[tfd],
            output_file="kw.txt",
            update=True,
        )
        assert g1.name == g2.name
        assert os.path.abspath(g1.template_file) == os.path.abspath(g2.template_file)
        assert (
            g1.transform_function_definitions[0] == g2.transform_function_definitions[0]
        )
        assert g1.output_file == g2.output_file
        assert g1.forward_init_file == g2.forward_init_file

        g3 = GenKwConfig(
            name="KW_NAME2",
            forward_init=False,
            template_file="template.txt",
            transform_function_definitions=[tfd],
            output_file="kw.txt",
            update=True,
        )
        g4 = GenKwConfig(
            name="KW_NAME",
            forward_init=False,
            template_file="empty.txt",
            transform_function_definitions=[tfd],
            output_file="kw.txt",
            update=True,
        )
        g5 = GenKwConfig(
            name="KW_NAME",
            forward_init=False,
            template_file="template.txt",
            transform_function_definitions=[],
            output_file="kw.txt",
            update=True,
        )
        g6 = GenKwConfig(
            name="KW_NAME",
            forward_init=False,
            template_file="template.txt",
            transform_function_definitions=[],
            output_file="empty.txt",
            update=True,
        )

        assert g1 != g3
        assert g1 != g4
        assert g1 != g5
        assert g1 != g6


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_pred_special_suggested_removal():
    with open("coeff_priors.txt", "a", encoding="utf-8") as f:
        f.write("a NORMAL 0 1")
    with open("config.ert", "a", encoding="utf-8") as f:
        f.write(
            "NUM_REALIZATIONS 1\n"
            "GEN_KW PRED coeff_priors.txt coeff_priors.txt coeff_priors.txt\n"
        )
    with pytest.warns(
        ConfigWarning,
        match="GEN_KW PRED used to hold a special meaning and be excluded.*",
    ) as warn_log:
        ErtConfig.from_file("config.ert")
    assert any("config.ert: Line 2" in str(w.message) for w in warn_log)


def make_context_string(msg: str, filename: str) -> ContextString:
    return ContextString.from_token(FileContextToken(Token("UNQUOTED", msg), filename))


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_config_validation():
    with open("template.txt", "w", encoding="utf-8") as f:
        f.write("Hello")

    GenKwConfig.from_config_list(
        [
            "KEY",
            ("template.txt", "Hello"),
            "nothing_here.txt",
            ("parameters.txt", "KEY  UNIFORM 0 1 \n"),
            {},
        ]
    )

    GenKwConfig.from_config_list(
        [
            "KEY",
            ("template.txt", "hello.txt"),
            "nothing_here.txt",
            (
                "parameters_with_comments.txt",
                dedent(
                    """\
                            KEY1  UNIFORM 0 1 -- COMMENT


                            KEY2  UNIFORM 0 1
                            --KEY3
                            ---KEY3
                            ------------
                            KEY3  UNIFORM 0 1
                            """
                ),
            ),
            {},
        ],
    )

    with pytest.raises(
        ConfigValidationError, match=r"config.ert.* No such template file"
    ):
        GenKwConfig.from_config_list(
            [
                "KEY",
                make_context_string("no_template_here.txt", "config.ert"),
                "nothing_here.txt",
                "parameters.txt",
                {},
            ]
        )


def test_incorrect_values_in_forward_init_file_fails(tmp_path):
    (tmp_path / "forward_init_1").write_text("incorrect", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=f"{tmp_path / 'forward_init_1'} did not contain numbers, got object",
    ):
        GenKwConfig(
            "GEN_KW",
            True,
            True,
            None,
            None,
            [],
            str(tmp_path / "forward_init_%d"),
        ).read_from_runpath(tmp_path, 1, 0)


@pytest.mark.usefixtures("use_tmpdir")
def test_suggestion_on_empty_parameter_file():
    Path("empty_template.txt").write_text("", encoding="utf-8")
    with pytest.warns(UserWarning, match="GEN_KW KEY coeffs.txt"):
        GenKwConfig.from_config_list(
            [
                "KEY",
                ("empty_template.txt", ""),
                "output.txt",
                (
                    make_context_string("coeffs.txt", "config.ert"),
                    "a UNIFORM 0 1",
                ),
                {},
            ]
        )


@pytest.mark.parametrize(
    "distribution, min, mode, max, error",
    [
        ("TRIANGULAR", "0", "2", "3", None),
        (
            "TRIANGULAR",
            "3.0",
            "3.0",
            "3.0",
            "Minimum 3.0 must be strictly less than the maximum 3.0",
        ),
        ("TRIANGULAR", "-1", "0", "1", None),
        (
            "TRIANGULAR",
            "3.0",
            "6.0",
            "5.5",
            "The mode 6.0 must be between the minimum 3.0 and maximum 5.5",
        ),
        (
            "TRIANGULAR",
            "3.0",
            "-6.0",
            "5.5",
            "The mode -6.0 must be between the minimum 3.0 and maximum 5.5",
        ),
    ],
)
def test_validation_triangular_distribution(
    tmpdir, distribution, min, mode, max, error
):
    with tmpdir.as_cwd():
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        config_list = [
            "KW_NAME",
            ("template.txt", "MY_KEYWORD <MY_KEYWORD>"),
            "kw.txt",
            ("prior.txt", f"MY_KEYWORD {distribution} {min} {mode} {max}"),
            {},
        ]

        if error:
            with pytest.raises(
                ConfigValidationError,
                match=error,
            ):
                GenKwConfig.from_config_list(config_list)
        else:
            GenKwConfig.from_config_list(config_list)


@pytest.mark.parametrize(
    "distribution, nbins, min, max, skew, width, error",
    [
        ("DERRF", "10", "-1", "3", "-1", "2", None),
        ("DERRF", "100", "-10", "10", "0", "1", None),
        ("DERRF", "2", "-0.5", "0.5", "1", "0.1", None),
        (
            "DERRF",
            "0",
            "-1",
            "3",
            "-1",
            "2",
            "NBINS 0.0 must be a positive integer larger than 1 for DERRF distributed parameter MY_KEYWORD",
        ),
        (
            "DERRF",
            "-5",
            "-1",
            "3",
            "-1",
            "2",
            "NBINS -5.0 must be a positive integer larger than 1 for DERRF distributed parameter MY_KEYWORD",
        ),
        (
            "DERRF",
            "1.5",
            "-1",
            "3",
            "-1",
            "2",
            "NBINS 1.5 must be a positive integer larger than 1 for DERRF distributed parameter MY_KEYWORD",
        ),
        (
            "DERRF",
            "10",
            "3",
            "-1",
            "-1",
            "2",
            "The minimum 3.0 must be less than the maximum -1.0 for DERRF distributed parameter MY_KEYWORD",
        ),
        (
            "DERRF",
            "10",
            "1",
            "1",
            "-1",
            "2",
            "The minimum 1.0 must be less than the maximum 1.0 for DERRF distributed parameter MY_KEYWORD",
        ),
        (
            "DERRF",
            "10",
            "-1",
            "3",
            "-1",
            "0",
            "The width 0.0 must be greater than 0 for DERRF distributed parameter MY_KEYWORD",
        ),
        (
            "DERRF",
            "10",
            "-1",
            "3",
            "-1",
            "-2",
            "The width -2.0 must be greater than 0 for DERRF distributed parameter MY_KEYWORD",
        ),
        (
            "DERRF",
            "2",
            "-999999",
            "999999",
            "0",
            "0.0001",
            None,
        ),
        (
            "DERRF",
            "1000",
            "-0.001",
            "0.001",
            "0",
            "0.0001",
            None,
        ),
    ],
)
def test_validation_derrf_distribution(
    tmpdir, distribution, nbins, min, max, skew, width, error
):
    with tmpdir.as_cwd():
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        config_list = [
            "KW_NAME",
            ("template.txt", "MY_KEYWORD <MY_KEYWORD>"),
            "kw.txt",
            (
                "prior.txt",
                f"MY_KEYWORD {distribution} {nbins} {min} {max} {skew} {width}",
            ),
            {},
        ]

        if error:
            with pytest.raises(
                ConfigValidationError,
                match=error,
            ):
                GenKwConfig.from_config_list(config_list)
        else:
            GenKwConfig.from_config_list(config_list)
