import math
import re
from pathlib import Path
from textwrap import dedent

import networkx as nx
import pytest
from lark import Token

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig, GenKwConfig
from ert.config.parsing.file_context_token import FileContextToken
from ert.run_models._create_run_path import create_run_path
from ert.runpaths import Runpaths
from ert.sample_prior import sample_prior


def test_short_definition_raises_config_error(tmp_path):
    parameter_file = tmp_path / "parameter.txt"
    parameter_file.write_text("incorrect", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Too few values"):
        GenKwConfig.from_config_list(
            [
                "GEN",
                str(parameter_file),
                {},
            ]
        )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            {"name": "KEY1", "distribution": {"name": "normal", "mean": 0, "std": 1}},
            {"key": "KEY1", "function": "NORMAL", "parameters": {"MEAN": 0, "STD": 1}},
        ),
        (
            {
                "name": "KEY2",
                "distribution": {"name": "lognormal", "mean": 2, "std": 3},
            },
            {
                "key": "KEY2",
                "function": "LOGNORMAL",
                "parameters": {"MEAN": 2, "STD": 3},
            },
        ),
        (
            {
                "name": "KEY3",
                "distribution": {
                    "name": "truncated_normal",
                    "mean": 4,
                    "std": 5,
                    "min": 6,
                    "max": 7,
                },
            },
            {
                "key": "KEY3",
                "function": "TRUNCATED_NORMAL",
                "parameters": {"MEAN": 4, "STD": 5, "MIN": 6, "MAX": 7},
            },
        ),
        (
            {
                "name": "KEY4",
                "distribution": {"name": "triangular", "min": 0, "mode": 1, "max": 2},
            },
            {
                "key": "KEY4",
                "function": "TRIANGULAR",
                "parameters": {"MIN": 0, "MODE": 1, "MAX": 2},
            },
        ),
        (
            {"name": "KEY5", "distribution": {"name": "uniform", "min": 2, "max": 3}},
            {"key": "KEY5", "function": "UNIFORM", "parameters": {"MIN": 2, "MAX": 3}},
        ),
        (
            {
                "name": "KEY6",
                "distribution": {"name": "dunif", "steps": 3, "min": 0, "max": 1},
            },
            {
                "key": "KEY6",
                "function": "DUNIF",
                "parameters": {"STEPS": 3, "MIN": 0, "MAX": 1},
            },
        ),
        (
            {
                "name": "KEY7",
                "distribution": {
                    "name": "errf",
                    "min": 0,
                    "max": 1,
                    "skewness": 2,
                    "width": 3,
                },
            },
            {
                "key": "KEY7",
                "function": "ERRF",
                "parameters": {"MIN": 0, "MAX": 1, "SKEWNESS": 2, "WIDTH": 3},
            },
        ),
        (
            {
                "name": "KEY8",
                "distribution": {
                    "name": "derrf",
                    "steps": 1,
                    "min": 1,
                    "max": 2,
                    "skewness": 3,
                    "width": 4,
                },
            },
            {
                "key": "KEY8",
                "function": "DERRF",
                "parameters": {
                    "STEPS": 1,
                    "MIN": 1,
                    "MAX": 2,
                    "SKEWNESS": 3,
                    "WIDTH": 4,
                },
            },
        ),
        (
            {"name": "KEY9", "distribution": {"name": "logunif", "min": 1, "max": 2}},
            {"key": "KEY9", "function": "LOGUNIF", "parameters": {"MIN": 1, "MAX": 2}},
        ),
        (
            {"name": "KEY10", "distribution": {"name": "const", "value": 10}},
            {"key": "KEY10", "function": "CONST", "parameters": {"VALUE": 10}},
        ),
    ],
    ids=[f"KEY{i}" for i in range(1, 11)],
)
def test_gen_kw_config_get_priors(spec, expected):
    cfg = GenKwConfig(**spec)
    assert expected in cfg.get_priors()


number_regex = r"[-+]?(?:\d*\.\d+|\d+)"


@pytest.mark.parametrize(
    ("distribution", "expect_log", "parameters_regex"),
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
    tmpdir, storage, distribution, expect_log, parameters_regex, run_args
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

        gen_kw_config = ert_config.ensemble_config.parameter_configs["MY_KEYWORD"]
        assert isinstance(gen_kw_config, GenKwConfig)
        experiment_id = storage.create_experiment(
            parameters=ert_config.ensemble_config.parameter_configuration
        )
        prior_ensemble = storage.create_ensemble(
            experiment_id, name="prior", ensemble_size=1
        )
        sample_prior(prior_ensemble, [0], 123)
        create_run_path(
            run_args=run_args(ert_config, prior_ensemble),
            ensemble=prior_ensemble,
            runpaths=Runpaths.from_config(ert_config),
            user_config_file=ert_config.user_config_file,
            forward_model_steps=ert_config.forward_model_steps,
            env_vars=ert_config.env_vars,
            env_pr_fm_step=ert_config.env_pr_fm_step,
            substitutions=ert_config.substitutions,
            parameters_file="parameters",
        )
        assert re.match(
            parameters_regex,
            Path("simulations/realization-0/iter-0/parameters.txt").read_text(
                encoding="utf-8"
            ),
        ), distribution


@pytest.mark.parametrize(
    ("distribution", "mean", "std", "error"),
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


def test_that_high_mean_stddev_lognormal_gives_warning():
    mean_log = 4
    stdev_log = 4
    expected_warning = r"Expectation value of the lognormal distribution is.*"
    with pytest.warns(
        ConfigWarning,
        match=expected_warning,
    ) as _:
        GenKwConfig(
            name="KEY1",
            distribution={"name": "lognormal", "mean": mean_log, "std": stdev_log},
        )


def test_that_very_high_mean_stddev_lognormal_gives_error():
    mean_log = 3000
    stdev_log = 10
    expected_error = r"Expectation value of the lognormal distribution is too large!.*"
    config_list = [
        "COEFFS",
        ["coeff_priors", f"MYVARIABLE LOGNORMAL {mean_log} {stdev_log}"],
        {},
    ]
    with pytest.raises(
        ConfigValidationError,
        match=expected_error,
    ):
        GenKwConfig.from_config_list(config_list)


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ("MYNAME NORMAL 0 1", None),
        ("MYNAME LOGNORMAL 0 1", None),
        (
            "MYNAME TRUNCATED_NORMAL 0 1 3 2",
            (
                "Minimum 3.0 must be strictly less than the maximum"
                " 2.0 for truncated_normal distribution"
            ),
        ),
        ("MYNAME TRUNCATED_NORMAL 0 1 2 3", None),
        ("MYNAME TRIANGULAR 0 1 2", None),
        ("MYNAME UNIFORM 0 1", None),
        ("MYNAME DUNIF 2 1 2", None),
        (
            "MYNAME DUNIF 0 1 2",
            "Number of steps 0 must be larger than 1 for duniform distribution",
        ),
        (
            "MYNAME DUNIF 0 2 1",
            (
                "Minimum 2.0 must be strictly less than the maximum"
                " 1.0 for duniform distribution"
            ),
        ),
        ("MYNAME ERRF 0 1 2 3", None),
        (
            "MYNAME ERRF 3 2 1 1",
            (
                "Minimum 3.0 must be strictly less than the maximum"
                " 2.0 for errf distribution"
            ),
        ),
        (
            "MYNAME ERRF 3 2 1 -1",
            "The width -1.0 must be greater than 0 for errf distribution",
        ),
        ("MYNAME DERRF 3 1 2 3 4", None),
        ("MYNAME LOGUNIF 0 1", None),
        ("MYNAME CONST 0", None),
        ("MYNAME RAW", None),
        (
            "MYNAME UNIFORM 0 1 2",
            (
                "Incorrect number of values: \\['0', '1', '2'\\], "
                "provided for variable MYNAME with distribution UNIFORM."
            ),
        ),
        (
            "MYNAME RANDOM 0 1",
            "Unknown distribution provided: RANDOM, for variable MYNAME",
        ),
        ("MYNAME DERRF 50 1.12345 2.3 3.14 10E-5", None),
        ("MYNAME DERRF 100 -14 -2.544545 10E5 10E+5", None),
        (
            "MYNAME CONST no-number",
            (
                "Unable to convert 'no-number' to float number for "
                "variable MYNAME with distribution CONST."
            ),
        ),
        ("MYNAME      CONST    0", None),  # spaces
        ("MYNAME\t\t\tCONST\t\t0", None),  # tabs
    ],
)
def test_gen_kw_params_parsing(tmpdir, params, error):
    with tmpdir.as_cwd():
        parts = params.split()
        name, dist_name, values = parts[0], parts[1], parts[2:]

        if error:
            with pytest.raises(ConfigValidationError, match=error):
                GenKwConfig._parse_distribution(name, dist_name, values)
        else:
            dist = GenKwConfig._parse_distribution(name, dist_name, values)
            GenKwConfig(
                name=name,
                forward_init=False,
                update=False,
                distribution=dist,
            )


@pytest.mark.parametrize(
    ("params", "xinput", "expected"),
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
        ("MYNAME LOGNORMAL 0 1", 1.0, math.e),
        ("MYNAME ERRF 1 2 0.1 0.1", -1.0, 1.00000000000000000000),
        ("MYNAME ERRF 1 2 0.1 0.1", 0.0, 1.84134474606854281475),
        ("MYNAME ERRF 1 2 0.1 0.1", 0.3, 1.99996832875816688002),
        ("MYNAME ERRF 1 2 0.1 0.1", 0.7, 1.99999999999999933387),
        ("MYNAME ERRF 1 2 0.1 0.1", 1.0, 2.00000000000000000000),
    ],
)
def test_gen_kw_trans_func(tmpdir, params, xinput, expected):
    name, dist_name, *values = params.split()
    with tmpdir.as_cwd():
        cfg = GenKwConfig(
            name=name,
            forward_init=False,
            update=False,
            distribution=GenKwConfig._parse_distribution(name, dist_name, values),
        )
        assert abs(cfg.distribution.transform(xinput) - expected) < 10**-15


def test_gen_kw_objects_equal(tmpdir):
    with tmpdir.as_cwd():
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")

        g1 = GenKwConfig.from_config_list(
            [
                "KW_NAME",
                ("prior.txt", "MY_KEYWORD UNIFORM 1 2"),
                {},
            ]
        )[0]
        assert g1.name == "MY_KEYWORD"
        assert g1.group == "KW_NAME"

        g2 = GenKwConfig(
            name="MY_KEYWORD",
            group="KW_NAME",
            distribution={"name": "uniform", "min": 1, "max": 2},
        )

        assert g1.name == g2.name
        assert g1.group == g2.group
        assert g1.distribution == g2.distribution


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


def make_context_string(msg: str, filename: str) -> FileContextToken:
    return FileContextToken(Token("UNQUOTED", msg), filename)


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_config_validation():
    Path("template.txt").write_text("Hello", encoding="utf-8")

    GenKwConfig.templates_from_config(
        [
            "KEY",
            ("template.txt", "Hello"),
            "nothing_here.txt",
            ("parameters.txt", "KEY  UNIFORM 0 1 \n"),
            {},
        ]
    )

    GenKwConfig.templates_from_config(
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
        GenKwConfig.templates_from_config(
            [
                "KEY",
                make_context_string("no_template_here.txt", "config.ert"),
                "nothing_here.txt",
                "parameters.txt",
                {},
            ]
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_suggestion_on_empty_parameter_file():
    Path("empty_template.txt").write_text("", encoding="utf-8")
    with pytest.warns(UserWarning, match="GEN_KW KEY coeffs.txt"):
        GenKwConfig.templates_from_config(
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
    ("distribution", "minimum", "maximum", "error"),
    [
        ("UNIFORM", "0", "1", None),
        (
            "UNIFORM",
            "1.0",
            "1.0",
            "Minimum 1.0 must be strictly less than the maximum 1.0",
        ),
        (
            "LOGUNIF",
            "1.0",
            "1.0",
            "Minimum 1.0 must be strictly less than the maximum 1.0",
        ),
    ],
)
def test_validation_unif_distribution(tmpdir, distribution, minimum, maximum, error):
    with tmpdir.as_cwd():
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        config_list = [
            "KW_NAME",
            ("template.txt", "MY_KEYWORD <MY_KEYWORD>"),
            "kw.txt",
            ("prior.txt", f"MY_KEYWORD {distribution} {minimum} {maximum}"),
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
    ("distribution", "minimum", "mode", "maximum", "error"),
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
    tmpdir, distribution, minimum, mode, maximum, error
):
    with tmpdir.as_cwd():
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        config_list = [
            "KW_NAME",
            ("template.txt", "MY_KEYWORD <MY_KEYWORD>"),
            "kw.txt",
            ("prior.txt", f"MY_KEYWORD {distribution} {minimum} {mode} {maximum}"),
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
    ("distribution", "nbins", "minimum", "maximum", "skew", "width", "error"),
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
            (
                "NBINS 0.0 must be a positive integer larger than 1 for "
                "DERRF distributed parameter MY_KEYWORD"
            ),
        ),
        (
            "DERRF",
            "-5",
            "-1",
            "3",
            "-1",
            "2",
            (
                "NBINS -5.0 must be a positive integer larger than 1 for "
                "DERRF distributed parameter MY_KEYWORD"
            ),
        ),
        (
            "DERRF",
            "1.5",
            "-1",
            "3",
            "-1",
            "2",
            (
                "NBINS 1.5 must be a positive integer larger than 1 for "
                "DERRF distributed parameter MY_KEYWORD"
            ),
        ),
        (
            "DERRF",
            "10",
            "3",
            "-1",
            "-1",
            "2",
            (
                "The minimum 3.0 must be less than the maximum -1.0 for "
                "DERRF distributed parameter MY_KEYWORD"
            ),
        ),
        (
            "DERRF",
            "10",
            "1",
            "1",
            "-1",
            "2",
            (
                "The minimum 1.0 must be less than the maximum 1.0 for "
                "DERRF distributed parameter MY_KEYWORD"
            ),
        ),
        (
            "DERRF",
            "10",
            "-1",
            "3",
            "-1",
            "0",
            (
                "The width 0.0 must be greater than 0 for "
                "DERRF distributed parameter MY_KEYWORD"
            ),
        ),
        (
            "DERRF",
            "10",
            "-1",
            "3",
            "-1",
            "-2",
            (
                "The width -2.0 must be greater than 0 for "
                "DERRF distributed parameter MY_KEYWORD"
            ),
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
    tmpdir, distribution, nbins, minimum, maximum, skew, width, error
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
                f"MY_KEYWORD {distribution} {nbins} {minimum} {maximum} {skew} {width}",
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


def test_genkw_paramgraph_transformfn_node_correspondence():
    config = GenKwConfig(
        name="param",
        group="COEFFS",
        distribution={"name": "uniform", "min": 1, "max": 2},
    )

    graph = config.load_parameter_graph()

    data = nx.node_link_data(graph, edges="edges")
    assert data["edges"] == []

    assert data["nodes"] == [{"id": 0}]


def test_genkw_raises_config_validation_error_from_pydantic_validation_error_given_invalid_model_input(  # noqa E501
    monkeypatch,
):
    # Bool is not a valid input to GenKwConfig's "distribution" attribute
    monkeypatch.setattr(GenKwConfig, "_parse_distribution", lambda *args: False)

    config_list = [
        "KW_NAME",
        ("template.txt", "MY_KEYWORD <MY_KEYWORD>"),
        "kw.txt",
        ("prior.txt", "MY_KEYWORD DERFF 10 20"),
        {},
    ]
    with pytest.raises(ConfigValidationError):
        GenKwConfig.from_config_list(config_list)


def test_that_const_keyword_sets_update_to_false(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GEN_KW CONST_TEST prior.txt UPDATE:TRUE
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("prior.txt", "w", encoding="utf-8") as fh:
            fh.writelines("CONST_TEST CONST 1")

        ert_config = ErtConfig.from_file("config.ert")

        gen_kw_config = ert_config.ensemble_config.parameter_configs["CONST_TEST"]
        assert gen_kw_config.update is False
