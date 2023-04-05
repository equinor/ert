import re
from pathlib import Path
from textwrap import dedent

import pytest

from ert.parsing import ConfigValidationError
from ert._c_wrappers.enkf import EnKFMain, ErtConfig, GenKwConfig


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_config():
    with open("template.txt", "w", encoding="utf-8") as f:
        f.write("Hello")

    with open("parameters.txt", "w", encoding="utf-8") as f:
        f.write("KEY  UNIFORM 0 1 \n")

    with open("parameters_with_comments.txt", "w", encoding="utf-8") as f:
        f.write("KEY1  UNIFORM 0 1 -- COMMENT\n")
        f.write("\n\n")  # Two blank lines
        f.write("KEY2  UNIFORM 0 1\n")
        f.write("--KEY3  \n")
        f.write("KEY3  UNIFORM 0 1\n")

    template_file = "template.txt"
    parameter_file = "parameters.txt"
    parameter_file_comments = "parameters_with_comments.txt"
    with pytest.raises(IOError):
        conf = GenKwConfig("KEY", template_file, "does_not_exist")

    with pytest.raises(IOError):
        conf = GenKwConfig("Key", "does_not_exist", parameter_file)

    conf = GenKwConfig("KEY", template_file, parameter_file)
    conf = GenKwConfig("KEY", template_file, parameter_file_comments)
    assert len(conf) == 3


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

    conf = GenKwConfig("KEY", template_file, parameter_file)
    priors = conf.get_priors()
    assert len(conf) == 10

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

        node = ert.ensembleConfig().getNode("KW_NAME")
        gen_kw_config = node.getModelConfig()
        assert isinstance(gen_kw_config, GenKwConfig)
        assert gen_kw_config.shouldUseLogScale(0) is expect_log
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
        )


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
