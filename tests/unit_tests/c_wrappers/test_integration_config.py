import json
import os
from argparse import ArgumentParser
from textwrap import dedent

import pytest

from ert.__main__ import ert_parser
from ert.cli import TEST_RUN_MODE
from ert.cli.main import run_cli
from ert.config import ErtConfig
from ert.enkf_main import EnKFMain
from ert.run_context import RunContext
from ert.storage import StorageAccessor


def _create_runpath(enkf_main: EnKFMain, storage: StorageAccessor) -> RunContext:
    """
    Instantiate an ERT runpath. This will create the parameter coefficients.
    """
    run_context = enkf_main.ensemble_context(
        storage.create_ensemble(
            storage.create_experiment(),
            name="prior",
            ensemble_size=enkf_main.getEnsembleSize(),
        ),
        [True] * enkf_main.getEnsembleSize(),
    )
    enkf_main.createRunPath(run_context)
    return run_context


@pytest.mark.parametrize(
    "append,numcpu",
    [
        ("", 1),  # Default is 1
        ("NUM_CPU 2\n", 2),
        ("DATA_FILE DATA\n", 8),  # Data file dictates NUM_CPU with PARALLEL
        ("NUM_CPU 3\nDATA_FILE DATA\n", 3),  # Explicit NUM_CPU supersedes PARALLEL
    ],
)
def test_num_cpu_subst(monkeypatch, tmp_path, append, numcpu, storage):
    """
    Make sure that <NUM_CPU> is substituted to the correct values
    """
    monkeypatch.chdir(tmp_path)

    (tmp_path / "test.ert").write_text(
        "JOBNAME test_%d\n"
        "NUM_REALIZATIONS 1\n"
        "INSTALL_JOB dump DUMP\n"
        "FORWARD_MODEL dump\n" + append
    )
    (tmp_path / "DATA").write_text("PARALLEL 8 /")
    (tmp_path / "DUMP").write_text("EXECUTABLE echo\nARGLIST <NUM_CPU>\n")

    config = ErtConfig.from_file(str(tmp_path / "test.ert"))
    enkf_main = EnKFMain(config)
    _create_runpath(enkf_main, storage)

    with open("simulations/realization-0/iter-0/jobs.json", encoding="utf-8") as f:
        assert f'"argList": ["{numcpu}"]' in f.read()


@pytest.fixture
def setenv_config(tmp_path):
    config = tmp_path / "test.ert"

    # Given that environment variables are set in the config
    config.write_text(
        """
        NUM_REALIZATIONS 1
        SETENV FIRST first:$PATH
        SETENV SECOND $MYVAR
        SETENV MYVAR foo
        SETENV THIRD  TheThirdValue
        SETENV FOURTH fourth:$MYVAR
        INSTALL_JOB ECHO ECHO.txt
        FORWARD_MODEL ECHO
        """,
        encoding="utf-8",
    )
    run_script = tmp_path / "run.py"
    run_script.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        'print(os.environ["FIRST"])\n'
        'print(os.environ["SECOND"])\n'
        'print(os.environ["THIRD"])\n'
        'print(os.environ["FOURTH"])\n',
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    (tmp_path / "ECHO.txt").write_text(
        dedent(
            """
        EXECUTABLE run.py
        """
        )
    )
    return config


expected_vars = {
    "FIRST": "first:$PATH",
    "SECOND": "$MYVAR",
    "MYVAR": "foo",
    "THIRD": "TheThirdValue",
    "FOURTH": "fourth:$MYVAR",
}


def test_that_setenv_config_is_parsed_correctly(setenv_config):
    config = ErtConfig.from_file(str(setenv_config))
    # then res config should read the SETENV as is
    assert config.env_vars == expected_vars


def test_that_setenv_sets_environment_variables_in_jobs(setenv_config):
    # When running the jobs
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            TEST_RUN_MODE,
            str(setenv_config),
            "--port-range",
            "1024-65535",
        ],
    )

    run_cli(parsed)

    # Then the environment variables are put into jobs.json
    with open("simulations/realization-0/iter-0/jobs.json", encoding="utf-8") as f:
        data = json.load(f)
        global_env = data.get("global_environment")
        assert global_env == expected_vars

    path = os.environ["PATH"]

    # and then job_dispatch should expand the variables on the compute side
    with open("simulations/realization-0/iter-0/ECHO.stdout.0", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 4
        # the compute-nodes path is the same since it's running locally,
        # so we can test that we can prepend to it
        assert lines[0].strip() == f"first:{path}"
        # MYVAR is not set in the compyte node yet, so it should not be expanded
        assert lines[1].strip() == "$MYVAR"
        # THIRD is just a simple value
        assert lines[2].strip() == "TheThirdValue"
        # now MYVAR now set, so should be expanded inside the value of FOURTH
        assert lines[3].strip() == "fourth:foo"


def replace_string_in_file(file: str, from_string: str, to_string: str):
    with open(file, encoding="utf-8") as f:
        content = f.read()

    with open(file, mode="w", encoding="utf-8") as f:
        f.write(content.replace(from_string, to_string))
