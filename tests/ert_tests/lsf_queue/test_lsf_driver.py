import os
import shutil
from argparse import ArgumentParser
from textwrap import dedent

import pytest

from ert_shared.cli import ENSEMBLE_EXPERIMENT_MODE
from ert_shared.cli.main import run_cli
from ert_shared.main import ert_parser


@pytest.fixture
def poly_case_context(tmpdir, source_root, mock_start_server):
    # Copy the poly_example files needed
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    # Add the mocked lsf commands to the runpath
    shutil.copytree(
        os.path.join(source_root, "tests", "ert_tests", "lsf_queue", "mocked_commands"),
        os.path.join(str(tmpdir), "poly_example", "mocked_commands"),
    )

    with tmpdir.as_cwd():
        yield


def test_run_mocked_lsf_queue(poly_case_context):
    apply_customized_config()
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_EXPERIMENT_MODE,
            "poly_example/poly.ert",
            "--port-range",
            "1024-65535",
        ],
    )

    run_cli(parsed)


def test_mock_bsub_fail_random(poly_case_context):
    """
    Approx 7/10 of the submits will fail due to the random generator in the
    created mocked bsub script. By using the retry functionality towards
    queue-errors in job_queue.cpp we should still manage to finalize all our runs
    before exhausting the limits
    """
    bsub_random_fail_name = "bsub_random_fail"
    introduce_bsub_failures(
        script_name=bsub_random_fail_name, fraction_successful_submits=0.3
    )

    apply_customized_config(mocked_bsub=bsub_random_fail_name)

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_EXPERIMENT_MODE,
            "poly_example/poly.ert",
            "--port-range",
            "1024-65535",
        ],
    )

    run_cli(parsed)


def apply_customized_config(
    max_running: int = 10,
    num_realizations: int = 10,
    min_realizations: int = 1,
    mocked_bsub: str = "mock_bsub",
):

    # Overwriting the "poly.ert" config file in tmpdir runpath
    # with our own customized config with at least sets queue option to LSF and
    # introducing the mocked jobs from "mocked_commands" folder.

    config = [
        "JOBNAME poly_%d\n",
        "QUEUE_SYSTEM  LSF\n",
        f"QUEUE_OPTION LSF  MAX_RUNNING {max_running}\n",
        "QUEUE_OPTION  LSF  BJOBS_CMD   mocked_commands/mock_bjobs\n",
        f"QUEUE_OPTION LSF  BSUB_CMD    mocked_commands/{mocked_bsub}\n",
        "RUNPATH poly_out/realization-%d/iter-%d\n",
        "OBS_CONFIG observations\n",
        "TIME_MAP time_map\n",
        f"NUM_REALIZATIONS {num_realizations}\n",
        f"MIN_REALIZATIONS {min_realizations}\n",
        "GEN_KW COEFFS coeff.tmpl coeffs.json coeff_priors\n",
        "GEN_DATA POLY_RES RESULT_FILE:poly_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII\n",
        "INSTALL_JOB poly_eval POLY_EVAL\n",
        "SIMULATION_JOB poly_eval\n",
    ]
    with open("poly_example/poly.ert", "w") as fh:
        fh.writelines(config)


def introduce_bsub_failures(script_name: str, fraction_successful_submits: float = 1.0):
    script = dedent(
        f"""#!/usr/bin/env python3
import random
import sys
import subprocess

num = random.random()
if num > {fraction_successful_submits}:
    exit(1)

job_dispatch_path = sys.argv[-2]
run_path = sys.argv[-1]

subprocess.call(["python", "mocked_commands/mock_bsub", job_dispatch_path, run_path])
        """
    )

    manipulated_script_path = f"poly_example/mocked_commands/{script_name}"
    with open(manipulated_script_path, "w") as fh:
        fh.write(script)

    os.chmod(manipulated_script_path, 0o755)
