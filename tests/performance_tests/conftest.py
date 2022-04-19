import os

import pytest
from performance_utils import make_poly_template, dark_storage_app
from argparse import ArgumentParser
from ert_shared.cli import ENSEMBLE_EXPERIMENT_MODE
from ert_shared.cli.main import run_cli
from ert_shared.main import ert_parser
from starlette.testclient import TestClient
import pandas as pd
import io
from requests import Response
from pytest import fixture
import py
import json
from pathlib import Path


def make_case(reals, x_size):
    return {
        "gen_data_count": 2,
        "gen_data_entries": x_size,
        "summary_data_entries": x_size,
        "reals": reals,
        "summary_data_count": 2,
        "sum_obs_count": 1,
        "gen_obs_count": 1,
        "sum_obs_every": 2,
        "gen_obs_every": 2,
        "parameter_entries": 3,
        "parameter_count": 1,
        "ministeps": 1,
    }


cases_to_run = [
    make_case(reals=10, x_size=20),
    make_case(reals=10, x_size=200),
    make_case(reals=10, x_size=2000),
    make_case(reals=100, x_size=20),
    make_case(reals=100, x_size=200),
    make_case(reals=100, x_size=2000),
    make_case(reals=1000, x_size=20000),
]


@fixture(
    scope="session",
    params=[
        pytest.param(
            params,
            marks=(
                pytest.mark.slow
                if params["reals"] > 10
                or params["gen_data_entries"] > 20
                or params["summary_data_entries"] > 20
                else []
            ),
        )
        for params in cases_to_run
    ],
    ids=[
        f"gen_x: {params['gen_data_entries']}, "
        f"sum_x: {params['summary_data_entries']} "
        f"reals: {params['reals']}"
        for params in cases_to_run
    ],
)
def poly_ran(request, source_root, tmp_path_factory):
    tmpdir = py.path.local(tmp_path_factory.mktemp("my_poly_tmp"))
    params = request.param
    params.update()

    poly_folder = make_poly_template(tmpdir, source_root, **params)
    params["folder"] = poly_folder

    with poly_folder.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        run_cli(parsed)

    yield params
