import hashlib
import json
from argparse import ArgumentParser

import pytest
from py import path
from pytest import fixture

from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_EXPERIMENT_MODE
from ert.cli.main import run_cli

from .performance_utils import make_poly_template

template_config_path = None


def make_case(reals, x_size, marks):
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
        "update_steps": 1,
        "marks": marks,
    }


cases_to_run = [
    make_case(reals=10, x_size=20, marks=pytest.mark.quick_only),
    make_case(reals=100, x_size=2000, marks=pytest.mark.slow),
    make_case(reals=1000, x_size=2000, marks=pytest.mark.slow),
]


@fixture(
    scope="session",
    params=[
        pytest.param(
            params,
            marks=params["marks"],
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
def template_config(request, source_root, tmp_path_factory):
    if template_config_path:
        tmpdir = path.local(template_config_path)
    else:
        tmpdir = path.local(tmp_path_factory.mktemp("my_poly_tmp"))

    params = request.param
    params.update()
    template_sorted = sorted([(k, v) for k, v in params.items() if k != "marks"])
    print(template_sorted)
    template_hash = hashlib.sha1(json.dumps(template_sorted).encode()).hexdigest()
    template_dir = tmpdir / template_hash

    if template_dir.isdir():
        poly_folder = template_dir / "poly"
        params["folder"] = poly_folder
        yield params
    else:
        poly_folder = make_poly_template(template_dir, source_root, **params)
        params["folder"] = poly_folder

        with poly_folder.as_cwd():
            parser = ArgumentParser(prog="test_main")
            parsed = ert_parser(
                parser,
                [
                    ENSEMBLE_EXPERIMENT_MODE,
                    "poly.ert",
                ],
            )
            run_cli(parsed)

        yield params


def pytest_configure(config):
    # pylint: disable=global-statement
    global template_config_path
    template_config_path = config.getoption("--template-config-path")


def pytest_addoption(parser):
    parser.addoption(
        "--template-config-path",
        default=None,
        help="specify to share previously generated template-config runs",
    )
