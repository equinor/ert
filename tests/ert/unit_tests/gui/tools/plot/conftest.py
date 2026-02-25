import io
import os
import shutil
from contextlib import contextmanager
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ert.gui.tools.plot import plot_api
from ert.gui.tools.plot.plot_api import PlotApi


class MockResponse:
    def __init__(self, json_data, status_code, text="", url="") -> None:
        self.json_data = json_data
        self.status_code = status_code
        self.text = text
        self.url = url

    def json(self):
        return self.json_data

    @property
    def content(self):
        return self.json_data

    def is_success(self):
        return self.status_code == 200


@pytest.fixture
def api(tmpdir, source_root, monkeypatch):
    @contextmanager
    def session(project: str):
        yield MagicMock(get=mocked_requests_get)

    monkeypatch.setattr(plot_api, "create_ertserver_client", session)

    with tmpdir.as_cwd():
        test_data_root = source_root / "test-data" / "ert"
        test_data_dir = os.path.join(test_data_root, "snake_oil")
        shutil.copytree(test_data_dir, "test_data")
        os.chdir("test_data")
        api = PlotApi(test_data_dir)
        yield api


def mocked_requests_get(*args, **kwargs):
    summary_data = {
        "2010-01-20 00:00:00": [0.1, 0.2, 0.3, 0.4],
        "2010-02-20 00:00:00": [0.2, 0.21, 0.19, 0.18],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_stream = io.BytesIO()
    summary_df.to_parquet(summary_stream)
    summary_parquet_data = summary_stream.getvalue()

    parameter_data = {"0": [0.1, 0.2, 0.3]}
    parameter_df = pd.DataFrame(parameter_data)
    parameter_stream = io.BytesIO()
    parameter_df.to_parquet(parameter_stream)
    parameter_parquet_data = parameter_stream.getvalue()

    history_data = {
        "0": [1.0, 0.2, 1.0, 1.0, 1.0],
        "1": [1.1, 0.2, 1.1, 1.1, 1.1],
        "2": [1.2, 1.2, 1.2, 1.2, 1.3],
    }
    history_df = pd.DataFrame(history_data)
    history_stream = io.BytesIO()
    history_df.to_parquet(history_stream)
    history_parquet_data = history_stream.getvalue()

    ensemble = {
        "/ensembles/ens_id_1": {
            "userdata": {
                "name": "ensemble_1",
                "experiment_name": "experiment",
                "started_at": "2012-12-10T00:00:00",
            }
        },
        "/ensembles/ens_id_2": {
            "userdata": {
                "name": ".ensemble_2",
                "experiment_name": "experiment",
                "started_at": "2012-12-10T01:00:00",
            }
        },
        "/ensembles/ens_id_3": {
            "userdata": {
                "name": "default_0",
                "experiment_name": "experiment",
                "started_at": "2012-12-10T02:00:00",
            }
        },
        "/ensembles/ens_id_4": {
            "userdata": {
                "name": "default_1",
                "experiment_name": "experiment",
                "started_at": "2012-12-10T03:00:00",
            }
        },
        "/ensembles/ens_id_5": {
            "userdata": {
                "name": "default_manyobs",
                "experiment_name": "experiment",
                "started_at": "2012-12-10T04:00:00",
            }
        },
        "/ensembles/ens_id_uninitialized": {
            "userdata": {
                "name": "uninitialized_ensemble",
                "experiment_name": "experiment",
                "started_at": "2012-12-10T04:00:00",
            },
            "realization_storage_states": {"UNDEFINED": 1},
        },
    }

    observations = {
        "/ensembles/ens_id_3/responses/FOPR/observations": [
            {
                "name": "FOPR",
                "errors": [0.05, 0.07],
                "values": [0.1, 0.7],
                "x_axis": ["2010-03-31T00:00:00", "2010-12-26T00:00:00"],
            }
        ],
    }
    parameters = {
        "/ensembles/ens_id_1/parameters": [
            {
                "name": "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
                "dimensionality": 1,
                "labels": [],
                "userdata": {"data_origin": "GEN_KW"},
            },
            {
                "name": "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
                "dimensionality": 1,
                "labels": [],
                "userdata": {"data_origin": "GEN_KW"},
            },
        ],
        "/ensembles/ens_id_3/parameters": [
            {
                "name": "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
                "dimensionality": 1,
                "labels": [],
                "userdata": {"data_origin": "GEN_KW"},
            },
            {
                "name": "I_AM_A_PARAM",
                "dimensionality": 1,
                "labels": [],
                "userdata": {"data_origin": "GEN_KW"},
            },
        ],
        "/ensembles/ens_id_5/parameters": [
            {
                "name": "I_AM_A_PARAM",
                "dimensionality": 1,
                "labels": [],
                "userdata": {"data_origin": "GEN_KW"},
            },
        ],
    }

    records = {
        "/ensembles/ens_id_3/responses/FOPR": summary_parquet_data,
        "/ensembles/ens_id_3/responses/BPR%25253A1%25252C3%25252C8": summary_parquet_data,  # noqa
        (
            "/ensembles/ens_id_3/parameters/SNAKE_OIL_PARAM%25253ABPR_138_PERSISTENCE"
        ): parameter_parquet_data,
        "/ensembles/ens_id_3/responses/FOPRH": history_parquet_data,
    }

    experiments = [
        {
            "name": "default",
            "id": "exp_1",
            "ensemble_ids": [
                "ens_id_1",
                "ens_id_2",
                "ens_id_3",
                "ens_id_4",
                "ens_id_5",
                "ens_id_uninitialized",
            ],
            "parameters": {
                "BPR_138_PERSISTENCE": {
                    "type": "gen_kw",
                    "name": "BPR_138_PERSISTENCE",
                    "forward_init": False,
                    "update": True,
                    "dimensionality": 1,
                    "distribution": {"name": "uniform", "min": 0.2, "max": 0.7},
                    "group": "SNAKE_OIL_PARAM",
                    "input_source": "sampled",
                },
                "OP1_DIVERGENCE_SCALE": {
                    "type": "gen_kw",
                    "name": "OP1_DIVERGENCE_SCALE",
                    "forward_init": False,
                    "update": True,
                    "dimensionality": 1,
                    "distribution": {"name": "uniform", "min": 0.5, "max": 1.5},
                    "group": "SNAKE_OIL_PARAM",
                    "input_source": "sampled",
                },
                "I_AM_A_PARAM": {
                    "type": "gen_kw",
                    "name": "I_AM_A_PARAM",
                    "forward_init": False,
                    "update": True,
                    "dimensionality": 1,
                    "distribution": {"name": "normal", "mean": 0.0, "std": 1.0},
                    "group": "SNAKE_OIL_PARAM",
                    "input_source": "sampled",
                },
            },
            "responses": {
                "summary": {
                    "type": "summary",
                    "keys": ["BPR:1,3,8", "FOPR", "WOPPER"],
                    "has_finalized_keys": True,
                },
                "gen_data": {
                    "type": "gen_data",
                    "keys": ["SNAKE_OIL_WPR_DIFF"],
                    "report_steps_list": [
                        [199],
                    ],
                    "has_finalized_keys": True,
                },
            },
            "derived_responses": {},
            "observations": {
                "summary": ["FOPR"],
                "gen_data": ["SNAKE_OIL_WPR_DIFF@199"],
            },
            "priors": {},
            "userdata": {},
        }
    ]
    available_responses = (
        ensemble | observations | parameters | records | {"/experiments": experiments}
    )
    if available_response := available_responses.get(args[0]):
        return MockResponse(available_response, 200)
    return MockResponse(None, 404, text="{'details': 'Not found'}", url=args[0])
