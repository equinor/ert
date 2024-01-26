import io
import os
import shutil
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ert.gui.tools.plot.plot_api import PlotApi
from ert.services import StorageService


class MockResponse:
    def __init__(self, json_data, status_code, text="", url=""):
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
    from contextlib import contextmanager

    @contextmanager
    def session():
        yield MagicMock(get=mocked_requests_get)

    monkeypatch.setattr(StorageService, "session", session)

    with tmpdir.as_cwd():
        test_data_root = source_root / "test-data"
        test_data_dir = os.path.join(test_data_root, "snake_oil")
        shutil.copytree(test_data_dir, "test_data")
        os.chdir("test_data")
        api = PlotApi()
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

    gen_data = {
        "0": [0.1, 0.2, 0.3],
        "1": [0.1, 0.2, 0.3],
        "2": [0.1, 0.2, 0.3],
        "3": [0.1, 0.2, 0.3],
        "4": [0.1, 0.2, 0.3],
        "5": [0.1, 0.2, 0.3],
    }
    gen_df = pd.DataFrame(gen_data)
    gen_stream = io.BytesIO()
    gen_df.to_parquet(gen_stream)
    gen_parquet_data = gen_stream.getvalue()

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
        "/ensembles/ens_id_1": {"name": "ensemble_1"},
        "/ensembles/ens_id_2": {"name": ".ensemble_2"},
        "/ensembles/ens_id_3": {"name": "default_0"},
        "/ensembles/ens_id_4": {"name": "default_1"},
    }
    observations = {
        "/ensembles/ens_id_3/records/WOPR:OP1/observations": {
            "name": "WOPR:OP1",
            "errors": [0.05, 0.07],
            "values": [0.1, 0.7],
            "x_axis": ["2010-03-31T00:00:00", "2010-12-26T00:00:00"],
        },
        "/ensembles/ens_id_4/records/WOPR:OP1/observations": {
            "name": "WOPR:OP1",
            "errors": [0.05, 0.07],
            "values": [0.1, 0.7],
            "x_axis": ["2010-03-31T00:00:00", "2010-12-26T00:00:00"],
        },
        "/ensembles/ens_id_3/records/SNAKE_OIL_WPR_DIFF@199/observations": {
            "name": "SNAKE_OIL_WPR_DIFF",
            "errors": [0.05, 0.07, 0.05],
            "values": [0.1, 0.7, 0.5],
            "x_axis": [
                "2010-03-31T00:00:00",
                "2010-12-26T00:00:00",
                "2011-12-21T00:00:00",
            ],
        },
        "/ensembles/ens_id_4/records/SNAKE_OIL_WPR_DIFF@199/observations": {
            "name": "WOPR:OP1",
            "errors": [0.05, 0.07, 0.05],
            "values": [0.1, 0.7, 0.5],
            "x_axis": [
                "2010-03-31T00:00:00",
                "2010-12-26T00:00:00",
                "2011-12-21T00:00:00",
            ],
        },
        "/ensembles/ens_id_3/records/FOPR/observations": {
            "name": "FOPR",
            "errors": [0.05, 0.07],
            "values": [0.1, 0.7],
            "x_axis": ["2010-03-31T00:00:00", "2010-12-26T00:00:00"],
        },
    }

    parameters = {
        "/ensembles/ens_id_1/parameters": [
            {
                "name": "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
                "labels": [],
                "userdata": {"data_origin": "GEN_KW"},
            },
            {
                "name": "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
                "labels": [],
                "userdata": {"data_origin": "GEN_KW"},
            },
        ],
        "/ensembles/ens_id_3/parameters": [
            {
                "name": "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
                "labels": [],
                "userdata": {"data_origin": "GEN_KW"},
            },
            {
                "name": "I_AM_A_PARAM",
                "labels": [],
                "userdata": {"data_origin": "GEN_KW"},
            },
        ],
    }

    responses = {
        "/ensembles/ens_id_1/responses": {
            "BPR:1,3,8": {
                "name": "BPR:1,3,8",
                "id": "id_1",
                "userdata": {"data_origin": "Summary"},
                "has_observations": False,
            },
            "FOPR": {
                "name": "FOPR",
                "id": "id_999",
                "userdata": {"data_origin": "Summary"},
                "has_observations": True,
            },
            "SNAKE_OIL_WPR_DIFF@199": {
                "id": "id_88",
                "name": "SNAKE_OIL_WPR_DIFF@199",
                "userdata": {"data_origin": "GEN_DATA"},
                "has_observations": False,
            },
        },
        "/ensembles/ens_id_3/responses": {
            "BPR:1,3,8": {
                "name": "BPR:1,3,8",
                "id": "id_111111",
                "userdata": {"data_origin": "Summary"},
                "has_observations": False,
            },
            "WOPPER": {
                "name": "WOPPER",
                "id": "id_999",
                "userdata": {"data_origin": "Summary"},
                "has_observations": False,
            },
        },
    }

    ensembles = {
        "/experiments/exp_1/ensembles": [
            {"id": "ens_id_1", "userdata": {"name": "ensemble_1"}, "size": 25},
            {"id": "ens_id_3", "userdata": {"name": "default_0"}, "size": 99},
        ]
    }

    records = {
        "/ensembles/ens_id_3/records/FOPR": summary_parquet_data,
        "/ensembles/ens_id_3/records/BPR:1,3,8": summary_parquet_data,
        "/ensembles/ens_id_3/records/SNAKE_OIL_PARAM:BPR_138_PERSISTENCE": parameter_parquet_data,  # noqa
        "/ensembles/ens_id_3/records/SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE": parameter_parquet_data,  # noqa
        "/ensembles/ens_id_3/records/SNAKE_OIL_WPR_DIFF@199": gen_parquet_data,
        "/ensembles/ens_id_3/records/FOPRH": history_parquet_data,
    }

    experiments = [
        {
            "name": "default",
            "id": "exp_1",
            "ensemble_ids": ["ens_id_1", "ens_id_2", "ens_id_3", "ens_id_4"],
            "priors": {},
            "userdata": {},
        }
    ]

    if args[0] in ensemble:
        return MockResponse({"userdata": ensemble[args[0]]}, 200)
    elif args[0] in observations:
        return MockResponse(
            [observations[args[0]]],
            200,
        )
    elif args[0] in ensembles:
        return MockResponse(ensembles[args[0]], 200)
    elif args[0] in parameters:
        return MockResponse(parameters[args[0]], 200)
    elif args[0] in responses:
        return MockResponse(responses[args[0]], 200)
    elif args[0] in records:
        return MockResponse(records[args[0]], 200)
    elif "/experiments" in args[0]:
        return MockResponse(experiments, 200)

    return MockResponse(None, 404, text="{'details': 'Not found'}", url=args[0])
