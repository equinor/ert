import datetime
import json

import polars as pl
import pytest

from ert.storage.migration.to22 import (
    DictEncodedDataFrame,
    migrate,
)

snake_oil_responses = {
    "summary": {
        "type": "summary",
        "name": "summary",
        "input_files": ["SNAKE_OIL_FIELD"],
        "keys": [
            "BPR:1,3,8",
            "BPR:5,5,5",
            "FGIP",
            "FGIPH",
            "FGOR",
            "FGORH",
            "FGPR",
            "FGPRH",
            "FGPT",
            "FGPTH",
            "FOIP",
            "FOIPH",
            "FOPR",
            "FOPRH",
            "FOPT",
            "FOPTH",
            "FWCT",
            "FWCTH",
            "FWIP",
            "FWIPH",
            "FWPR",
            "FWPRH",
            "FWPT",
            "FWPTH",
            "TIME",
            "WGOR:OP1",
            "WGOR:OP2",
            "WGORH:OP1",
            "WGORH:OP2",
            "WGPR:OP1",
            "WGPR:OP2",
            "WGPRH:OP1",
            "WGPRH:OP2",
            "WOPR:OP1",
            "WOPR:OP2",
            "WOPRH:OP1",
            "WOPRH:OP2",
            "WWCT:OP1",
            "WWCT:OP2",
            "WWCTH:OP1",
            "WWCTH:OP2",
            "WWPR:OP1",
            "WWPR:OP2",
            "WWPRH:OP1",
            "WWPRH:OP2",
        ],
        "has_finalized_keys": True,
    },
    "gen_data": {
        "type": "gen_data",
        "name": "gen_data",
        "input_files": [
            "snake_oil_opr_diff_%d.txt",
            "snake_oil_wpr_diff_%d.txt",
            "snake_oil_gpr_diff_%d.txt",
        ],
        "keys": ["SNAKE_OIL_OPR_DIFF", "SNAKE_OIL_WPR_DIFF", "SNAKE_OIL_GPR_DIFF"],
        "has_finalized_keys": True,
        "report_steps_list": [[199], [199], [199]],
    },
}

snake_oil_parameters = {
    "OP1_PERSISTENCE": {
        "type": "gen_kw",
        "name": "OP1_PERSISTENCE",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": 0.01, "max": 0.4},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
    "OP1_OCTAVES": {
        "type": "gen_kw",
        "name": "OP1_OCTAVES",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": 3.0, "max": 5.0},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
    "OP1_DIVERGENCE_SCALE": {
        "type": "gen_kw",
        "name": "OP1_DIVERGENCE_SCALE",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": 0.25, "max": 1.25},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
    "OP1_OFFSET": {
        "type": "gen_kw",
        "name": "OP1_OFFSET",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": -0.1, "max": 0.1},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
    "OP2_PERSISTENCE": {
        "type": "gen_kw",
        "name": "OP2_PERSISTENCE",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": 0.1, "max": 0.6},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
    "OP2_OCTAVES": {
        "type": "gen_kw",
        "name": "OP2_OCTAVES",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": 5.0, "max": 12.0},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
    "OP2_DIVERGENCE_SCALE": {
        "type": "gen_kw",
        "name": "OP2_DIVERGENCE_SCALE",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": 0.5, "max": 1.5},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
    "OP2_OFFSET": {
        "type": "gen_kw",
        "name": "OP2_OFFSET",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": -0.2, "max": 0.2},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
    "BPR_555_PERSISTENCE": {
        "type": "gen_kw",
        "name": "BPR_555_PERSISTENCE",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": 0.1, "max": 0.5},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
    "BPR_138_PERSISTENCE": {
        "type": "gen_kw",
        "name": "BPR_138_PERSISTENCE",
        "forward_init": False,
        "update": True,
        "distribution": {"name": "uniform", "min": 0.2, "max": 0.7},
        "group": "SNAKE_OIL_PARAM",
        "input_source": "sampled",
    },
}

snake_oil_template_1 = """OP1_PERSISTENCE:<OP1_PERSISTENCE>
OP1_OCTAVES:<OP1_OCTAVES>
OP1_DIVERGENCE_SCALE:<OP1_DIVERGENCE_SCALE>
OP1_OFFSET:<OP1_OFFSET>
OP2_PERSISTENCE:<OP2_PERSISTENCE>
OP2_OCTAVES:<OP2_OCTAVES>
OP2_DIVERGENCE_SCALE:<OP2_DIVERGENCE_SCALE>
OP2_OFFSET:<OP2_OFFSET>
BPR_555_PERSISTENCE:<BPR_555_PERSISTENCE>
BPR_138_PERSISTENCE:<BPR_138_PERSISTENCE>
"""

gendata_observations = pl.from_dicts(
    [
        {
            "response_key": "SNAKE_OIL_WPR_DIFF",
            "observation_key": "WPR_DIFF_1",
            "report_step": 199,
            "index": 400,
            "observations": 0.0,
            "std": 0.10000000149011612,
        },
        {
            "response_key": "SNAKE_OIL_WPR_DIFF",
            "observation_key": "WPR_DIFF_1",
            "report_step": 199,
            "index": 800,
            "observations": 0.10000000149011612,
            "std": 0.20000000298023224,
        },
        {
            "response_key": "SNAKE_OIL_WPR_DIFF",
            "observation_key": "WPR_DIFF_1",
            "report_step": 199,
            "index": 1200,
            "observations": 0.20000000298023224,
            "std": 0.15000000596046448,
        },
        {
            "response_key": "SNAKE_OIL_WPR_DIFF",
            "observation_key": "WPR_DIFF_1",
            "report_step": 199,
            "index": 1800,
            "observations": 0.0,
            "std": 0.05000000074505806,
        },
    ],
    schema={
        "response_key": pl.String,
        "observation_key": pl.String,
        "report_step": pl.UInt16,
        "index": pl.UInt16,
        "observations": pl.Float32,
        "std": pl.Float32,
    },
)

summary_observations = pl.from_dicts(
    (
        [
            {
                "response_key": "FOPR",
                "observation_key": "FOPR",
                "time": datetime.datetime(2010, 1, 10)
                + datetime.timedelta(days=10 * i),
                "observations": (
                    ((i if i < 60 else (120 - i if i < 120 else 0)) / 60.0)
                    + ((i % 7) * 0.001)
                ),
                "std": 0.10 + ((i % 10) * 0.001),
            }
            for i in range(200)
        ]
        + [
            {
                "response_key": "WOPR:OP1",
                "observation_key": f"WOPR_OP1_{k}",
                "time": datetime.datetime(2010, 1, 10)
                + datetime.timedelta(days=10 * k),
                "observations": 0.1 + (k % 9) * 0.05,  # just some jittery steps
                "std": 0.05 + (k % 4) * 0.01,
            }
            for k in range(36, 200, 36)
        ]
    ),
    schema={
        "response_key": pl.String,
        "observation_key": pl.String,
        "time": pl.Datetime(time_unit="ms", time_zone=None),
        "observations": pl.Float32,
        "std": pl.Float32,
    },
)


@pytest.fixture
def setup_snake_oil_experiment(tmp_path):
    """Sets up a temporary directory structure for migration tests."""
    # tmp_path is a built-in pytest fixture that provides a temporary directory
    root_dir = tmp_path / "project"
    root_dir.mkdir()

    experiment_path = root_dir / "experiments" / "exp1"
    experiment_path.mkdir(parents=True)

    observations_path = experiment_path / "observations"
    observations_path.mkdir()

    # Create dummy files
    (experiment_path / "metadata.json").write_text(json.dumps({"weights": [0.2, 0.8]}))
    (experiment_path / "index.json").write_text(
        json.dumps(
            {
                "id": "4228ae19-7941-4d85-88f3-a4833725a920",
                "name": "ensemble-experiment",
                "ensembles": [],
            }
        )
    )
    (experiment_path / "responses.json").write_text(json.dumps(snake_oil_responses))
    (experiment_path / "parameter.json").write_text(json.dumps(snake_oil_parameters))
    (experiment_path / "templates.json").write_text(
        json.dumps(
            [
                ["templates/seed_template_0.txt", "seed.txt"],
                ["templates/snake_oil_template_1.txt", "snake_oil_params.txt"],
            ]
        )
    )
    (experiment_path / "templates").mkdir()
    (experiment_path / "templates" / "snake_oil_template_1.txt").write_text(
        snake_oil_template_1
    )
    (experiment_path / "templates" / "seed_template_0.txt").write_text("SEED:<IENS>\n")
    summary_observations.write_parquet(observations_path / "summary")
    gendata_observations.write_parquet(observations_path / "gen_data")

    # Yield the root directory to the test
    return root_dir

    # Teardown is handled automatically by the tmp_path fixture


def test_that_storage_migration_to_21_migrates_successfully(setup_snake_oil_experiment):
    """
    Tests that the migration correctly moves files and integrates
    data into index.json.
    """
    # Run the migration on the temporary environment
    migrate(setup_snake_oil_experiment)

    experiment_path = setup_snake_oil_experiment / "experiments" / "exp1"

    # 1. Check that source files and directories are deleted
    assert not (experiment_path / "metadata.json").exists()
    assert not (experiment_path / "responses.json").exists()
    assert not (experiment_path / "parameter.json").exists()
    assert not (experiment_path / "observations").exists()

    # 2. Check that index.json is updated correctly
    index_path = experiment_path / "index.json"
    assert index_path.exists()

    with open(index_path, encoding="utf-8") as f:
        updated_index = json.load(f)

    # Check that original data in index.json is preserved
    assert updated_index["id"] == "4228ae19-7941-4d85-88f3-a4833725a920"

    # 3. Check that data is migrated into the 'experiment' field
    assert "experiment" in updated_index
    experiment_data = updated_index["experiment"]

    parameters_dict = {v["name"]: v for v in experiment_data["parameter_configuration"]}
    assert parameters_dict == snake_oil_parameters

    responses_dict = {v["name"]: v for v in experiment_data["response_configuration"]}
    assert responses_dict == snake_oil_responses

    gen_obs = DictEncodedDataFrame.model_validate(
        experiment_data["observations"]["gen_data"]
    )
    summary_obs = DictEncodedDataFrame.model_validate(
        experiment_data["observations"]["summary"]
    )

    assert gen_obs.to_polars().equals(gendata_observations)
    assert summary_obs.to_polars().equals(summary_observations)
