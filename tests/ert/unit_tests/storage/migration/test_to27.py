import json

import pytest

from ert.storage.migration.to27 import migrate


def _make_index(observations):
    return {
        "id": "exp-id",
        "experiment": {"observations": observations},
    }


def _write_experiment(root, observations):
    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)
    index_file = exp_path / "index.json"
    index_file.write_text(json.dumps(_make_index(observations)), encoding="utf-8")
    return index_file


def _read_experiment(index_file):
    return json.loads(index_file.read_text(encoding="utf-8"))["experiment"]


def test_that_summary_observation_localization_is_migrated_to_shape_registry(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    index_file = _write_experiment(
        root,
        [
            {
                "type": "summary_observation",
                "name": "OBS1",
                "value": 1.0,
                "error": 0.1,
                "key": "FOPR",
                "date": "2020-01-01",
                "east": 100.0,
                "north": 200.0,
                "radius": 3000.0,
            },
        ],
    )

    migrate(root)

    experiment = _read_experiment(index_file)
    obs = experiment["observations"][0]
    assert "east" not in obs
    assert "north" not in obs
    assert "radius" not in obs
    assert obs["shape_id"] == 0

    registry = experiment["shape_registry"]["shapes"]
    assert registry["0"]["east"] == pytest.approx(100.0)
    assert registry["0"]["north"] == pytest.approx(200.0)
    assert registry["0"]["radius"] == pytest.approx(3000.0)
    assert registry["0"]["type"] == "circle"


def test_that_summary_observation_without_localization_gets_shape_id_none(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    original_obs = {
        "type": "summary_observation",
        "name": "OBS1",
        "value": 1.0,
        "error": 0.1,
        "key": "FOPR",
        "date": "2020-01-01",
    }
    index_file = _write_experiment(root, [original_obs.copy()])

    migrate(root)

    experiment = _read_experiment(index_file)
    obs = experiment["observations"][0]
    assert obs["shape_id"] is None
    assert experiment["shape_registry"] == {"shapes": {}}


def test_that_summary_observation_defaults_radius_to_2000_when_missing(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    index_file = _write_experiment(
        root,
        [
            {
                "type": "summary_observation",
                "name": "OBS1",
                "value": 1.0,
                "error": 0.1,
                "key": "FOPR",
                "date": "2020-01-01",
                "east": 100.0,
                "north": 200.0,
            },
        ],
    )

    migrate(root)

    experiment = _read_experiment(index_file)
    registry = experiment["shape_registry"]["shapes"]
    assert registry["0"]["radius"] == 2000


def test_that_rft_observation_radius_is_migrated_to_shape_registry(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    index_file = _write_experiment(
        root,
        [
            {
                "type": "rft_observation",
                "name": "RFT1",
                "well": "W1",
                "date": "2020-01-01",
                "property": "PRESSURE",
                "value": 300.0,
                "error": 10.0,
                "east": 50.0,
                "north": 60.0,
                "tvd": 1500.0,
                "radius": 2400.0,
            },
        ],
    )

    migrate(root)

    experiment = _read_experiment(index_file)
    obs = experiment["observations"][0]
    assert obs["east"] == pytest.approx(50.0)
    assert obs["north"] == pytest.approx(60.0)
    assert "radius" not in obs
    assert obs["shape_id"] == 0

    registry = experiment["shape_registry"]["shapes"]
    assert registry["0"]["east"] == pytest.approx(50.0)
    assert registry["0"]["north"] == pytest.approx(60.0)
    assert registry["0"]["radius"] == pytest.approx(2400.0)


def test_that_identical_shapes_share_shape_id(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    index_file = _write_experiment(
        root,
        [
            {
                "type": "summary_observation",
                "name": "OBS1",
                "value": 1.0,
                "error": 0.1,
                "key": "FOPR",
                "date": "2020-01-01",
                "east": 100.0,
                "north": 200.0,
                "radius": 3000.0,
            },
            {
                "type": "summary_observation",
                "name": "OBS2",
                "value": 2.0,
                "error": 0.2,
                "key": "FWPR",
                "date": "2020-06-01",
                "east": 100.0,
                "north": 200.0,
                "radius": 3000.0,
            },
            {
                "type": "summary_observation",
                "name": "OBS3",
                "value": 3.0,
                "error": 0.3,
                "key": "FGPR",
                "date": "2020-12-01",
                "east": 500.0,
                "north": 600.0,
                "radius": 1000.0,
            },
        ],
    )

    migrate(root)

    experiment = _read_experiment(index_file)
    obs_list = experiment["observations"]
    assert obs_list[0]["shape_id"] == obs_list[1]["shape_id"]
    assert obs_list[0]["shape_id"] != obs_list[2]["shape_id"]

    registry = experiment["shape_registry"]["shapes"]
    assert len(registry) == 2


def test_that_mixed_observation_types_are_migrated(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    index_file = _write_experiment(
        root,
        [
            {
                "type": "summary_observation",
                "name": "OBS1",
                "value": 1.0,
                "error": 0.1,
                "key": "FOPR",
                "date": "2020-01-01",
                "east": 100.0,
                "north": 200.0,
                "radius": 3000.0,
            },
            {
                "type": "rft_observation",
                "name": "RFT1",
                "well": "W1",
                "date": "2020-01-01",
                "property": "PRESSURE",
                "value": 300.0,
                "error": 10.0,
                "east": 100.0,
                "north": 200.0,
                "tvd": 1500.0,
                "radius": 3000.0,
            },
            {
                "type": "general_observation",
                "name": "GEN1",
                "data": "RESPONSE",
                "value": 5.0,
                "error": 0.5,
                "restart": 0,
                "index": 0,
            },
            {
                "type": "breakthrough",
                "name": "BT1",
                "date": "2020-01-01",
                "value": 300.0,
                "error": 10.0,
                "east": 100.0,
                "north": 200.0,
                "radius": 3000.0,
                "threshold": 0.2,
            },
        ],
    )

    migrate(root)

    experiment = _read_experiment(index_file)
    obs_list = experiment["observations"]

    # Summary: east/north/radius removed, shape_id set
    assert "east" not in obs_list[0]
    assert obs_list[0]["shape_id"] == 0

    # RFT: east/north kept, radius removed, shape_id set
    assert obs_list[1]["east"] == pytest.approx(100.0)
    assert obs_list[1]["north"] == pytest.approx(200.0)
    assert "radius" not in obs_list[1]
    assert obs_list[1]["shape_id"] == 0

    # General: shape_id set to None
    assert obs_list[2]["shape_id"] is None

    # Breakthrough: east/north/radius removed, shape_id set
    assert "east" not in obs_list[3]
    assert obs_list[3]["shape_id"] == 0

    registry = experiment["shape_registry"]["shapes"]
    assert len(registry) == 1


def test_that_experiment_without_observations_has_empty_shape_registry(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)
    index_data = {"id": "exp-id", "experiment": {}}
    index_file = exp_path / "index.json"
    index_file.write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    experiment = json.loads(index_file.read_text(encoding="utf-8"))["experiment"]
    assert experiment["shape_registry"] == {"shapes": {}}


def test_that_migration_converts_bool_update_to_strategy_string(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    params = [
        {
            "type": "gen_kw",
            "name": "PARAM1",
            "update": True,
            "forward_init": False,
            "distribution": {"name": "uniform"},
        },
        {
            "type": "gen_kw",
            "name": "PARAM2",
            "update": False,
            "forward_init": False,
            "distribution": {"name": "normal"},
        },
        {
            "type": "field",
            "name": "PORO",
            "update": True,
            "forward_init": False,
        },
        {
            "type": "everest_parameters",
            "name": "ctrl",
            "update": False,
            "forward_init": False,
        },
    ]

    index_data = {
        "id": "exp-id",
        "experiment": {"parameter_configuration": params},
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    updated_params = updated["experiment"]["parameter_configuration"]

    assert updated_params[0]["update"] == "ADAPTIVE"
    assert updated_params[1]["update"] is None
    assert updated_params[2]["update"] == "ADAPTIVE"
    assert updated_params[3]["update"] is None


def test_that_migration_leaves_string_update_values_unchanged(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    params = [
        {
            "type": "gen_kw",
            "name": "PARAM1",
            "update": "ADAPTIVE",
            "forward_init": False,
        },
        {
            "type": "gen_kw",
            "name": "PARAM2",
            "update": None,
            "forward_init": False,
        },
        {
            "type": "gen_kw",
            "name": "PARAM3",
            "update": "DISTANCE",
            "forward_init": False,
        },
    ]

    index_data = {
        "id": "exp-id",
        "experiment": {"parameter_configuration": params},
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    updated_params = updated["experiment"]["parameter_configuration"]

    assert updated_params[0]["update"] == "ADAPTIVE"
    assert updated_params[1]["update"] is None
    assert updated_params[2]["update"] == "DISTANCE"


def test_that_migration_handles_missing_experiments_dir(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    # Should not raise
    migrate(root)
