import json

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings

from ert.storage.migration.to38 import migrate

_OLD_KEY = "restart_run"
_NEW_KEY = "select_prior"
_MDA_EXP_TYPE = "Multiple Data Assimilation"


def migrate_and_load_updated_experiment(tmp_path, original_experiment_data):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    index_data = {
        "id": "exp-id",
        "name": "exp1",
        "ensembles": [],
        "experiment": {
            **original_experiment_data,
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    return updated["experiment"]


def test_that_restart_run_key_not_present_in_migrated_experiment(tmp_path):

    original_experiment_data = {
        "experiment_type": _MDA_EXP_TYPE,
        _OLD_KEY: True,
    }

    migrated_experiment = migrate_and_load_updated_experiment(
        tmp_path, original_experiment_data
    )

    assert _OLD_KEY not in migrated_experiment


def test_that_select_prior_key_is_present_in_migrated_experiment(tmp_path):

    original_experiment_data = {
        "experiment_type": _MDA_EXP_TYPE,
        _OLD_KEY: True,
    }

    migrated_experiment = migrate_and_load_updated_experiment(
        tmp_path, original_experiment_data
    )

    assert _NEW_KEY in migrated_experiment


@pytest.mark.parametrize("original_value", [True, False])
def test_that_new_key_value_is_equal_to_old_key_value(tmp_path, original_value):

    original_experiment_data = {
        "experiment_type": _MDA_EXP_TYPE,
        _OLD_KEY: original_value,
    }

    migrated_experiment = migrate_and_load_updated_experiment(
        tmp_path, original_experiment_data
    )

    assert migrated_experiment[_NEW_KEY] == original_value


@given(non_mda_experiment_type=st.text())
@settings(max_examples=10)
def test_that_non_mda_experiment_type_is_not_migrated(
    tmp_path_factory, non_mda_experiment_type
):

    # to avoid hypothesis tmp_path fixture error
    tmp_path = tmp_path_factory.mktemp("arbitrary")

    assume(non_mda_experiment_type != _MDA_EXP_TYPE)

    original_experiment_data = {
        "experiment_type": non_mda_experiment_type,
        _OLD_KEY: True,
    }

    migrated_experiment = migrate_and_load_updated_experiment(
        tmp_path, original_experiment_data
    )

    assert _OLD_KEY in migrated_experiment
    assert _NEW_KEY not in migrated_experiment
