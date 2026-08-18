import json

import hypothesis.strategies as st
from hypothesis import assume, given, settings

from ert.storage.migration.to39 import migrate

_RESTART_RUN_KEY = "restart_run"
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
        _RESTART_RUN_KEY: True,
    }

    migrated_experiment = migrate_and_load_updated_experiment(
        tmp_path, original_experiment_data
    )

    assert _RESTART_RUN_KEY not in migrated_experiment


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
        _RESTART_RUN_KEY: True,
    }

    migrated_experiment = migrate_and_load_updated_experiment(
        tmp_path, original_experiment_data
    )

    assert _RESTART_RUN_KEY in migrated_experiment
