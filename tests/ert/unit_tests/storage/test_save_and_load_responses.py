import shutil
from datetime import datetime

import polars as pl
import pytest

from ert.config import (
    SummaryConfig,
)
from ert.storage import (
    open_storage,
)


def test_that_saving_response_updates_configs(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [
                    SummaryConfig(
                        keys=["*", "FOPR"], input_files=["not_relevant"]
                    ).model_dump(mode="json")
                ]
            }
        )
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        summary_df = pl.DataFrame(
            {
                "response_key": ["FOPR", "FOPT:OP1", "FOPR:OP3", "FLAP", "F*"],
                "time": pl.Series(
                    [datetime(2000, 1, i) for i in range(1, 6)]  # noqa: DTZ001
                ).dt.cast_time_unit("ms"),
                "values": pl.Series([0.0, 1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
            }
        )

        mapping_before = experiment.response_key_to_response_type
        smry_config_before = experiment.simulation_response_configuration["summary"]

        assert not ensemble.experiment._has_finalized_response_keys("summary")
        ensemble.save_response("summary", summary_df, 0)

        assert ensemble.experiment._has_finalized_response_keys("summary")
        assert ensemble.experiment.response_key_to_response_type == {
            "FOPR:OP3": "summary",
            "F*": "summary",
            "FLAP": "summary",
            "FOPR": "summary",
            "FOPT:OP1": "summary",
        }
        assert ensemble.experiment.response_type_to_response_keys == {
            "summary": ["F*", "FLAP", "FOPR", "FOPR:OP3", "FOPT:OP1"]
        }

        mapping_after = experiment.response_key_to_response_type
        smry_config_after = experiment.simulation_response_configuration["summary"]

        assert set(mapping_before) == set()
        assert set(smry_config_before.keys) == {"*", "FOPR"}

        assert set(mapping_after) == {"F*", "FOPR", "FOPT:OP1", "FOPR:OP3", "FLAP"}
        assert set(smry_config_after.keys) == {
            "FOPR",
            "FOPT:OP1",
            "FOPR:OP3",
            "FLAP",
            "F*",
        }


def test_that_load_responses_throws_exception(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=1)

        with pytest.raises(
            expected_exception=ValueError, match="I_DONT_EXIST is not a response"
        ):
            ensemble.load_responses("I_DONT_EXIST", (1,))


def test_keyword_type_checks(snake_oil_default_storage):
    assert (
        "BPR:1,3,8"
        in snake_oil_default_storage.experiment.response_type_to_response_keys[
            "summary"
        ]
    )


def test_keyword_type_checks_missing_key(snake_oil_default_storage):
    assert (
        "nokey"
        not in snake_oil_default_storage.experiment.response_type_to_response_keys[
            "summary"
        ]
    )


@pytest.mark.filterwarnings("ignore:.*Use load_responses.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_load_scalars_returns_empty_dataframe_when_no_scalars_exist(
    snake_oil_case,
):
    with open_storage(snake_oil_case.ens_path, mode="w") as storage:
        experiment = storage.create_experiment()
        empty_case = experiment.create_ensemble(name="new_case", ensemble_size=25)

        data = [
            empty_case.load_scalars(),
        ]

        for dataframe in data:
            assert isinstance(dataframe, pl.DataFrame)
            assert dataframe.is_empty()


def test_save_response_will_create_realization_directory(storage):
    # Given a fresh ensemble storage with no realizations
    dummy_ensemble = storage.create_experiment(
        experiment_config={
            "response_configuration": [
                SummaryConfig(keys=["DUMMY"]).model_dump(mode="json")
            ]
        }
    ).create_ensemble(name="dummy", ensemble_size=1)
    assert dummy_ensemble._path.exists(), "Assumptions for test has changed"
    assert not (dummy_ensemble._path / "realization-0").exists()

    # When a response is saved:
    dummy_ensemble.save_response(
        "summary",
        pl.DataFrame(
            {
                "response_key": ["DUMMY"],
                "time": pl.Series([datetime(2000, 1, 1)]).dt.cast_time_unit("ms"),  # noqa: DTZ001
                "values": pl.Series([0.0], dtype=pl.Float32),
            }
        ),
        0,
    )

    # Then the realization directory was implicitly created
    assert (dummy_ensemble._path / "realization-0").exists()
    assert list(dummy_ensemble._path.glob("realization-0/*.parquet"))


def test_save_response_will_not_recreate_ensemble_directory(storage):
    dummy_ensemble = storage.create_experiment().create_ensemble(
        name="dummy", ensemble_size=1
    )
    # Emulate a user deleting storage:
    shutil.rmtree(dummy_ensemble._path)

    # Function test will raise:
    with pytest.raises(FileNotFoundError):
        dummy_ensemble.save_response(
            "summary",
            pl.DataFrame(
                {
                    "response_key": ["DUMMY"],
                    "time": pl.Series([datetime(2000, 1, 1)]).dt.cast_time_unit("ms"),  # noqa: DTZ001
                    "values": pl.Series([0.0], dtype=pl.Float32),
                }
            ),
            0,
        )

    # and the ensemble path will not be created
    assert not dummy_ensemble._path.exists()
