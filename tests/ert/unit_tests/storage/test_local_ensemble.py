import io
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from pydantic import ValidationError

from ert.analysis.event import AnalysisCompleteEvent, AnalysisMatrixEvent, DataSection
from ert.config import GenKwConfig, SummaryConfig
from ert.storage import LocalExperiment, open_storage
from ert.storage.blob_data import (
    BlobStorageData,
    BlobType,
    MatrixStorageData,
    ObservationReportData,
)
from ert.storage.local_ensemble import (
    _write_responses_to_storage,
)
from ert.storage.mode import ModeError


def test_that_load_scalar_keys_loads_all_parameters(tmp_path):
    """Test that load_scalar_keys loads all scalar parameters when keys=None."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param2",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param3",
                        group="group2",
                        distribution={"name": "normal", "mean": 0, "std": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        # Save parameters
        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                    "param2": [4.0, 5.0, 6.0],
                    "param3": [7.0, 8.0, 9.0],
                }
            )
        )

        # Load all parameters
        df = ensemble.load_scalar_keys()
        assert df.shape == (3, 4)
        assert "realization" in df.columns
        assert "param1" in df.columns
        assert "param2" in df.columns
        assert "param3" in df.columns
        assert df["param1"].to_list() == [1.0, 2.0, 3.0]


def test_that_load_scalar_keys_loads_specific_parameters(tmp_path):
    """Test that load_scalar_keys loads only specified parameters."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param2",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param3",
                        group="group2",
                        distribution={"name": "normal", "mean": 0, "std": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                    "param2": [4.0, 5.0, 6.0],
                    "param3": [7.0, 8.0, 9.0],
                }
            )
        )

        # Load only param1 and param3
        df = ensemble.load_scalar_keys(keys=["param1", "param3"])
        assert df.shape == (3, 3)
        assert "realization" in df.columns
        assert "param1" in df.columns
        assert "param3" in df.columns
        assert "param2" not in df.columns


def test_that_load_scalar_keys_filters_by_realizations(tmp_path):
    """Test that load_scalar_keys filters by specified realizations."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2, 3, 4],
                    "param1": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            )
        )

        # Load only realizations 1 and 3
        df = ensemble.load_scalar_keys(keys=["param1"], realizations=np.array([1, 3]))
        assert df.shape == (2, 2)
        assert df["realization"].to_list() == [1, 3]
        assert df["param1"].to_list() == [2.0, 4.0]


def test_that_load_scalar_keys_raises_key_error_for_missing_parameters(tmp_path):
    """Test that load_scalar_keys raises KeyError for non-existent parameters."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        with pytest.raises(KeyError, match="No SCALAR dataset in storage"):
            ensemble.load_scalar_keys(keys=["param1"])


def test_that_load_scalar_keys_raises_key_error_for_unregistered_parameters(tmp_path):
    """Test that load_scalar_keys raises KeyError for parameters not in experiment."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                }
            )
        )

        with pytest.raises(
            KeyError,
            match="Parameters not registered to the experiment: \\{'param2'\\}",
        ):
            ensemble.load_scalar_keys(keys=["param1", "param2"])


def test_that_load_scalar_keys_raises_index_error_for_missing_realizations(tmp_path):
    """Test that load_scalar_keys raises IndexError when no matching realizations."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                }
            )
        )

        with pytest.raises(
            IndexError,
            match="No matching realizations \\[5 6\\] found for \\['param1'\\]",
        ):
            ensemble.load_scalar_keys(keys=["param1"], realizations=np.array([5, 6]))


def test_that_save_blob_raises_in_read_mode(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        storage.create_ensemble(experiment, ensemble_size=1, iteration=0, name="prior")

    with open_storage(tmp_path, mode="r") as storage:
        ensemble = next(iter(storage.ensembles))
        event = AnalysisCompleteEvent(
            data=DataSection(
                header=["x"],
                data=[(1,)],
            ),
            update_algorithm="ensemble_smoother",
        )
        with pytest.raises(ModeError):
            ensemble.save_blob(event)


def test_that_observation_report_blob_writes_parquet_metadata_and_can_be_loaded(
    tmp_path,
):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        event = AnalysisCompleteEvent(
            data=DataSection(
                header=["observation_key", "status", "value"],
                data=[
                    ("OBS_1", "Active", 1.5),
                    ("OBS_2", "Deactivated, outlier", 2.0),
                ],
            ),
            update_algorithm="ensemble_smoother",
        )
        ensemble.save_blob(event)

        blob_dir = ensemble._path / "blobs"

        assert blob_dir.is_dir(), "Expected blob directory to be created"

        blob_files = list(blob_dir.glob("*.blob"))
        json_files = list(blob_dir.glob("*.blob.json"))
        assert len(blob_files) == 1
        assert len(json_files) == 1

        loaded_df = pl.read_parquet(blob_files[0])
        assert loaded_df.columns == ["observation_key", "status", "value"]
        assert len(loaded_df) == 2
        assert loaded_df["observation_key"][0] == "OBS_1"

        metadata = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert metadata["blob_info"]["blob_type"] == "observation_report"
        assert metadata["blob_info"]["update_algorithm"] == "ensemble_smoother"
        assert metadata["name"] == "observation_report"
        assert metadata["file_size"] > 0

        loaded_blobs = ensemble.load_blobs(BlobType.OBSERVATION_REPORT)
        assert len(loaded_blobs) == 1
        assert isinstance(loaded_blobs[0], BlobStorageData)
        assert isinstance(loaded_blobs[0].blob_info, ObservationReportData)
        assert loaded_blobs[0].blob_info.update_algorithm == "ensemble_smoother"
        assert loaded_blobs[0].name == "observation_report"
        assert loaded_blobs[0].file_type == "application/parquet"


@pytest.mark.parametrize(
    ("blob_event", "expected_exception"),
    [
        pytest.param(
            AnalysisMatrixEvent.model_construct(
                event_type="AnalysisMatrixEvent",
                name="bad_shape",
                sparse=False,
                shape=(2,),
                data_type="float64",
                update_algorithm="enif",
                matrix_bytes=b"matrix-bytes",
            ),
            ValidationError,
            id="matrix-shape-has-one-dimension",
        ),
        pytest.param(
            AnalysisMatrixEvent.model_construct(
                event_type="AnalysisMatrixEvent",
                name="bad_bytes",
                sparse=False,
                shape=(1, 1),
                data_type="float64",
                update_algorithm="enif",
                matrix_bytes="matrix-bytes",
            ),
            TypeError,
            id="matrix-bytes-are-text",
        ),
        pytest.param(
            AnalysisCompleteEvent.model_construct(
                event_type="AnalysisCompleteEvent",
                data={"header": ["x"], "data": [(1,)]},
                update_algorithm="ensemble_smoother",
            ),
            AttributeError,
            id="observation-data-is-a-dict",
        ),
    ],
)
def test_that_save_blob_rejects_malformed_blob_events(
    tmp_path,
    blob_event,
    expected_exception,
):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        with pytest.raises(expected_exception):
            ensemble.save_blob(blob_event)

        blob_dir = ensemble._path / "blobs"
        assert not list(blob_dir.glob("*.blob"))
        assert not list(blob_dir.glob("*.json"))


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(b"{", id="invalid-json"),
        pytest.param(
            json.dumps(
                {
                    "uri": "metadata.blob",
                    "file_size": 1,
                    "file_type": "application/parquet",
                    "name": "observation_report",
                }
            ).encode("utf-8"),
            id="missing-blob-info",
        ),
        pytest.param(
            json.dumps(
                {
                    "uri": "metadata.blob",
                    "file_size": 1,
                    "file_type": "application/parquet",
                    "name": "observation_report",
                    "blob_info": {
                        "blob_type": "unknown",
                        "update_algorithm": "ensemble_smoother",
                    },
                }
            ).encode("utf-8"),
            id="unknown-blob-type",
        ),
    ],
)
def test_that_load_blobs_rejects_malformed_blob_metadata(
    tmp_path,
    metadata,
):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )
        blob_dir = ensemble._path / "blobs"
        blob_dir.mkdir()
        (blob_dir / "metadata.blob.json").write_bytes(metadata)

        with pytest.raises(ValidationError):
            ensemble.load_blobs()


@pytest.mark.parametrize(
    "uri_template",
    [
        pytest.param("missing.blob", id="missing-file"),
        pytest.param("../index.json", id="parent-directory"),
        pytest.param("{ensemble_path}/index.json", id="absolute-path"),
    ],
)
def test_that_load_blob_raises_file_not_found_for_invalid_uri(
    tmp_path,
    uri_template,
):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )
        uri = uri_template.format(ensemble_path=ensemble._path)

        with pytest.raises(FileNotFoundError):
            ensemble.load_blob(uri)


def test_that_sparse_and_dense_matrix_blobs_can_be_saved_and_loaded(tmp_path):

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        # Create a sparse matrix and serialize it
        sparse_matrix = sp.csc_array(np.array([[1.0, 0.0], [0.0, 2.0]]))
        sparse_buf = io.BytesIO()
        sp.save_npz(sparse_buf, sparse_matrix)
        sparse_bytes = sparse_buf.getvalue()

        sparse_event = AnalysisMatrixEvent(
            name="H",
            sparse=True,
            shape=(2, 2),
            data_type="float64",
            update_algorithm="enif",
            matrix_bytes=sparse_bytes,
        )
        ensemble.save_blob(sparse_event)

        # Create a dense matrix and serialize it
        dense_matrix = np.array([[3.0, 4.0], [5.0, 6.0]])
        dense_buf = io.BytesIO()
        np.save(dense_buf, dense_matrix)
        dense_bytes = dense_buf.getvalue()

        dense_event = AnalysisMatrixEvent(
            name="Prec_u",
            sparse=False,
            shape=(2, 2),
            data_type="float64",
            update_algorithm="enif",
            matrix_bytes=dense_bytes,
        )
        ensemble.save_blob(dense_event)

        # load_blobs returns all matrix metadata
        all_blobs = ensemble.load_blobs(BlobType.MATRIX)
        assert len(all_blobs) == 2
        assert all(isinstance(m, BlobStorageData) for m in all_blobs)
        assert all(isinstance(m.blob_info, MatrixStorageData) for m in all_blobs)

        by_name = {m.name: m for m in all_blobs}
        assert "H" in by_name
        assert "Prec_u" in by_name

        h_meta = by_name["H"]
        assert isinstance(h_meta.blob_info, MatrixStorageData)
        assert h_meta.blob_info.sparse is True
        assert h_meta.blob_info.shape == (2, 2)
        assert h_meta.file_type == "application/x-npz"

        prec_meta = by_name["Prec_u"]
        assert isinstance(prec_meta.blob_info, MatrixStorageData)
        assert prec_meta.blob_info.sparse is False
        assert prec_meta.blob_info.shape == (2, 2)
        assert prec_meta.file_type == "application/x-npy"

        h_bytes = ensemble.load_blob(h_meta.uri)
        loaded_sparse = sp.load_npz(io.BytesIO(h_bytes))
        np.testing.assert_array_equal(loaded_sparse.toarray(), sparse_matrix.toarray())

        prec_bytes = ensemble.load_blob(prec_meta.uri)
        loaded_dense = np.load(io.BytesIO(prec_bytes))
        np.testing.assert_array_equal(loaded_dense, dense_matrix)

        assert h_meta.blob_info.parameter_group_sizes == {}
        assert prec_meta.blob_info.parameter_group_sizes == {}


def test_that_parameter_group_sizes_is_stored_in_matrix_blob_metadata(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        dense_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        dense_buf = io.BytesIO()
        np.save(dense_buf, dense_matrix)

        event = AnalysisMatrixEvent(
            name="K",
            sparse=False,
            shape=(2, 2),
            data_type="float64",
            update_algorithm="enif",
            parameter_group_sizes={"PORO": 8, "PERM": 3},
            matrix_bytes=dense_buf.getvalue(),
        )
        ensemble.save_blob(event)

        blobs = ensemble.load_blobs(BlobType.MATRIX)
        assert len(blobs) == 1
        assert isinstance(blobs[0].blob_info, MatrixStorageData)
        assert blobs[0].blob_info.parameter_group_sizes == {"PORO": 8, "PERM": 3}


async def test_that_writing_and_reading_empty_response_in_storage_results_in_empty_df_with_schema_columns(  # noqa: E501
    tmp_path, monkeypatch
):
    """This test writes an empty set of responses to storage and asserts that the
    parquet file contains the correct columns.
    Then the test also checks that loading said file through ensemble results in an
    empty dataframe.
    """
    response_column_scheme = ["realization", "response_key", "time", "values"]
    empty_response = pl.DataFrame({"response_key": [], "time": [], "values": []})
    monkeypatch.setattr(SummaryConfig, "read_from_file", lambda *args: empty_response)
    monkeypatch.setattr(
        LocalExperiment,
        "simulation_response_configuration",
        {"summary": SummaryConfig()},
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        await _write_responses_to_storage(str(tmp_path), 0, ensemble)

        summary_response_path = ensemble._realization_dir(0) / "summary.parquet"
        assert Path(summary_response_path).is_file()
        responses = pl.read_parquet(summary_response_path)
        assert responses.is_empty()
        assert responses.columns == response_column_scheme

        # Mock response config to contain a response key, else the parquet file
        # will never be read as the code exits earlier.
        monkeypatch.setattr(
            LocalExperiment,
            "simulation_response_configuration",
            {"summary": SummaryConfig(keys=["FOPR"])},
        )
        monkeypatch.setattr(
            LocalExperiment, "response_key_to_response_type", {"FOPR": "summary"}
        )

        responses = ensemble.load_responses("FOPR", (0,))
        assert responses.is_empty()
        assert responses.columns == response_column_scheme
