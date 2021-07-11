import collections

import pytest

import ert
import ert3


@pytest.mark.requires_ert_storage
def test_init(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)


@pytest.mark.requires_ert_storage
def test_double_init(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    with pytest.raises(ValueError, match="Storage already initialized"):
        ert3.storage.init(workspace=tmpdir)


@pytest.mark.requires_ert_storage
def test_ensemble_size_zero(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    with pytest.raises(ValueError, match="Ensemble cannot have a size <= 0"):
        ert3.storage.init_experiment(
            workspace=tmpdir,
            experiment_name="my_experiment",
            parameters={},
            ensemble_size=0,
            responses=[],
        )


@pytest.mark.requires_ert_storage
def test_none_as_experiment_name(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    with pytest.raises(ValueError, match="Cannot initialize experiment without a name"):
        ert3.storage.init_experiment(
            workspace=tmpdir,
            experiment_name=None,
            parameters={},
            ensemble_size=10,
            responses=[],
        )


@pytest.mark.requires_ert_storage
def test_double_add_experiment(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    ert3.storage.init_experiment(
        workspace=tmpdir,
        experiment_name="my_experiment",
        parameters={},
        ensemble_size=42,
        responses=[],
    )
    with pytest.raises(
        ert3.exceptions.ElementExistsError,
        match="Cannot initialize existing experiment",
    ):
        ert3.storage.init_experiment(
            workspace=tmpdir,
            experiment_name="my_experiment",
            parameters={},
            ensemble_size=42,
            responses=[],
        )


@pytest.mark.requires_ert_storage
def test_add_experiments(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    experiment_names = ["a", "b", "c", "super-experiment", "explosions"]
    experiment_parameter_records = [
        ["x"],
        ["a", "b"],
        ["alpha", "beta"],
        ["oxygen", "heat", "fuel"],
    ]
    experiments = zip(experiment_names, experiment_parameter_records)
    for idx, (experiment_name, experiment_parameter_records) in enumerate(experiments):
        experiment_parameters = {
            key: ["some_coeff"] for key in experiment_parameter_records
        }
        ert3.storage.init_experiment(
            workspace=tmpdir,
            experiment_name=experiment_name,
            parameters=experiment_parameters,
            ensemble_size=42,
            responses=[],
        )
        expected_names = sorted(experiment_names[: idx + 1])
        retrieved_names = sorted(ert3.storage.get_experiment_names(workspace=tmpdir))
        assert expected_names == retrieved_names

        parameters = ert3.storage.get_experiment_parameters(
            workspace=tmpdir, experiment_name=experiment_name
        )
        assert experiment_parameter_records == parameters


@pytest.mark.requires_ert_storage
def test_get_parameters_unknown_experiment(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    with pytest.raises(
        ert3.exceptions.NonExistantExperiment,
        match="Cannot get parameters from non-existing experiment: unknown-experiment",
    ):
        ert3.storage.get_experiment_parameters(
            workspace=tmpdir, experiment_name="unknown-experiment"
        )


def _assert_equal_data(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert sorted(a.keys()) == sorted(b.keys())
        for key in a.keys():
            _assert_equal_data(a[key], b[key])
    elif isinstance(a, list):
        assert isinstance(b, list)
        assert len(a) == len(b)
        for elem_a, elem_b in zip(a, b):
            _assert_equal_data(elem_a, elem_b)
    else:
        assert a == b


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize(
    "raw_ensrec",
    (
        [{"data": [i + 0.5, i + 1.1, i + 2.2]} for i in range(3)],
        [{"data": {"a": i + 0.5, "b": i + 1.1, "c": i + 2.2}} for i in range(5)],
        [{"data": {2: i + 0.5, 5: i + 1.1, 7: i + 2.2}} for i in range(2)],
    ),
)
def test_add_and_get_ensemble_record(tmpdir, raw_ensrec, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    ensrecord = ert.data.EnsembleRecord(records=raw_ensrec)
    ert3.storage.add_ensemble_record(
        workspace=tmpdir,
        record_name="my_ensemble_record",
        ensemble_record=ensrecord,
    )

    retrieved_ensrecord = ert3.storage.get_ensemble_record(
        workspace=tmpdir, record_name="my_ensemble_record"
    )

    assert ensrecord == retrieved_ensrecord


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize(
    "raw_ensrec",
    (
        [{"data": [i + 0.5, i + 1.1, i + 2.2]} for i in range(3)],
        [{"data": {"a": i + 0.5, "b": i + 1.1, "c": i + 2.2}} for i in range(5)],
        [{"data": {2: i + 0.5, 5: i + 1.1, 7: i + 2.2}} for i in range(2)],
    ),
)
def test_add_and_get_ensemble_parameter_record(tmpdir, raw_ensrec, ert_storage):
    """This tests a workaround so that webviz-ert is able to visualise parameters.
    It expects records which are marked as parameter to contain only one value
    per realisation, while ERT 3 uses multiple variables per record. That is, in
    ERT 3, it is possible to specify a record named "coefficients" which
    contains the variables "a", "b" and "c". These are rendered as column
    labels, which ERT Storage accepts, but webviz-ert doesn't know how to use.

    The workaround involves splitting up any parameter record into its variable
    parts, so "coefficients.a", "coefficients.b", "coefficients.c". This is an
    implementation detail that exists entirely within the `ert3.storage` module,
    and isn't visible outside of it.

    In order to test this behaviour, we use a testing-only kwarg called
    `_flatten` in the function `ert3.storage.get_ensemble_record_names`, which
    lets us see the record names that are hidden outside of the `ert3.storage`
    module.

    """
    raw_data = raw_ensrec[0]["data"]
    assert isinstance(raw_data, (list, dict))
    if isinstance(raw_data, list):
        indices = [str(x) for x in range(len(raw_data))]
    else:
        indices = [str(x) for x in raw_data]

    ert3.storage.init(workspace=tmpdir)
    ert3.storage.init_experiment(
        workspace=tmpdir,
        experiment_name="experiment_name",
        parameters={"my_ensemble_record": indices},
        ensemble_size=len(raw_ensrec),
        responses=[],
    )

    ensrecord = ert.data.EnsembleRecord(records=raw_ensrec)
    ert3.storage.add_ensemble_record(
        workspace=tmpdir,
        experiment_name="experiment_name",
        record_name="my_ensemble_record",
        ensemble_record=ensrecord,
    )

    record_names = ert3.storage.get_ensemble_record_names(
        workspace=tmpdir, experiment_name="experiment_name", _flatten=False
    )
    assert {f"my_ensemble_record.{x}" for x in indices} == set(record_names)

    retrieved_ensrecord = ert3.storage.get_ensemble_record(
        workspace=tmpdir,
        experiment_name="experiment_name",
        record_name="my_ensemble_record",
    )

    assert ensrecord == retrieved_ensrecord


@pytest.mark.requires_ert_storage
def test_add_ensemble_record_twice(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    ensrecord = ert.data.EnsembleRecord(records=[{"data": [42]}])
    ert3.storage.add_ensemble_record(
        workspace=tmpdir, record_name="my_ensemble_record", ensemble_record=ensrecord
    )

    with pytest.raises(
        ert3.exceptions.ElementExistsError,
        match="Record already exists",
    ):
        ert3.storage.add_ensemble_record(
            workspace=tmpdir,
            record_name="my_ensemble_record",
            ensemble_record=ensrecord,
        )


@pytest.mark.requires_ert_storage
def test_get_unknown_ensemble_record(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    with pytest.raises(ert3.exceptions.ElementMissingError):
        ert3.storage.get_ensemble_record(
            workspace=tmpdir, record_name="my_ensemble_record"
        )


@pytest.mark.requires_ert_storage
def test_add_and_get_experiment_ensemble_record(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    ensemble_size = 5
    for eid in range(1, 2):
        experiment = eid * "e"
        ert3.storage.init_experiment(
            workspace=tmpdir,
            experiment_name=experiment,
            parameters={},
            ensemble_size=ensemble_size,
            responses=[],
        )
        for nid in range(1, 3):
            name = nid * "n"
            ensemble_record = ert.data.EnsembleRecord(
                records=[
                    ert.data.NumericalRecord(data=[nid * eid * rid])
                    for rid in range(ensemble_size)
                ]
            )
            ert3.storage.add_ensemble_record(
                workspace=tmpdir,
                record_name=name,
                ensemble_record=ensemble_record,
                experiment_name=experiment,
            )

    for eid in range(1, 2):
        experiment = eid * "e"
        for nid in range(1, 3):
            name = nid * "n"
            ensemble_record = ert.data.EnsembleRecord(
                records=[
                    ert.data.NumericalRecord(data=[nid * eid * rid])
                    for rid in range(ensemble_size)
                ]
            )
            fetched_ensemble_record = ert3.storage.get_ensemble_record(
                workspace=tmpdir, record_name=name, experiment_name=experiment
            )
            assert ensemble_record == fetched_ensemble_record


@pytest.mark.requires_ert_storage
def test_add_ensemble_record_to_non_existing_experiment(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    with pytest.raises(
        ert3.exceptions.NonExistantExperiment,
        match="Cannot add my_record data to non-existing experiment",
    ):
        ert3.storage.add_ensemble_record(
            workspace=tmpdir,
            record_name="my_record",
            ensemble_record=ert.data.EnsembleRecord(records=[{"data": [0, 1, 2]}]),
            experiment_name="non_existing_experiment",
        )


@pytest.mark.requires_ert_storage
def test_get_ensemble_record_to_non_existing_experiment(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    with pytest.raises(
        ert3.exceptions.NonExistantExperiment,
        match="Cannot get my_record data, no experiment named: non_existing_experiment",
    ):
        ert3.storage.get_ensemble_record(
            workspace=tmpdir,
            record_name="my_record",
            experiment_name="non_existing_experiment",
        )


@pytest.mark.requires_ert_storage
def test_get_record_names(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    ensemble_size = 5
    experiment_records = collections.defaultdict(list)
    for eid in [1, 2, 3]:
        experiment = "e" + str(eid)
        ert3.storage.init_experiment(
            workspace=tmpdir,
            experiment_name=experiment,
            parameters={},
            ensemble_size=ensemble_size,
            responses=[],
        )
        for nid in range(1, 3):
            name = nid * "n"
            ensemble_record = ert.data.EnsembleRecord(
                records=[
                    ert.data.NumericalRecord(data=[0]) for rid in range(ensemble_size)
                ]
            )
            ert3.storage.add_ensemble_record(
                workspace=tmpdir,
                record_name=name,
                ensemble_record=ensemble_record,
                experiment_name=experiment,
            )
            experiment_records[str(experiment)].append(name)

            recnames = ert3.storage.get_ensemble_record_names(
                workspace=tmpdir, experiment_name=experiment
            )
            assert sorted(experiment_records[str(experiment)]) == sorted(recnames)


@pytest.mark.requires_ert_storage
def test_get_record_names_unknown_experiment(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)

    with pytest.raises(
        ert3.exceptions.NonExistantExperiment,
        match="Cannot get record names of non-existing experiment: unknown-experiment",
    ):
        ert3.storage.get_ensemble_record_names(
            workspace=tmpdir, experiment_name="unknown-experiment"
        )


@pytest.mark.requires_ert_storage
def test_delete_experiment(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    ert3.storage.init_experiment(
        workspace=tmpdir,
        experiment_name="test",
        parameters={},
        ensemble_size=42,
        responses=[],
    )

    assert "test" in ert3.storage.get_experiment_names(workspace=tmpdir)

    with pytest.raises(
        ert3.exceptions.NonExistantExperiment,
        match="Experiment does not exist: does_not_exist",
    ):
        ert3.storage.delete_experiment(
            workspace=tmpdir, experiment_name="does_not_exist"
        )

    ert3.storage.delete_experiment(workspace=tmpdir, experiment_name="test")

    assert "test" not in ert3.storage.get_experiment_names(workspace=tmpdir)


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize(
    "responses, records, expected_result",
    [
        (["resp1", "resp2"], ["resp1", "resp2", "rec1"], ["resp1", "resp2"]),
        ([], ["resp1", "resp2", "rec1"], []),
        (["resp1", "resp2"], ["rec1"], []),
    ],
)
def test_get_ensemble_responses(
    responses, records, expected_result, tmpdir, ert_storage
):
    ert3.storage.init(workspace=tmpdir)
    experiment = "exp"
    ert3.storage.init_experiment(
        workspace=tmpdir,
        experiment_name=experiment,
        parameters={},
        ensemble_size=1,
        responses=responses,
    )
    for name in records:
        ensemble_record = ert.data.EnsembleRecord(
            records=[ert.data.NumericalRecord(data=[1, 2, 3])]
        )
        ert3.storage.add_ensemble_record(
            workspace=tmpdir,
            record_name=name,
            ensemble_record=ensemble_record,
            experiment_name=experiment,
        )

    fetched_ensemble_responses = ert3.storage.get_experiment_responses(
        workspace=tmpdir, experiment_name=experiment
    )

    assert set(fetched_ensemble_responses) == set(expected_result)


@pytest.mark.requires_ert_storage
def test_ensemble_responses_and_parameters(tmpdir, ert_storage):
    ert3.storage.init(workspace=tmpdir)
    responses = ["resp1", "resp2"]
    experiment = "exp"
    with pytest.raises(
        ert3.exceptions.StorageError,
        match="parameters and responses cannot have a name in common",
    ):
        ert3.storage.init_experiment(
            workspace=tmpdir,
            experiment_name=experiment,
            parameters={"resp1": ["some-key"]},
            ensemble_size=1,
            responses=responses,
        )


@pytest.mark.requires_ert_storage
def test_add_and_get_blob_record(tmpdir, ert_storage):
    # this test should be added to the
    # test_add_and_get_ensemble_parameter_record() test above
    record_name = "test_data"
    experiment_name = "exp"
    data = b"ASDFQEWRGFWEzwz3TRHSDFBVWRET12awd"

    ert3.storage.init(workspace=tmpdir)
    ert3.storage.init_experiment(
        workspace=tmpdir,
        experiment_name=experiment_name,
        parameters={},
        ensemble_size=1,
        responses=[],
    )

    ensemble_record = ert.data.EnsembleRecord(records=[ert.data.Record(data=[data])])
    ert3.storage.add_ensemble_record(
        workspace=tmpdir,
        record_name=record_name,
        ensemble_record=ensemble_record,
        experiment_name=experiment_name,
    )

    blob_ensemble_record = ert3.storage.get_ensemble_record(
        workspace=tmpdir, record_name=record_name, experiment_name=experiment_name
    )

    assert blob_ensemble_record.records[0].data[0] == data
