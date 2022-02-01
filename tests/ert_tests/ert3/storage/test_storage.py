import collections

import pytest

import ert
from ert_shared.async_utils import get_event_loop


@pytest.mark.requires_ert_storage
def test_init(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)


@pytest.mark.requires_ert_storage
def test_double_init(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)
    with pytest.raises(ValueError, match="Storage already initialized"):
        ert.storage.init(workspace_name=tmpdir)


@pytest.mark.requires_ert_storage
def test_ensemble_size_zero(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)
    with pytest.raises(ValueError, match="Ensemble cannot have a size <= 0"):
        ert.storage.init_experiment(
            experiment_name="my_experiment",
            parameters={},
            ensemble_size=0,
            responses=[],
        )


@pytest.mark.requires_ert_storage
def test_none_as_experiment_name(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)
    with pytest.raises(ValueError, match="Cannot initialize experiment without a name"):
        ert.storage.init_experiment(
            experiment_name=None,
            parameters={},
            ensemble_size=10,
            responses=[],
        )


@pytest.mark.requires_ert_storage
def test_double_add_experiment(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)
    ert.storage.init_experiment(
        experiment_name="my_experiment",
        parameters={},
        ensemble_size=42,
        responses=[],
    )
    with pytest.raises(
        ert.exceptions.ElementExistsError,
        match="Cannot initialize existing experiment",
    ):
        ert.storage.init_experiment(
            experiment_name="my_experiment",
            parameters={},
            ensemble_size=42,
            responses=[],
        )


@pytest.mark.requires_ert_storage
def test_add_experiments(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)

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
        ert.storage.init_experiment(
            experiment_name=experiment_name,
            parameters=experiment_parameters,
            ensemble_size=42,
            responses=[],
        )
        expected_names = sorted(experiment_names[: idx + 1])
        retrieved_names = sorted(
            ert.storage.get_experiment_names(workspace_name=tmpdir)
        )
        assert expected_names == retrieved_names

        parameters = ert.storage.get_experiment_parameters(
            experiment_name=experiment_name
        )
        assert experiment_parameter_records == parameters


@pytest.mark.requires_ert_storage
def test_get_parameters_unknown_experiment(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)

    with pytest.raises(
        ert.exceptions.NonExistantExperiment,
        match="Cannot get parameters from non-existing experiment: unknown-experiment",
    ):
        ert.storage.get_experiment_parameters(experiment_name="unknown-experiment")


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
        [{"data": b"asdfkasjdhjflkjah21WE123TTDSG34f"}],
    ),
)
def test_add_and_get_ensemble_record(
    tmpdir, raw_ensrec, raw_ensrec_to_records, ert_storage
):
    ert.storage.init(workspace_name=tmpdir)

    ensrecord = ert.data.RecordCollection(records=raw_ensrec_to_records(raw_ensrec))
    future = ert.storage.transmit_record_collection(
        record_coll=ensrecord,
        record_name="my_ensemble_record",
        workspace_name=tmpdir,
    )
    get_event_loop().run_until_complete(future)

    res = ert.storage.get_ensemble_record(
        workspace_name=tmpdir,
        record_name="my_ensemble_record",
        ensemble_size=len(raw_ensrec),
    )

    assert res == ensrecord


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize(
    "raw_ensrec",
    (
        [{"data": [0.5, 1.1, 2.2]}],
        [{"data": {"a": 0.5, "b": 1.1, "c": 2.2}}],
        [{"data": {2: 0.5, 5: 1.1, 7: 2.2}}],
        [{"data": b"asdfkasjdhjflkjah21WE123TTDSG34f"}],
    ),
)
def test_add_and_get_uniform_ensemble_record(
    tmpdir, raw_ensrec, raw_ensrec_to_records, ert_storage
):
    ert.storage.init(workspace_name=tmpdir)
    ens_size = 5
    ensrecord = ert.data.RecordCollection(
        records=raw_ensrec_to_records(raw_ensrec),
        length=ens_size,
        collection_type=ert.data.RecordCollectionType.UNIFORM,
    )
    future = ert.storage.transmit_record_collection(
        record_coll=ensrecord,
        record_name="my_ensemble_record",
        workspace_name=tmpdir,
    )
    get_event_loop().run_until_complete(future)

    res = ert.storage.get_ensemble_record(
        workspace_name=tmpdir, record_name="my_ensemble_record", ensemble_size=ens_size
    )
    assert res == ensrecord


@pytest.mark.requires_ert_storage
def test_add_ensemble_record_twice(tmpdir, raw_ensrec_to_records, ert_storage):
    ert.storage.init(workspace_name=tmpdir)

    ensrecord = ert.data.RecordCollection(
        records=raw_ensrec_to_records([{"data": [42]}])
    )
    future = ert.storage.transmit_record_collection(
        record_coll=ensrecord,
        record_name="my_ensemble_record",
        workspace_name=tmpdir,
    )
    get_event_loop().run_until_complete(future)

    with pytest.raises(
        ert.exceptions.ElementExistsError,
    ):
        future = ert.storage.transmit_record_collection(
            record_coll=ensrecord,
            record_name="my_ensemble_record",
            workspace_name=tmpdir,
        )
        get_event_loop().run_until_complete(future)


@pytest.mark.requires_ert_storage
def test_get_unstored_ensemble_record(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)

    with pytest.raises(ert.exceptions.ElementMissingError):
        ert.storage.get_ensemble_record(
            workspace_name=tmpdir, record_name="my_ensemble_record", ensemble_size=2
        )


@pytest.mark.requires_ert_storage
def test_add_and_get_experiment_ensemble_record(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)

    ensemble_size = 5
    for eid in range(1, 2):
        experiment = eid * "e"
        ert.storage.init_experiment(
            experiment_name=experiment,
            parameters={},
            ensemble_size=ensemble_size,
            responses=[],
        )
        for nid in range(1, 3):
            name = nid * "n"
            ensemble_record = ert.data.RecordCollection(
                records=tuple(
                    [
                        ert.data.NumericalRecord(data=[nid * eid * rid])
                        for rid in range(ensemble_size)
                    ]
                )
            )
            get_event_loop().run_until_complete(
                ert.storage.transmit_record_collection(
                    record_coll=ensemble_record,
                    record_name=name,
                    workspace_name=tmpdir,
                    experiment_name=experiment,
                )
            )

    for eid in range(1, 2):
        experiment = eid * "e"
        for nid in range(1, 3):
            name = nid * "n"
            ensemble_record = ert.data.RecordCollection(
                records=tuple(
                    [
                        ert.data.NumericalRecord(data=[nid * eid * rid])
                        for rid in range(ensemble_size)
                    ]
                )
            )
            fetched_ensemble_record = ert.storage.get_ensemble_record(
                workspace_name=tmpdir,
                record_name=name,
                experiment_name=experiment,
                ensemble_size=ensemble_size,
            )
            assert ensemble_record == fetched_ensemble_record


@pytest.mark.requires_ert_storage
def test_add_ensemble_record_to_non_existing_experiment(
    tmpdir, ert_storage, raw_ensrec_to_records
):
    ert.storage.init(workspace_name=tmpdir)
    with pytest.raises(
        ert.exceptions.NonExistantExperiment,
        match="Experiment non_existing_experiment does not exist",
    ):
        get_event_loop().run_until_complete(
            ert.storage.transmit_record_collection(
                record_coll=ert.data.RecordCollection(
                    records=raw_ensrec_to_records([{"data": [0, 1, 2]}])
                ),
                record_name="my_record",
                workspace_name=tmpdir,
                experiment_name="non_existing_experiment",
            )
        )


@pytest.mark.requires_ert_storage
def test_get_ensemble_record_to_non_existing_experiment(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)
    with pytest.raises(
        ert.exceptions.NonExistantExperiment,
        match="Experiment non_existing_experiment does not exist",
    ):
        ert.storage.get_ensemble_record(
            workspace_name=tmpdir,
            record_name="my_record",
            experiment_name="non_existing_experiment",
            ensemble_size=2,
        )


@pytest.mark.requires_ert_storage
def test_get_record_names(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)
    ensemble_size = 5
    experiment_records = collections.defaultdict(list)
    for eid in [1, 2, 3]:
        experiment = "e" + str(eid)
        ert.storage.init_experiment(
            experiment_name=experiment,
            parameters={},
            ensemble_size=ensemble_size,
            responses=[],
        )
        for nid in range(1, 3):
            name = nid * "n"
            ensemble_record = ert.data.RecordCollection(
                records=tuple(
                    [ert.data.NumericalRecord(data=[0]) for rid in range(ensemble_size)]
                )
            )
            future = ert.storage.transmit_record_collection(
                record_coll=ensemble_record,
                record_name=name,
                workspace_name=tmpdir,
                experiment_name=experiment,
            )

            get_event_loop().run_until_complete(future)
            experiment_records[str(experiment)].append(name)

            recnames = ert.storage.get_ensemble_record_names(
                workspace_name=tmpdir, experiment_name=experiment
            )
            assert sorted(experiment_records[str(experiment)]) == sorted(recnames)


@pytest.mark.requires_ert_storage
def test_get_record_names_unknown_experiment(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)

    with pytest.raises(
        ert.exceptions.NonExistantExperiment,
        match="Cannot get record names of non-existing experiment: unknown-experiment",
    ):
        ert.storage.get_ensemble_record_names(
            workspace_name=tmpdir, experiment_name="unknown-experiment"
        )


@pytest.mark.requires_ert_storage
def test_delete_experiment(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)
    ert.storage.init_experiment(
        experiment_name="test",
        parameters={},
        ensemble_size=42,
        responses=[],
    )

    assert "test" in ert.storage.get_experiment_names(workspace_name=tmpdir)

    with pytest.raises(
        ert.exceptions.NonExistantExperiment,
        match="Experiment does not exist: does_not_exist",
    ):
        ert.storage.delete_experiment(experiment_name="does_not_exist")

    ert.storage.delete_experiment(experiment_name="test")

    assert "test" not in ert.storage.get_experiment_names(workspace_name=tmpdir)


@pytest.mark.requires_ert_storage
@pytest.mark.parametrize(
    "responses, expected_result",
    [
        (["resp1", "resp2"], ["resp1", "resp2"]),
        ([], []),
        (["resp1"], ["resp1"]),
        (["resp_num", "resp_blob"], ["resp_num"]),
    ],
)
def test_get_ensemble_responses(responses, expected_result, tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)
    experiment = "exp"
    ert.storage.init_experiment(
        experiment_name=experiment,
        parameters=[],
        ensemble_size=1,
        responses=responses,
    )
    # we need to transmitt the responses
    for response_name in responses:
        if "blob" in response_name:
            ensemble_record = ert.data.RecordCollection(
                records=tuple([ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89")])
            )
        else:
            ensemble_record = ert.data.RecordCollection(
                records=tuple([ert.data.NumericalRecord(data=[0]) for rid in range(1)])
            )
        future = ert.storage.transmit_record_collection(
            record_coll=ensemble_record,
            record_name=response_name,
            workspace_name=tmpdir,
            experiment_name=experiment,
        )
        get_event_loop().run_until_complete(future)

    fetched_ensemble_responses = ert.storage.get_experiment_responses(
        experiment_name=experiment
    )

    assert set(fetched_ensemble_responses) == set(expected_result)


@pytest.mark.requires_ert_storage
def test_ensemble_responses_and_parameters(tmpdir, ert_storage):
    ert.storage.init(workspace_name=tmpdir)
    responses = ["resp1", "resp2"]
    experiment = "exp"
    with pytest.raises(
        ert.exceptions.StorageError,
        match="parameters and responses cannot have a name in common",
    ):
        ert.storage.init_experiment(
            experiment_name=experiment,
            parameters={"resp1": ["some-key"]},
            ensemble_size=1,
            responses=responses,
        )
