import os

import pandas as pd

from ert.config import GenDataConfig, GenKwConfig, SummaryConfig
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.storage import open_storage
from ert.storage.realization_state import (
    _SingleRealizationStateDict,
    _SingleRealizationStateDictEntry,
)
from tests.performance_tests.test_memory_usage import make_gen_data, make_summary_data


def test_that_realization_states_with_no_params_or_responses_shows_empty(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=1)

        ensemble._refresh_realization_states()

        states = ensemble._realization_states
        assert states.is_empty()
        assert ensemble._realization_states.to_dataframe().empty


def test_that_realization_states_shows_all_params_present(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            parameters=[
                GenKwConfig(
                    name="PARAMETER_GROUP",
                    forward_init=False,
                    template_file="",
                    transform_function_definitions=[
                        TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
                        TransformFunctionDefinition("KEY2", "UNIFORM", [0, 1]),
                        TransformFunctionDefinition("KEY3", "UNIFORM", [0, 1]),
                    ],
                    output_file="kw.txt",
                    update=True,
                )
            ]
        )
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=25)
        for i in range(1, 25):
            ensemble.save_parameters(
                "PARAMETER_GROUP",
                i,
                pd.DataFrame(
                    data={
                        "names": ["KEY1", "KEY2", "KEY3"],
                        "values": [1, 2, 3],
                        "transformed_values": [2, 4, 6],
                    }
                )
                .set_index(["names"])
                .to_xarray(),
            )

        ensemble._refresh_realization_states()
        state_df_before_combine = ensemble._realization_states.to_dataframe()

        for i in range(1, 25):
            real_state = ensemble._realization_states.get_single_realization_state(i)
            ds_path = ensemble._realization_dir(i) / "PARAMETER_GROUP.nc"

            tob = os.path.getctime(ds_path) if os.path.exists(ds_path) else -1

            assert real_state.has_parameter_key_or_group("PARAMETER_GROUP")
            assert real_state.get_parameter("PARAMETER_GROUP").timestamp == tob

        ensemble.unify_parameters()
        state_df_after_combine = ensemble._realization_states.to_dataframe()

        assert state_df_before_combine["value"].equals(state_df_after_combine["value"])
        assert (
            sum(
                state_df_after_combine["timestamp"]
                - state_df_before_combine["timestamp"]
            )
            > 0
        )

        ds_path = ensemble._path / "PARAMETER_GROUP.nc"
        tob = os.path.getctime(ds_path)

        real_state0 = ensemble._realization_states.get_single_realization_state(0)
        assert not real_state0.has_parameter_key_or_group("PARAMETER_GROUP")

        for i in range(1, 25):
            real_state = ensemble._realization_states.get_single_realization_state(i)

            assert real_state.has_parameter_key_or_group("PARAMETER_GROUP")
            assert real_state.get_parameter("PARAMETER_GROUP").timestamp == tob


def test_that_realization_states_shows_some_params_present(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            parameters=[
                GenKwConfig(
                    name="PARAMETER_GROUP",
                    forward_init=False,
                    template_file="",
                    transform_function_definitions=[
                        TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
                        TransformFunctionDefinition("KEY2", "UNIFORM", [0, 1]),
                        TransformFunctionDefinition("KEY3", "UNIFORM", [0, 1]),
                    ],
                    output_file="kw.txt",
                    update=True,
                ),
                GenKwConfig(
                    name="PARAMETER_GROUP2",
                    forward_init=False,
                    template_file="",
                    transform_function_definitions=[
                        TransformFunctionDefinition("KEY11", "UNIFORM", [0, 1]),
                        TransformFunctionDefinition("KEY21", "UNIFORM", [0, 1]),
                        TransformFunctionDefinition("KEY31", "UNIFORM", [0, 1]),
                    ],
                    output_file="kw.txt",
                    update=True,
                ),
            ]
        )
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=25)
        for i in range(1, 25):
            if i % 2 == 0:
                ensemble.save_parameters(
                    "PARAMETER_GROUP",
                    i,
                    pd.DataFrame(
                        data={
                            "names": ["KEY1", "KEY2", "KEY3"],
                            "values": [1, 2, 3],
                            "transformed_values": [2, 4, 6],
                        }
                    )
                    .set_index(["names"])
                    .to_xarray(),
                )

            if i % 3 == 0:
                ensemble.save_parameters(
                    "PARAMETER_GROUP2",
                    i,
                    pd.DataFrame(
                        data={
                            "names": ["KEY1", "KEY2", "KEY3"],
                            "values": [1, 2, 3],
                            "transformed_values": [2, 4, 6],
                        }
                    )
                    .set_index(["names"])
                    .to_xarray(),
                )
        ensemble._refresh_realization_states()
        state_df_before_combine = ensemble._realization_states.to_dataframe()

        for i in range(1, 25):
            ds_path_1 = ensemble._realization_dir(i) / "PARAMETER_GROUP.nc"
            ds_path_2 = ensemble._realization_dir(i) / "PARAMETER_GROUP2.nc"

            tob_1 = os.path.getctime(ds_path_1) if os.path.exists(ds_path_1) else -1
            tob_2 = os.path.getctime(ds_path_2) if os.path.exists(ds_path_2) else -1

            real_state = ensemble._realization_states.get_single_realization_state(i)
            if i % 6 == 0:
                assert real_state.has_parameter_key_or_group("PARAMETER_GROUP")
                assert real_state.has_parameter_key_or_group("PARAMETER_GROUP2")
                assert real_state.get_parameter("PARAMETER_GROUP").timestamp == tob_1
                assert real_state.get_parameter("PARAMETER_GROUP2").timestamp == tob_2
            elif i % 2 == 0:
                assert real_state.has_parameter_key_or_group("PARAMETER_GROUP")
                assert not real_state.has_parameter_key_or_group("PARAMETER_GROUP2")
                assert real_state.get_parameter("PARAMETER_GROUP").timestamp == tob_1
            elif i % 3 == 0:
                assert not real_state.has_parameter_key_or_group("PARAMETER_GROUP")
                assert real_state.has_parameter_key_or_group("PARAMETER_GROUP2")
                assert real_state.get_parameter("PARAMETER_GROUP2").timestamp == tob_2
            else:
                assert not real_state.has_parameter_key_or_group("PARAMETER_GROUP2")
                assert not real_state.has_parameter_key_or_group("PARAMETER_GROUP")

        ensemble.unify_parameters()

        states = ensemble._realization_states
        state_df_after_combine = ensemble._realization_states.to_dataframe()

        assert state_df_before_combine["value"].equals(state_df_after_combine["value"])
        assert (
            sum(
                state_df_after_combine["timestamp"]
                - state_df_before_combine["timestamp"]
            )
            > 0
        )

        tob_1 = os.path.getctime(ensemble._path / "PARAMETER_GROUP.nc")
        tob_2 = os.path.getctime(ensemble._path / "PARAMETER_GROUP2.nc")
        for i in range(1, 25):
            real_state = states.get_single_realization_state(i)

            if i % 6 == 0:
                assert real_state.get_parameter("PARAMETER_GROUP").timestamp == tob_1
                assert real_state.get_parameter("PARAMETER_GROUP2").timestamp == tob_2
            elif i % 2 == 0:
                assert real_state.get_parameter("PARAMETER_GROUP").timestamp == tob_1

                assert not real_state.has_parameter_key_or_group("PARAMETER_GROUP2")
            elif i % 3 == 0:
                assert not real_state.has_parameter_key_or_group("PARAMETER_GROUP")
                assert real_state.get_parameter("PARAMETER_GROUP2").timestamp == tob_2
            else:
                assert not real_state.has_parameter_key_or_group("PARAMETER_GROUP2")
                assert not real_state.has_parameter_key_or_group("PARAMETER_GROUP")


def test_that_realization_states_update_after_rewrite_realization(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            parameters=[
                GenKwConfig(
                    name="PARAMETER_GROUP",
                    forward_init=False,
                    template_file="",
                    transform_function_definitions=[
                        TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
                        TransformFunctionDefinition("KEY2", "UNIFORM", [0, 1]),
                        TransformFunctionDefinition("KEY3", "UNIFORM", [0, 1]),
                    ],
                    output_file="kw.txt",
                    update=True,
                )
            ]
        )
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=25)
        for i in range(1, 25):
            ensemble.save_parameters(
                "PARAMETER_GROUP",
                i,
                pd.DataFrame(
                    data={
                        "names": ["KEY1", "KEY2", "KEY3"],
                        "values": [1, 2, 3],
                        "transformed_values": [2, 4, 6],
                    }
                )
                .set_index(["names"])
                .to_xarray(),
            )

        ensemble._refresh_realization_states()
        state_df_before_remove_param = ensemble._realization_states.to_dataframe()
        os.rename(
            ensemble._realization_dir(1) / "PARAMETER_GROUP.nc",
            ensemble._realization_dir(1) / "PARAMETER_GROUP_TMP.nc",
        )
        ensemble._refresh_realization_states()
        state_df_after_remove_param = ensemble._realization_states.to_dataframe()
        assert not state_df_after_remove_param.equals(state_df_before_remove_param)
        assert state_df_after_remove_param.drop(1).equals(
            state_df_before_remove_param.drop(1)
        )

        real_state1 = ensemble._realization_states.get_single_realization_state(1)
        assert not real_state1.has_parameter_key_or_group("PARAMETER_GROUP")

        os.rename(
            ensemble._realization_dir(1) / "PARAMETER_GROUP_TMP.nc",
            ensemble._realization_dir(1) / "PARAMETER_GROUP.nc",
        )
        ensemble._refresh_realization_states()
        real_state1 = ensemble._realization_states.get_single_realization_state(1)
        assert real_state1.has_parameter_key_or_group("PARAMETER_GROUP")
        assert real_state1.get_parameter(
            "PARAMETER_GROUP"
        ).timestamp == os.path.getctime(
            ensemble._realization_dir(1) / "PARAMETER_GROUP.nc"
        )

        ensemble.unify_parameters()
        real_state1 = ensemble._realization_states.get_single_realization_state(1)
        assert real_state1.has_parameter_key_or_group("PARAMETER_GROUP")
        assert real_state1.get_parameter(
            "PARAMETER_GROUP"
        ).timestamp == os.path.getctime(ensemble._path / "PARAMETER_GROUP.nc")


def test_that_realization_states_shows_all_responses_present(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[
                GenDataConfig(name="WOPR_OP1"),
                GenDataConfig(name="WOPR_OP2"),
                SummaryConfig(
                    name="summary", input_file=None, keys=["one", "two", "three"]
                ),
            ],
        )
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=25)
        for i in range(1, 25):
            if i % 2 == 0:
                ensemble.save_response(
                    "summary",
                    make_summary_data(
                        ["one", "two", "three"], ["2011-01-01", "2011-02-01"]
                    ),
                    i,
                )

            if i % 3 == 0:
                ensemble.save_response("WOPR_OP1", make_gen_data(20), i)

            if i % 5 == 0:
                ensemble.save_response("WOPR_OP2", make_gen_data(20), i)

        ensemble._refresh_realization_states()
        states = ensemble._realization_states
        for i in range(1, 25):
            real_state = states.get_single_realization_state(i)

            assert real_state.has_response_key_or_group("one") == (i % 2 == 0)
            assert real_state.has_response_key_or_group("two") == (i % 2 == 0)
            assert real_state.has_response_key_or_group("three") == (i % 2 == 0)

            if i % 2 == 0:
                assert real_state.get_response("one").timestamp == os.path.getctime(
                    ensemble._realization_dir(i) / "summary.nc"
                )
                assert real_state.get_response("two").timestamp == os.path.getctime(
                    ensemble._realization_dir(i) / "summary.nc"
                )
                assert real_state.get_response("three").timestamp == os.path.getctime(
                    ensemble._realization_dir(i) / "summary.nc"
                )

            assert real_state.has_response_key_or_group("WOPR_OP1") == (i % 3 == 0)
            if i % 3 == 0:
                assert real_state.get_response(
                    "WOPR_OP1"
                ).timestamp == os.path.getctime(
                    ensemble._realization_dir(i) / "WOPR_OP1.nc"
                )

            if i % 5 == 0:
                assert real_state.has_response_key_or_group("WOPR_OP2")
                assert real_state.get_response(
                    "WOPR_OP2"
                ).timestamp == os.path.getctime(
                    ensemble._realization_dir(i) / "WOPR_OP2.nc"
                )
            else:
                assert not real_state.has_response_key_or_group("WOPR_OP2")

        ensemble.unify_responses()
        states = ensemble._realization_states
        smry_tobs = os.path.getctime(ensemble._path / "summary.nc")
        gen_data_tobs = os.path.getctime(ensemble._path / "gen_data.nc")

        for i in range(1, 25):
            real_state = states.get_single_realization_state(i)

            assert real_state.has_response_key_or_group("one") == (i % 2 == 0)
            assert real_state.has_response_key_or_group("two") == (i % 2 == 0)
            assert real_state.has_response_key_or_group("three") == (i % 2 == 0)
            assert real_state.has_response_key_or_group("WOPR_OP1") == (i % 3 == 0)
            assert real_state.has_response_key_or_group("WOPR_OP2") == (i % 5 == 0)

            if real_state.has_response_key_or_group("summmary"):
                assert real_state.get_response("one").timestamp == smry_tobs
                assert real_state.get_response("two").timestamp == smry_tobs
                assert real_state.get_response("three").timestamp == smry_tobs
                assert real_state.get_response("summary").timestamp == smry_tobs

            if real_state.has_response_key_or_group("WOPR_OP1"):
                assert real_state.get_response("WOPR_OP1").timestamp == gen_data_tobs
                assert real_state.get_response("gen_data").timestamp == gen_data_tobs

            if real_state.has_response_key_or_group("WOPR_OP2"):
                assert real_state.get_response("WOPR_OP2").timestamp == gen_data_tobs
                assert real_state.get_response("gen_data").timestamp == gen_data_tobs


def test_single_realization_state_transfer_clear_responses():
    state_old = _SingleRealizationStateDict()
    state_old._items_by_kind = {
        "summary": {
            "A": _SingleRealizationStateDictEntry(value=True, timestamp=1),
            "B": _SingleRealizationStateDictEntry(value=False, timestamp=-1),
            "C": _SingleRealizationStateDictEntry(value=True, timestamp=1),
        }
    }

    assert state_old.has_response_key_or_group("A")
    assert state_old.get_response("A").timestamp == 1

    assert not state_old.has_response_key_or_group("B")
    assert state_old.get_response("B").timestamp == -1

    assert state_old.has_response_key_or_group("C")
    assert state_old.get_response("C").timestamp == 1

    state_emptying = _SingleRealizationStateDict()
    state_emptying._items_by_kind = {
        "summary": {
            "A": _SingleRealizationStateDictEntry(value=False, timestamp=8888),
            "B": _SingleRealizationStateDictEntry(value=True, timestamp=-2),
            "C": _SingleRealizationStateDictEntry(value=False, timestamp=9999),
        }
    }

    assert not state_emptying.has_response_key_or_group("A")
    assert state_emptying.get_response("A").timestamp == 8888

    assert state_emptying.has_response_key_or_group("B")
    assert state_emptying.get_response("B").timestamp == -2

    assert not state_emptying.has_response_key_or_group("C")
    assert state_emptying.get_response("C").timestamp == 9999

    state_with_no_responses = state_old.copy().assign_state(state_emptying)

    assert not state_with_no_responses.has_response_key_or_group("A")
    assert state_with_no_responses.get_response("A").timestamp == 8888

    # We expect the -1 timestamp to be kept as it is greater than the
    # -2 timestamp
    assert not state_with_no_responses.has_response_key_or_group("B")
    assert state_with_no_responses.get_response("B").timestamp == -1

    assert not state_with_no_responses.has_response_key_or_group("C")
    assert state_with_no_responses.get_response("C").timestamp == 9999


def test_single_realization_state_transfer_clear_all_but_one_response():
    state_old = _SingleRealizationStateDict()
    state_old._items_by_kind = {
        "summary": {
            "A": _SingleRealizationStateDictEntry(value=True, timestamp=999),
            "B": _SingleRealizationStateDictEntry(value=False, timestamp=-1),
            "C": _SingleRealizationStateDictEntry(value=True, timestamp=888),
        }
    }

    assert state_old.has_response_key_or_group("A")
    assert state_old.get_response("A").timestamp == 999

    assert not state_old.has_response_key_or_group("B")
    assert state_old.get_response("B").timestamp == -1

    assert state_old.has_response_key_or_group("C")
    assert state_old.get_response("C").timestamp == 888

    state_emptying = _SingleRealizationStateDict()
    state_emptying._items_by_kind = {
        "summary": {
            "A": _SingleRealizationStateDictEntry(value=False, timestamp=1010),
            "B": _SingleRealizationStateDictEntry(value=False, timestamp=-1),
            "C": _SingleRealizationStateDictEntry(value=True, timestamp=9999),
        }
    }

    assert not state_emptying.has_response_key_or_group("A")
    assert state_emptying.get_response("A").timestamp == 1010

    assert not state_emptying.has_response_key_or_group("B")
    assert state_emptying.get_response("B").timestamp == -1

    assert state_emptying.has_response_key_or_group("C")
    assert state_emptying.get_response("C").timestamp == 9999

    states_with_no_responses = state_old.copy().assign_state(state_emptying)

    assert not states_with_no_responses.has_response_key_or_group("A")
    assert states_with_no_responses.get_response("A").timestamp == 1010

    assert not states_with_no_responses.has_response_key_or_group("B")
    assert states_with_no_responses.get_response("B").timestamp == -1

    assert states_with_no_responses.has_response_key_or_group("C")
    assert states_with_no_responses.get_response("C").timestamp == 9999


def test_single_realization_state_transfer_from_state_without_any_responses():
    state_old = _SingleRealizationStateDict()
    state_old._items_by_kind = {
        "summary": {
            "A": _SingleRealizationStateDictEntry(value=True, timestamp=999),
            "B": _SingleRealizationStateDictEntry(value=False, timestamp=-1),
            "C": _SingleRealizationStateDictEntry(value=True, timestamp=888),
        }
    }

    assert state_old.has_response_key_or_group("A")
    assert state_old.get_response("A").timestamp == 999

    assert not state_old.has_response_key_or_group("B")
    assert state_old.get_response("B").timestamp == -1

    assert state_old.has_response_key_or_group("C")
    assert state_old.get_response("C").timestamp == 888

    state_emptying = _SingleRealizationStateDict()
    state_emptying._items_by_kind = {
        "summary": {
            "A": _SingleRealizationStateDictEntry(value=False, timestamp=1010),
            "B": _SingleRealizationStateDictEntry(value=False, timestamp=-1),
            "C": _SingleRealizationStateDictEntry(value=False, timestamp=9999),
        }
    }

    assert not state_emptying.has_response_key_or_group("A")
    assert state_emptying.get_response("A").timestamp == 1010

    assert not state_emptying.has_response_key_or_group("B")
    assert state_emptying.get_response("B").timestamp == -1

    assert not state_emptying.has_response_key_or_group("C")
    assert state_emptying.get_response("C").timestamp == 9999

    states_with_no_responses = state_old.copy().assign_state(state_emptying)

    assert not states_with_no_responses.has_response_key_or_group("A")
    assert states_with_no_responses.get_response("A").timestamp == 1010

    assert not states_with_no_responses.has_response_key_or_group("B")
    assert states_with_no_responses.get_response("B").timestamp == -1

    assert not states_with_no_responses.has_response_key_or_group("C")
    assert states_with_no_responses.get_response("C").timestamp == 9999


def test_single_realization_state_transfer_with_new_responses():
    state_old = _SingleRealizationStateDict()
    state_old._items_by_kind = {
        "summary": {
            "A": _SingleRealizationStateDictEntry(value=True, timestamp=999),
            "B": _SingleRealizationStateDictEntry(value=False, timestamp=-1),
            "C": _SingleRealizationStateDictEntry(value=True, timestamp=888),
        }
    }

    assert state_old.has_response_key_or_group("A")
    assert state_old.get_response("A").timestamp == 999

    assert not state_old.has_response_key_or_group("B")
    assert state_old.get_response("B").timestamp == -1

    assert state_old.has_response_key_or_group("C")
    assert state_old.get_response("C").timestamp == 888

    state_with_more = _SingleRealizationStateDict()
    state_with_more._items_by_kind = {
        "summary": {
            "AA": _SingleRealizationStateDictEntry(value=True, timestamp=1010),
            "BB": _SingleRealizationStateDictEntry(value=True, timestamp=-1),
            "CC": _SingleRealizationStateDictEntry(value=True, timestamp=9999),
        }
    }

    assert state_with_more.has_response_key_or_group("AA")
    assert state_with_more.get_response("AA").timestamp == 1010

    assert state_with_more.has_response_key_or_group("BB")
    assert state_with_more.get_response("BB").timestamp == -1

    assert state_with_more.has_response_key_or_group("CC")
    assert state_with_more.get_response("CC").timestamp == 9999

    states_with_all = state_old.copy().assign_state(state_with_more)

    assert states_with_all.has_response_key_or_group("A")
    assert states_with_all.get_response("A").timestamp == 999

    assert not states_with_all.has_response_key_or_group("B")
    assert states_with_all.get_response("B").timestamp == -1

    assert states_with_all.has_response_key_or_group("C")
    assert states_with_all.get_response("C").timestamp == 888

    assert states_with_all.has_response_key_or_group("AA")
    assert states_with_all.get_response("AA").timestamp == 1010

    assert states_with_all.has_response_key_or_group("BB")
    assert states_with_all.get_response("BB").timestamp == -1

    assert states_with_all.has_response_key_or_group("CC")
    assert states_with_all.get_response("CC").timestamp == 9999
