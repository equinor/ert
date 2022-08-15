from ert._c_wrappers.enkf import EnKFMain
from ert._c_wrappers.enkf.enums.realization_state_enum import RealizationStateEnum


def test_load_results_manually(setup_case):
    res_config = setup_case("local/mini_ert", "mini_fail_config")
    ert = EnKFMain(res_config)
    load_into_case = "A1"
    load_from_case = "default_1"

    load_into = ert.getEnkfFsManager().getFileSystem(load_into_case)
    load_from = ert.getEnkfFsManager().getFileSystem(load_from_case)

    ert.getEnkfFsManager().switchFileSystem(load_from)
    realisations = [True] * 10
    realisations[7] = False
    iteration = 0

    loaded = ert.loadFromForwardModel(realisations, iteration, load_into)

    load_into_case_state_map = load_into.getStateMap()

    load_into_states = [state for state in load_into_case_state_map]

    expected = [
        RealizationStateEnum.STATE_HAS_DATA,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_UNDEFINED,
        RealizationStateEnum.STATE_HAS_DATA,
        RealizationStateEnum.STATE_HAS_DATA,
    ]

    assert load_into_states == expected
    assert loaded == 3


def test_load_results_from_run_context(setup_case):
    res_config = setup_case("local/mini_ert", "mini_fail_config")
    ert = EnKFMain(res_config)
    load_into_case = "A1"
    load_from_case = "default_0"

    load_into = ert.getEnkfFsManager().getFileSystem(load_into_case)
    load_from = ert.getEnkfFsManager().getFileSystem(load_from_case)

    ert.getEnkfFsManager().switchFileSystem(load_from)
    realisations = [True] * 10
    realisations[7] = False

    run_context = ert.create_ensemble_experiment_run_context(
        source_filesystem=load_into, active_mask=realisations, iteration=0
    )

    loaded = ert.loadFromRunContext(run_context, load_into)

    load_into_case_state_map = load_into.getStateMap()
    load_into_states = [state for state in load_into_case_state_map]

    expected = [
        RealizationStateEnum.STATE_HAS_DATA,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_UNDEFINED,
        RealizationStateEnum.STATE_HAS_DATA,
        RealizationStateEnum.STATE_HAS_DATA,
    ]

    assert load_into_states == expected
    assert loaded == 3
