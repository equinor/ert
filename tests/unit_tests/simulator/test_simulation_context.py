from ert.enkf_main import EnKFMain
from ert.realization_state import RealizationState
from ert.simulator import SimulationContext
from tests.utils import wait_until


def test_simulation_context(setup_case, storage):
    ert_config = setup_case("batch_sim", "sleepy_time.ert")
    ert = EnKFMain(ert_config)

    size = 4
    even_mask = [True, False] * (size // 2)
    odd_mask = [False, True] * (size // 2)

    experiment_id = storage.create_experiment()
    even_half = storage.create_ensemble(
        experiment_id,
        name="even_half",
        ensemble_size=ert_config.model_config.num_realizations,
    )
    odd_half = storage.create_ensemble(
        experiment_id,
        name="odd_half",
        ensemble_size=ert_config.model_config.num_realizations,
    )

    case_data = [(geo_id, {}) for geo_id in range(size)]
    even_ctx = SimulationContext(ert, even_half, even_mask, 0, case_data)
    odd_ctx = SimulationContext(ert, odd_half, odd_mask, 0, case_data)

    for iens in range(size):
        if iens % 2 == 0:
            assert not even_ctx.isRealizationFinished(iens)
        else:
            assert not odd_ctx.isRealizationFinished(iens)

    wait_until(lambda: not even_ctx.isRunning() and not odd_ctx.isRunning(), timeout=90)

    for iens in range(size):
        if iens % 2 == 0:
            assert even_ctx.get_run_args(iens).runpath.endswith(
                f"runpath/realization-{iens}-{iens}/iter-0"
            )
        else:
            assert odd_ctx.get_run_args(iens).runpath.endswith(
                f"runpath/realization-{iens}-{iens}/iter-0"
            )

    assert even_ctx.getNumFailed() == 0
    assert even_ctx.getNumRunning() == 0
    assert even_ctx.getNumSuccess() == size / 2

    assert odd_ctx.getNumFailed() == 0
    assert odd_ctx.getNumRunning() == 0
    assert odd_ctx.getNumSuccess() == size / 2

    for iens in range(size):
        if iens % 2 == 0:
            assert even_ctx.didRealizationSucceed(iens)
            assert not even_ctx.didRealizationFail(iens)
            assert even_ctx.isRealizationFinished(iens)

            assert even_half.state_map[iens] == RealizationState.HAS_DATA
        else:
            assert odd_ctx.didRealizationSucceed(iens)
            assert not odd_ctx.didRealizationFail(iens)
            assert odd_ctx.isRealizationFinished(iens)

            assert odd_half.state_map[iens] == RealizationState.HAS_DATA
