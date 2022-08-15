from ...libres_utils import wait_until

from ert._c_wrappers.enkf import EnKFMain
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert.simulator import SimulationContext


def test_simulation_context(setup_case):
    res_config = setup_case("local/batch_sim", "sleepy_time.ert")
    ert = EnKFMain(res_config)

    size = 4
    even_mask = [True, False] * (size // 2)
    odd_mask = [False, True] * (size // 2)

    fs_manager = ert.getEnkfFsManager()
    even_half = fs_manager.getFileSystem("even_half")
    odd_half = fs_manager.getFileSystem("odd_half")

    case_data = [(geo_id, {}) for geo_id in range(size)]
    even_ctx = SimulationContext(ert, even_half, even_mask, 0, case_data)
    odd_ctx = SimulationContext(ert, odd_half, odd_mask, 0, case_data)

    for iens in range(size):
        if iens % 2 == 0:
            assert not even_ctx.isRealizationFinished(iens)
        else:
            assert not odd_ctx.isRealizationFinished(iens)

    def all_stopped():
        assert not even_ctx.isRunning() and not odd_ctx.isRunning()

    wait_until(all_stopped, timeout=90)

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

    even_state_map = even_half.getStateMap()
    odd_state_map = odd_half.getStateMap()

    for iens in range(size):
        if iens % 2 == 0:
            assert even_ctx.didRealizationSucceed(iens)
            assert not even_ctx.didRealizationFail(iens)
            assert even_ctx.isRealizationFinished(iens)

            assert even_state_map[iens] == RealizationStateEnum.STATE_HAS_DATA
        else:
            assert odd_ctx.didRealizationSucceed(iens)
            assert not odd_ctx.didRealizationFail(iens)
            assert odd_ctx.isRealizationFinished(iens)

            assert odd_state_map[iens] == RealizationStateEnum.STATE_HAS_DATA
