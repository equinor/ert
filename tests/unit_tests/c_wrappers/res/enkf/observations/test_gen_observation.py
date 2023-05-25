import pytest

from ert._c_wrappers.enkf import ActiveList, GenObservation


def test_create(tmp_path):
    with pytest.raises(ValueError):
        gen_obs = GenObservation()

    with open(tmp_path / "obs1.txt", "w", encoding="utf-8") as f:
        f.write("10  5  12 6\n")

    with pytest.raises(ValueError):
        gen_obs = GenObservation(
            scalar_value=(1, 2), obs_file=str(tmp_path / "obs1.txt")
        )

    with pytest.raises(TypeError):
        gen_obs = GenObservation(scalar_value=1)

    with pytest.raises(IOError):
        gen_obs = GenObservation(obs_file="does/not/exist")

    gen_obs = GenObservation(obs_file=str(tmp_path / "obs1.txt"), data_index="10,20")
    assert len(gen_obs) == 2
    assert gen_obs[0] == (10, 5)
    assert gen_obs[1] == (12, 6)

    assert gen_obs.getValue(0) == 10
    assert gen_obs.getDataIndex(1) == 20
    assert gen_obs.getStdScaling(0) == 1
    assert gen_obs.getStdScaling(1) == 1

    active_list = ActiveList()
    gen_obs.updateStdScaling(0.25, active_list)
    assert gen_obs.getStdScaling(0) == 0.25
    assert gen_obs.getStdScaling(1) == 0.25

    active_list.addActiveIndex(1)
    gen_obs.updateStdScaling(2.00, active_list)
    assert gen_obs.getStdScaling(0) == 0.25
    assert gen_obs.getStdScaling(1) == 2.00
