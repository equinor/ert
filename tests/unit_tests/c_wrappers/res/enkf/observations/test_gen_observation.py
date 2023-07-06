import numpy as np
import pytest

from ert._c_wrappers.enkf import EnkfObs, GenObservation


def test_create(tmp_path):
    with pytest.raises(ValueError):
        gen_obs = EnkfObs._create_gen_obs()

    with open(tmp_path / "obs1.txt", "w", encoding="utf-8") as f:
        f.write("10  5  12 6\n")

    with pytest.raises(ValueError):
        gen_obs = EnkfObs._create_gen_obs(
            scalar_value=(1, 2), obs_file=str(tmp_path / "obs1.txt")
        )

    with pytest.raises(TypeError):
        gen_obs = EnkfObs._create_gen_obs(scalar_value=1)

    with pytest.raises(IOError):
        gen_obs = EnkfObs._create_gen_obs(obs_file="does/not/exist")

    gen_obs = EnkfObs._create_gen_obs(
        obs_file=str(tmp_path / "obs1.txt"), data_index="10,20"
    )
    assert gen_obs == GenObservation(
        np.array([10.0, 12.0]),
        np.array([5.0, 6.0]),
        np.array([10, 20]),
        np.full(2, 1.0),
    )
