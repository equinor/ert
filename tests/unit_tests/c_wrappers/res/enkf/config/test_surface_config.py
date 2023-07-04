import numpy as np
import pytest
import xtgeo

from ert._c_wrappers.enkf.config.surface_config import SurfaceConfig


@pytest.fixture
def surface():
    rng = np.random.default_rng()
    nrow = 3
    ncol = 5
    data = rng.standard_normal(size=(nrow, ncol))
    return xtgeo.RegularSurface(
        ncol=ncol,
        nrow=nrow,
        xinc=1.0,
        yinc=2.0,
        xori=3.0,
        yori=4.0,
        rotation=10,
        values=data,
    )


def test_runpath_roundtrip(tmp_path, storage, surface):
    ensemble = storage.create_experiment().create_ensemble(name="text", ensemble_size=1)
    config = SurfaceConfig(
        "some_name",
        forward_init=True,
        ncol=surface.ncol,
        nrow=surface.nrow,
        xori=surface.xori,
        yori=surface.yori,
        xinc=surface.xinc,
        yinc=surface.yinc,
        rotation=surface.rotation,
        yflip=surface.yflip,
        forward_init_file="input_%d",
        output_file=tmp_path / "output",
        base_surface_path="base_surface",
    )

    surface.to_file(tmp_path / "input_0", fformat="irap_ascii")

    # run_path -> storage
    ds = config.read_from_runpath(tmp_path, 0)
    ensemble.save_parameters(config.name, 0, ds)

    # storage -> run_path
    config.forward_init_file = "output_%d"
    config.write_to_runpath(tmp_path, 0, ensemble)

    # compare contents
    # Data is saved as 'irap_ascii', which means that we only keep 6 significant digits
    actual_surface = xtgeo.surface_from_file(tmp_path / "output", fformat="irap_ascii")
    np.testing.assert_allclose(
        actual_surface.values, surface.values, rtol=0, atol=1e-06
    )

    # Compare header, set all properties to different values to assert
    for prop, val in (
        ("ncol", 5),
        ("nrow", 3),
        ("xori", 3),
        ("yori", 4),
        ("xinc", 1),
        ("yinc", 2),
        ("yflip", 1.0),
        ("rotation", 10),
    ):
        assert (
            getattr(config, prop) == getattr(actual_surface, prop) == val
        ), f"Failed for: {prop}"
