from pathlib import Path

import numpy as np
import pytest
import xtgeo

from ert.config import ConfigValidationError, SurfaceConfig


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


def test_init_files_must_contain_placeholder_when_not_forward_init():
    with pytest.raises(
        ConfigValidationError,
        match="INIT_FILES must contain %d when FORWARD_INIT:FALSE",
    ):
        SurfaceConfig.from_config_list(
            [
                "TOP",
                "INIT_FILES:path/surf.irap",
                "OUTPUT_FILE:path/not_surface",
                "BASE_SURFACE:surface/small_out.irap",
            ]
        )


def test_when_base_surface_does_not_exist_gives_config_error():
    with pytest.raises(
        ConfigValidationError,
        match="surface/small_out.irap not found",
    ):
        SurfaceConfig.from_config_list(
            [
                "TOP",
                "INIT_FILES:path/%dsurf.irap",
                "OUTPUT_FILE:path/not_surface",
                "BASE_SURFACE:surface/small_out.irap",
            ]
        )


def test_surface_without_output_file_gives_config_error():
    with pytest.raises(
        ConfigValidationError,
        match="Missing required OUTPUT_FILE",
    ):
        SurfaceConfig.from_config_list(["TOP"])


def test_surface_without_init_file_gives_config_error():
    with pytest.raises(
        ConfigValidationError,
        match="Missing required INIT_FILES",
    ):
        SurfaceConfig.from_config_list(["TOP"])


def test_surface_without_base_surface_gives_config_error():
    with pytest.raises(
        ConfigValidationError,
        match="Missing required BASE_SURFACE",
    ):
        SurfaceConfig.from_config_list(["TOP"])


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "forward_init_option, expected_forward_init",
    [(["FORWARD_INIT:False"], False), (["FORWARD_INIT:True"], True), ([], False)],
)
def test_config_file_line_sets_the_corresponding_properties(
    forward_init_option, expected_forward_init
):
    xtgeo.RegularSurface(
        ncol=2, nrow=3, xori=4.0, yori=5.0, xinc=6.0, yinc=7.0, rotation=8.0, yflip=-1
    ).to_file("base_surface.irap", fformat="irap_ascii")
    surface_config = SurfaceConfig.from_config_list(
        [
            "TOP",
            "BASE_SURFACE:base_surface.irap",
            "OUTPUT_FILE:out.txt",
            "INIT_FILES:%dsurf.irap",
        ]
        + forward_init_option,
    )

    assert surface_config.ncol == 2
    assert surface_config.nrow == 3
    assert surface_config.xori == pytest.approx(4.0)
    assert surface_config.yori == pytest.approx(5.0)
    assert surface_config.xinc == pytest.approx(6.0)
    assert surface_config.yinc == pytest.approx(7.0)
    assert surface_config.rotation == pytest.approx(8.0)
    assert surface_config.yflip == -1
    assert surface_config.name == "TOP"
    assert surface_config.forward_init_file == "%dsurf.irap"
    assert surface_config.forward_init == expected_forward_init


@pytest.mark.usefixtures("use_tmpdir")
def test_invalid_surface_files_gives_config_error():
    Path("base_surface.irap").write_text("not valid irap", encoding="utf-8")
    with pytest.raises(
        ConfigValidationError, match="Could not load surface 'base_surface.irap'"
    ):
        _ = SurfaceConfig.from_config_list(
            [
                "TOP",
                "BASE_SURFACE:base_surface.irap",
                "OUTPUT_FILE:out.txt",
                "INIT_FILES:%dsurf.irap",
            ]
        )
