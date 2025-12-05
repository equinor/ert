from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import xtgeo
from surfio import IrapHeader, IrapSurface

from ert.config import ConfigValidationError, SurfaceConfig
from ert.config.parameter_config import InvalidParameterFile


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
    config = SurfaceConfig(
        name="some_name",
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
        update=True,
    )
    ensemble = storage.create_experiment(parameters=[config]).create_ensemble(
        name="text", ensemble_size=1
    )
    surface.to_file(tmp_path / "input_0", fformat="irap_ascii")

    # run_path -> storage
    ds = config.read_from_runpath(tmp_path, 0, 0)
    ensemble.save_parameters(ds, config.name, 0)

    # storage -> run_path
    config.forward_init_file = "output_%d"
    config.write_to_runpath(tmp_path, 0, ensemble)

    # compare contents
    # Data is saved as 'irap_ascii', which means that we only keep 6 significant digits
    actual_surface = xtgeo.surface_from_file(
        tmp_path / "output", fformat="irap_ascii", dtype=np.float32
    )
    actual_surface_surfio = IrapSurface.from_ascii_file(tmp_path / "output")

    np.testing.assert_allclose(
        actual_surface.values, surface.values, rtol=0, atol=1e-06
    )
    np.testing.assert_allclose(
        actual_surface_surfio.values, surface.values, rtol=0, atol=1e-06
    )

    # Compare header, set all properties to different values to assert
    for prop, val in (
        ("ncol", surface.ncol),
        ("nrow", surface.nrow),
        ("xori", surface.xori),
        ("yori", surface.yori),
        ("xinc", surface.xinc),
        ("yinc", surface.yinc),
    ):
        assert (
            getattr(config, prop)
            == getattr(actual_surface_surfio.header, prop)
            == getattr(actual_surface, prop)
            == val
        ), f"Failed for: {prop}"

    assert actual_surface.yflip == config.yflip == surface.yflip
    assert actual_surface.rotation == config.rotation == surface.rotation

    assert actual_surface_surfio.header.xrot == surface.xori
    assert actual_surface_surfio.header.yrot == surface.yori
    assert (
        actual_surface_surfio.header.xmax
        == surface.xori + (surface.ncol - 1) * surface.xinc
    )
    assert (
        actual_surface_surfio.header.ymax
        == surface.yori + (surface.nrow - 1) * surface.yinc
    )


def test_init_files_must_contain_placeholder_when_not_forward_init():
    with pytest.raises(
        ConfigValidationError,
        match="INIT_FILES must contain %d or <IENS> when FORWARD_INIT:FALSE",
    ):
        SurfaceConfig.from_config_list(
            [
                "TOP",
                {
                    "INIT_FILES": "path/surf.irap",
                    "OUTPUT_FILE": "path/not_surface",
                    "BASE_SURFACE": "surface/small_out.irap",
                },
            ]
        )


def test_when_base_surface_does_not_exist_gives_config_error():
    with pytest.raises(
        ConfigValidationError,
        match="surface/small_out\\.irap not found",
    ):
        SurfaceConfig.from_config_list(
            [
                "TOP",
                {
                    "INIT_FILES": "path/%dsurf.irap",
                    "OUTPUT_FILE": "path/not_surface",
                    "BASE_SURFACE": "surface/small_out.irap",
                },
            ]
        )


def test_surface_without_output_file_gives_config_error():
    with pytest.raises(
        ConfigValidationError,
        match="Missing required OUTPUT_FILE",
    ):
        SurfaceConfig.from_config_list(["TOP", {}])


def test_surface_without_init_file_gives_config_error():
    with pytest.raises(
        ConfigValidationError,
        match="Missing required INIT_FILES",
    ):
        SurfaceConfig.from_config_list(["TOP", {}])


def test_surface_without_base_surface_gives_config_error():
    with pytest.raises(
        ConfigValidationError,
        match="Missing required BASE_SURFACE",
    ):
        SurfaceConfig.from_config_list(["TOP", {}])


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "forward_init_option, expected_forward_init",
    [({"FORWARD_INIT": "False"}, False), ({"FORWARD_INIT": "True"}, True), ({}, False)],
)
def test_config_file_line_sets_the_corresponding_properties(
    forward_init_option, expected_forward_init
):
    xtgeo.RegularSurface(
        ncol=2,
        nrow=3,
        xori=4.0,
        yori=5.0,
        xinc=6.0,
        yinc=7.0,
        rotation=8.0,
        yflip=-1,
        values=[1.0] * 6,
    ).to_file("base_surface.irap", fformat="irap_ascii")
    surface_config = SurfaceConfig.from_config_list(
        [
            "TOP",
            {
                "BASE_SURFACE": "base_surface.irap",
                "OUTPUT_FILE": "out.txt",
                "INIT_FILES": "%dsurf.irap",
                **forward_init_option,
            },
        ],
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
        ConfigValidationError, match="Could not load surface 'base_surface\\.irap'"
    ):
        _ = SurfaceConfig.from_config_list(
            [
                "TOP",
                {
                    "BASE_SURFACE": "base_surface.irap",
                    "OUTPUT_FILE": "out.txt",
                    "INIT_FILES": "%dsurf.irap",
                },
            ]
        )


@pytest.mark.parametrize(
    "shape,expected_nodes,expected_links",
    [
        ((0, 0), [], []),
        ((1, 0), [], []),
        ((0, 1), [], []),
        ((1, 1), [], []),
        ((1, 2), [{"id": 0}, {"id": 1}], [{"source": 0, "target": 1}]),
        (
            (10, 1),
            [{"id": i} for i in range(10)],
            [
                {"source": 0, "target": 1},
                {"source": 1, "target": 2},
                {"source": 2, "target": 3},
                {"source": 3, "target": 4},
                {"source": 4, "target": 5},
                {"source": 5, "target": 6},
                {"source": 6, "target": 7},
                {"source": 7, "target": 8},
                {"source": 8, "target": 9},
            ],
        ),
        (
            (3, 3),
            [
                {"id": 0},
                {"id": 1},
                {"id": 3},
                {"id": 2},
                {"id": 4},
                {"id": 5},
                {"id": 6},
                {"id": 7},
                {"id": 8},
            ],
            [
                {"source": 0, "target": 1},
                {"source": 0, "target": 3},
                {"source": 1, "target": 2},
                {"source": 1, "target": 4},
                {"source": 3, "target": 4},
                {"source": 3, "target": 6},
                {"source": 2, "target": 5},
                {"source": 4, "target": 5},
                {"source": 4, "target": 7},
                {"source": 5, "target": 8},
                {"source": 6, "target": 7},
                {"source": 7, "target": 8},
            ],
        ),
    ],
)
def test_surface_parameter_graph(shape, expected_nodes, expected_links):
    config = SurfaceConfig(
        name="surf",
        forward_init=False,
        update=True,
        ncol=shape[0],
        nrow=shape[1],
        xori=0,
        yori=0,
        xinc=0,
        yinc=0,
        rotation=0,
        yflip=0,
        forward_init_file="0",
        output_file=Path("0"),
        base_surface_path="0",
    )

    g = config.load_parameter_graph()
    data = nx.node_link_data(g, edges="links")
    assert data["nodes"] == expected_nodes
    assert data["links"] == expected_links


def test_surface_create_storage_datasets_raises_surface_mismatch_error_when_the_number_of_surface_parameters_is_different_than_base_surface_size():  # noqa
    surface_name = "foo_surface"
    realization = 13
    surface_parameters = 10
    base_surface_name = "base_surface.irap"
    base_surface_col = 3
    base_surface_row = 3
    config = SurfaceConfig(
        name=surface_name,
        forward_init=False,
        update=True,
        ncol=base_surface_col,
        nrow=base_surface_row,
        xori=0,
        yori=0,
        xinc=0,
        yinc=0,
        rotation=0,
        yflip=0,
        forward_init_file="0",
        output_file=Path("0"),
        base_surface_path=f"foo/bar/{base_surface_name}",
    )
    storage_dataset_iterator = config.create_storage_datasets(
        from_data=np.ndarray(
            [surface_parameters, 1],
        ),
        iens_active_index=np.array([realization]),
    )

    expected_error_msg = (
        rf"Saving parameters for SURFACE '{surface_name}' for realization "
        rf"{realization} to storage failed. "
        rf"SURFACE {surface_name} \(size {surface_parameters}\) and BASE_SURFACE "
        rf"{base_surface_name} \(size {base_surface_col * base_surface_row}\) "
        rf"must have the same size."
    )
    with pytest.raises(InvalidParameterFile, match=expected_error_msg):
        next(storage_dataset_iterator)


@pytest.fixture
def surface_for_dl():
    # Return a SurfaceConfig object for test purpose

    # Create a synthetic surface grid with rotation
    nx = 100
    ny = 120
    xori = 1000.0
    yori = 2000.0
    xsize = 500.0
    ysize = 600.0
    xinc = xsize / nx
    yinc = ysize / ny
    rotation = 0.0
    return SurfaceConfig(
        type="surface",
        name="MySurface",
        forward_init=True,
        update=True,
        ncol=nx,
        nrow=ny,
        xori=xori,
        yori=yori,
        xinc=xinc,
        yinc=yinc,
        rotation=rotation,
        yflip=1,
        forward_init_file="dummy.txt",
        output_file=Path("dummy.txt"),
        base_surface_path="dummy.txt",
    )


@pytest.mark.parametrize(
    "xpos, ypos, main_range, perp_range, anisotropy_angle",
    [
        (
            [50.0, 250.0, 250.0, 250.0, 0.0, 500.0, 0.0],  # xpos
            [50.0, 300.0, 300.0, 300.0, 0.0, 0.0, 600.0],  # ypos
            [100.0, 300.0, 300.0, 600.0, 1000.0, 300.0, 300.0],  # main_range
            [100.0, 100.0, 100.0, 200.0, 100.0, 10.0, 100.0],  # perp_range
            [0.0, 35.0, 135.0, -135.0, 45.0, -45.0, -60.0],  # angle
        ),
    ],
)
def test_calc_rho_for_2d_grid_layer(
    snapshot,
    surface_for_dl,
    xpos: list[float],
    ypos: list[float],
    main_range: list[float],
    perp_range: list[float],
    anisotropy_angle: list[float],
):
    write_surface_file = False

    xposition = np.array(xpos)
    yposition = np.array(ypos)
    mainrange = np.array(main_range)
    perprange = np.array(perp_range)
    angles = np.array(anisotropy_angle)

    #   Dimension of rho_for_one_grid_layer is (nx,ny,nobs)
    rho_for_one_grid_layer = surface_for_dl.calc_rho_for_2d_grid_layer(
        xposition,
        yposition,
        mainrange,
        perprange,
        angles,
    )
    # Ensure -0 and +0 will be 0
    rho_for_one_grid_layer = np.where(
        rho_for_one_grid_layer == 0, 0.0, rho_for_one_grid_layer
    )

    snapshot.assert_match(
        str(rho_for_one_grid_layer) + "\n", "testdata_rho_for_one_grid_layer.txt"
    )
    if write_surface_file:
        # Write surface of rho for visualization

        for obs_indx in range(len(xposition)):
            surf = IrapSurface(
                header=IrapHeader(
                    ncol=surface_for_dl.ncol,
                    nrow=surface_for_dl.nrow,
                    xori=surface_for_dl.xori,
                    yori=surface_for_dl.yori,
                    xinc=surface_for_dl.xinc,
                    yinc=surface_for_dl.yinc,
                    rot=surface_for_dl.rotation,
                    xrot=surface_for_dl.xori,
                    yrot=surface_for_dl.yori,
                ),
                values=rho_for_one_grid_layer[:, :, obs_indx],
            )

            file_path = Path("tmp_2d_rho_for_obs_" + str(obs_indx) + ".txt")
            print(f"Write file: {file_path}")
            surf.to_ascii_file(file_path)


@pytest.mark.parametrize(
    "utmx, utmy, expected_x, expected_y",
    [
        (
            [1100.0, 1300.0, 1500.0],
            [2000, 2100, 2500],
            [100.0, 300.0, 500.0],
            [0.0, 100.0, 500.0],
        ),
    ],
)
def test_transform_positions_to_local_field_coordinates(
    surface_for_dl, utmx, utmy, expected_x, expected_y
):
    """Test transformation of position from global to local coordinates."""
    tolerance = 1e-8
    xpos = np.array(utmx)
    ypos = np.array(utmy)

    x_transf, y_transf = surface_for_dl.transform_positions_to_local_field_coordinates(
        xpos, ypos
    )
    print(f"{x_transf=}")
    print(f"{y_transf=}")
    reference_x = np.array(expected_x)
    reference_y = np.array(expected_y)
    for i in range(len(x_transf)):
        assert abs(x_transf[i] - reference_x[i]) < tolerance, (
            f"Expected x_transf[i]={reference_x[i]}, got {x_transf[i]}"
        )
        assert abs(y_transf[i] - reference_y[i]) < tolerance, (
            f"Expected y_transf[i]={reference_y[i]}, got {y_transf[i]}"
        )


@pytest.mark.parametrize(
    "input_angle, expected_angle",
    [
        (0.0, 0.0),
    ],
)
def test_transform_localization_ellipse_angle_to_local_coordinates(
    surface_for_dl, input_angle, expected_angle
):
    tolerance = 1e-8
    output_angle = surface_for_dl.transform_local_ellipse_angle_to_local_coords(
        input_angle
    )
    assert abs(output_angle - expected_angle) < tolerance
