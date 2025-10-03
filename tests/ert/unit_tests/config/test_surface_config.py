from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import xtgeo
from surfio import IrapSurface

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
    data = nx.node_link_data(g)
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
