import numpy as np
import pytest
import xtgeo

from ert.storage import field_utils


@pytest.mark.parametrize(
    "shape, mask",
    [
        (
            (2, 3, 4),
            np.random.choice(a=[False, True], size=(2, 3, 4), p=[0.3, 0.7]),
        ),
        (
            (4, 3, 2),
            np.random.choice(a=[False, True], size=(4, 3, 2), p=[0.3, 0.7]),
        ),
        (
            (3, 2, 4),
            np.random.choice(a=[False, True], size=(3, 2, 4), p=[0.3, 0.7]),
        ),
        (
            (20, 33, 44),
            np.random.choice(a=[False, True], size=(20, 33, 44), p=[0.3, 0.7]),
        ),
    ],
)
def test_read_bgrdecl(tmpdir, shape, mask):
    with tmpdir.as_cwd():
        field_path = "test.bgrdecl"
        grid_path = "test.egrid"
        field_name = "MY_PARAM"
        values = np.random.standard_normal(mask.shape)

        grid = xtgeo.create_box_grid(dimension=shape)
        grid_mask = grid.get_actnum()
        grid_mask.values = [int(not num) for num in mask.flatten()]
        grid.set_actnum(grid_mask)
        grid.to_file(grid_path, "egrid")

        prop = xtgeo.GridProperty(
            ncol=shape[0],
            nrow=shape[1],
            nlay=shape[2],
            name=field_name,
            grid=grid,
            values=values.flatten(),
        )
        prop.to_file(field_path, fformat="bgrdecl")

        data = np.ma.MaskedArray(values, mask, fill_value=np.nan)
        masked_field = field_utils.get_masked_field(field_path, field_name, grid_path)
        assert np.ma.allclose(masked_field, data, masked_equal=True)
