import numpy as np
import pytest

from ert.field_utils import save_field


@pytest.mark.usefixtures("use_tmpdir")
def test_save_grdecl(benchmark):
    rng = np.random.default_rng(42)
    values = rng.standard_normal(
        dtype=np.float32,
        size=(100, 100, 100),
    )
    field = np.ma.masked_array(values)

    def run():
        save_field(field, "FOPR", "test.grdecl", "grdecl")

    benchmark(run)
