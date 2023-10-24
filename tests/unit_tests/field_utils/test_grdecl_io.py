from string import ascii_letters

import hypothesis.strategies as st
import numpy as np
import pytest
import resfo
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import array_shapes, arrays
from numpy.testing import assert_allclose

from ert.field_utils.grdecl_io import export_grdecl, import_bgrdecl, import_grdecl


def test_that_importing_mess_from_bgrdecl_raises_field_io_error(tmp_path):
    resfo.write(tmp_path / "test.bgrdecl", [("FOPR    ", resfo.MESS)])
    with pytest.raises(ValueError, match="FOPR in .* has MESS type"):
        import_bgrdecl(tmp_path / "test.bgrdecl", "FOPR    ", (1, 1, 1))


def test_that_importing_discrete_from_bgrdecl_raises_field_io_error(tmp_path):
    resfo.write(
        tmp_path / "test.bgrdecl", [("FOPR    ", np.array([1, 2, 3], dtype=np.int32))]
    )
    with pytest.raises(
        ValueError,
        match=r"Ert does not support discrete .*\."
        r" Attempted to import integer typed field FOPR in .*test\.bgrdecl",
    ):
        import_bgrdecl(tmp_path / "test.bgrdecl", "FOPR    ", (1, 1, 1))


def test_that_importing_names_longer_than_eight_characters_clamps(tmp_path):
    export_grdecl(
        np.ma.MaskedArray([[[0.0]]]),
        tmp_path / "test.grdecl",
        "MORETHANEIGHT",
        binary=False,
    )
    assert import_grdecl(tmp_path / "test.grdecl", "MORETHAN", (1, 1, 1)).size == 1


def test_that_importing_unterminated_grdecl_fails(tmp_path):
    (tmp_path / "test.grdecl").write_text("KEYWORD\n 1 2 3")
    with pytest.raises(ValueError, match="Reached end of stream while reading KEYWORD"):
        _ = import_grdecl(tmp_path / "test.grdecl", "KEYWORD", (1, 1, 1))


def test_that_importing_missing_keyword_in_bgrdecl_fails(tmp_path):
    resfo.write(tmp_path / "test.bgrdecl", [("NOTTHIS ", resfo.MESS)])
    with pytest.raises(ValueError, match="Did not find field parameter FOPR"):
        import_bgrdecl(tmp_path / "test.bgrdecl", "FOPR    ", (1, 1, 1))


def test_that_import_picks_just_the_field_with_given_name(tmp_path):
    (tmp_path / "test.grdecl").write_text("KEYWORD1\n 1.0 /\nKEYWORD2\n 2.0 /")
    assert (
        import_grdecl(tmp_path / "test.grdecl", "KEYWORD2", (1, 1, 1))[0, 0, 0] == 2.0
    )


def test_that_importing_missing_keyword_in_grdecl_fails(tmp_path):
    export_grdecl(
        np.ma.MaskedArray([[[0.0]]]), tmp_path / "test.grdecl", "NOTTHIS ", binary=False
    )
    with pytest.raises(ValueError, match="Did not find field parameter FOPR"):
        import_grdecl(tmp_path / "test.grdecl", "FOPR    ", (1, 1, 1))


@given(
    array=arrays(np.float32, shape=array_shapes(min_dims=3, max_dims=3)),
    name=st.text(ascii_letters, min_size=8, max_size=8),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_that_binary_export_and_import_are_inverses(array, name, tmp_path):
    masked_array = np.ma.masked_invalid(array)
    export_grdecl(masked_array, tmp_path / "test.bgrdecl", name, binary=True)
    assert (
        (
            import_bgrdecl(tmp_path / "test.bgrdecl", name, dimensions=array.shape)
            == masked_array
        )
        .filled(True)
        .all()
    )


@given(
    array=arrays(np.float32, shape=array_shapes(min_dims=3, max_dims=3)),
    name=st.text(ascii_letters, min_size=8, max_size=8),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_that_text_export_and_import_are_inverses(array, name, tmp_path):
    masked_array = np.ma.masked_invalid(array)
    export_grdecl(masked_array, tmp_path / "test.bgrdecl", name, binary=False)
    assert_allclose(
        import_grdecl(tmp_path / "test.bgrdecl", name, dimensions=array.shape),
        masked_array,
        rtol=1e-6,
        atol=1e-6,
    )
