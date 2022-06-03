from ._base import BaseStorage, Namespace
from ._enkf_fs import EnkfFs, EnkfFsMt
from ._ert_storage import ErtStorage
from ._pandas import PdHdf5, PdHdf5Open
from ._sqlite import Sqlite
from ._xarray import XArrayNetCDF
from typing import List, Mapping, Type


MODULES: Mapping[str, Type[BaseStorage]] = {
    klass.__name__: klass
    for klass in (
        EnkfFs,
        EnkfFsMt,
        ErtStorage,
        PdHdf5,
        PdHdf5Open,
        Sqlite,
        XArrayNetCDF,
    )
}
TESTS: List[str] = list(BaseStorage.__abstractmethods__) + ["validate_parameter", "validate_response"]

__all__ = ["MODULES", "TESTS", "Namespace"]
