from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from typing import Callable

    import xarray as xr

    from ert.storage import EnsembleReader, ExperimentReader


NotifierType = TypedDict(
    "NotifierType",
    {
        "experiment:create": "Callable[[ExperimentReader], None]",
        "ensemble:create": "Callable[[EnsembleReader], None]",
        "parameters:create": "Callable[[EnsembleReader, str, int, xr.Dataset], None]",
        "responses:create": "Callable[[EnsembleReader, str, int, xr.Dataset], None]",
    },
)


dummy_notifier: NotifierType = {
    "experiment:create": lambda _: None,
    "ensemble:create": lambda _: None,
    "parameters:create": lambda _, _name, _real, _ds: None,
    "responses:create": lambda _, _name, _real, _ds: None,
}
