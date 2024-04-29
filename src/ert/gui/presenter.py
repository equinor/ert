from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from annotated_types import Ge, Le

from ert.gui.ertnotifier import ErtNotifier
from ert.libres_facade import LibresFacade

if TYPE_CHECKING:
    from ert.storage import EnsembleReader


class _UpdateMethod(Enum):
    IES = auto()
    ES = auto()


class Presenter:
    def __init__(self, libres_facade: LibresFacade, notifier: ErtNotifier):
        self.libres_facade = libres_facade
        self.notifier = notifier
        self._selected_update_method = None

    @property
    def current_ensemble_changed(self):
        return self.notifier.current_case_changed

    @property
    def ert_changed(self):
        return self.notifier.ertChanged

    @property
    def storage_changed(self):
        return self.notifier.storage_changed

    @property
    def is_storage_available(self):
        return self.notifier.is_storage_available

    @property
    def num_iterations(self):
        return self.libres_facade.config.analysis_config.num_iterations

    @num_iterations.setter
    def num_iterations(self, value):
        if self.num_iterations != value:
            self.libres_facade.config.analysis_config.set_num_iterations(value)
            self.notifier.emitErtChange()

    @property
    def current_case(self):
        return self.notifier.current_case

    @current_case.setter
    def current_case(self, value):
        self.notifier.set_current_case(value)

    def ensembles(self, initialized: bool = True) -> Iterable[EnsembleReader]:
        if initialized:
            return [e for e in self.notifier.storage.ensembles if e.is_initalized]
        return self.notifier.storage.ensembles

    def ies_selected(self) -> bool:
        return self._selected_update_method == _UpdateMethod.IES

    def es_selected(self) -> bool:
        return self._selected_update_method == _UpdateMethod.ES

    def select_ies(self) -> None:
        self._selected_update_method = _UpdateMethod.IES

    def select_es(self) -> None:
        self._selected_update_method = _UpdateMethod.ES

    def _analysis_module(self):
        if self._selected_update_method == _UpdateMethod.IES:
            return self.libres_facade.config.analysis_config.ies_module
        elif self._selected_update_method == _UpdateMethod.ES:
            return self.libres_facade.config.analysis_config.es_module
        else:
            raise ValueError("No update method selected")

    def current_analysis_module_description(self):
        return self._analysis_module().__doc__

    def analysis_range_variables(self) -> Iterator[Any]:
        analysis_module = self._analysis_module()
        if self.ies_selected:
            for variable_name in (
                name
                for name in self._analysis_module().model_fields
                if "steplength" in name
            ):
                metadata = analysis_module.model_fields[variable_name]
                yield (
                    variable_name,
                    metadata.title,
                    analysis_module.__getattribute__(variable_name),
                    [val for val in metadata.metadata if isinstance(val, Ge)][0].ge,
                    [val for val in metadata.metadata if isinstance(val, Le)][0].le,
                )

        metadata = analysis_module.model_fields["enkf_truncation"]
        yield (
            "enkf_truncation",
            "Singular value truncation",
            analysis_module.enkf_truncation,
            [val for val in metadata.metadata if isinstance(val, Ge)][0].ge,
            [val for val in metadata.metadata if isinstance(val, Le)][0].le,
        )

    def inversion_description(self):
        return self._analysis_module().model_fields["inversion"].description
