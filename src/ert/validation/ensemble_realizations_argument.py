from collections.abc import Iterable
from typing import TYPE_CHECKING

from .range_string_argument import RangeStringArgument
from .rangestring import rangestring_to_list
from .validation_status import ValidationStatus

if TYPE_CHECKING:
    from ert.storage import Ensemble
    from ert.storage.realization_storage_state import RealizationStorageState


class EnsembleRealizationsArgument(RangeStringArgument):
    UNINITIALIZED_REALIZATIONS_SPECIFIED = (
        "The specified realization(s) %s are not found in selected ensemble."
    )

    def __init__(
        self,
        ensemble: "Ensemble",
        max_value: int | None,
        required_realization_storage_states: Iterable["RealizationStorageState"],
        **kwargs: bool,
    ) -> None:
        super().__init__(max_value, **kwargs)
        self.__ensemble = ensemble
        self._required_realization_storage_states: Iterable[RealizationStorageState] = (
            required_realization_storage_states
        )

    def set_ensemble(self, ensemble: "Ensemble") -> None:
        self.__ensemble = ensemble

    def validate(self, token: str) -> ValidationStatus:
        validation_status = super().validate(token)
        if not validation_status:
            return validation_status
        attempted_realizations = rangestring_to_list(token)

        invalid_realizations = []
        found_realization_ids = [
            index
            for index, state in enumerate(self.__ensemble.get_ensemble_state())
            if set(self._required_realization_storage_states).issubset(state)
        ]
        for realization in attempted_realizations:
            if realization not in found_realization_ids:
                invalid_realizations.append(realization)

        if invalid_realizations:
            validation_status.setFailed()
            validation_status.addToMessage(
                EnsembleRealizationsArgument.UNINITIALIZED_REALIZATIONS_SPECIFIED
                % str(invalid_realizations)
            )
            return validation_status

        validation_status.setValue(token)
        return validation_status
