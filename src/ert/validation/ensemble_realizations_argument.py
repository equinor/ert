from typing import TYPE_CHECKING

from .range_string_argument import RangeStringArgument
from .rangestring import rangestring_to_list
from .validation_status import ValidationStatus

if TYPE_CHECKING:
    from ert.storage import Ensemble


class EnsembleRealizationsArgument(RangeStringArgument):
    UNINITIALIZED_REALIZATIONS_SPECIFIED = (
        "The specified realization(s) %s are not found in selected ensemble."
    )

    def __init__(
        self, ensemble: "Ensemble", max_value: int | None, **kwargs: bool
    ) -> None:
        super().__init__(max_value, **kwargs)
        self.__ensemble = ensemble

    def set_ensemble(self, ensemble: "Ensemble") -> None:
        self.__ensemble = ensemble

    def validate(self, token: str) -> ValidationStatus:
        if not token:
            return ValidationStatus()

        validation_status = super().validate(token)
        if not validation_status:
            return validation_status
        attempted_realizations = rangestring_to_list(token)

        invalid_realizations = []
        initialized_realization_ids = self.__ensemble.is_initalized()
        for realization in attempted_realizations:
            if realization not in initialized_realization_ids:
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
