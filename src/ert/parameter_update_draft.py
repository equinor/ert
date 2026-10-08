from collections.abc import Iterable, Mapping

from ert.config import LocalizationType, ParameterConfig
from ert.config.parameter_config import apply_parameter_update_overrides


class ParameterUpdateDraft:
    """Localization edits against configured parameters or a selected prior.

    Configured overrides survive source switches; prior overrides are discarded
    when its identity or parameter snapshot changes, or when leaving the prior.
    Mutation methods return whether state changed. Returned parameters and
    overrides are independent copies.
    """

    def __init__(self, parameters: Iterable[ParameterConfig]) -> None:
        self._configured_parameters = [
            parameter.model_copy(deep=True) for parameter in parameters
        ]
        self._configured_overrides: dict[str, LocalizationType] = {}
        self._prior_parameters: list[ParameterConfig] = []
        self._prior_overrides: dict[str, LocalizationType] = {}
        self._prior_selected = False
        self._prior_id: str | None = None

    @property
    def has_parameter_source(self) -> bool:
        return not self._prior_selected or self._prior_id is not None

    @property
    def prior_id(self) -> str | None:
        return self._prior_id

    @property
    def parameters(self) -> list[ParameterConfig]:
        return apply_parameter_update_overrides(
            self._active_source_parameters, self.overrides
        )

    @property
    def overrides(self) -> dict[str, LocalizationType]:
        return dict(self._active_overrides)

    @property
    def has_changes(self) -> bool:
        return bool(self._configured_overrides or self._prior_overrides)

    def select_prior(
        self,
        prior_id: str | None,
        parameters: Iterable[ParameterConfig],
    ) -> bool:
        """Select a prior snapshot; a missing ID represents an unavailable source."""
        prior_parameters = list(parameters) if prior_id is not None else []
        if (
            self._prior_selected
            and prior_id == self._prior_id
            and prior_parameters == self._prior_parameters
        ):
            return False
        self._prior_selected = True
        self._prior_id = prior_id
        self._prior_parameters = [
            parameter.model_copy(deep=True) for parameter in prior_parameters
        ]
        self._prior_overrides = {}
        return True

    def use_configured_parameters(self) -> bool:
        if not self._prior_selected:
            return False
        self._prior_selected = False
        self._prior_id = None
        self._prior_parameters = []
        self._prior_overrides = {}
        return True

    def apply_strategies_by_type(
        self, strategies: Mapping[str, LocalizationType]
    ) -> bool:
        overrides = self.overrides
        for parameter in self._active_source_parameters:
            if parameter.update_strategy is None or parameter.type not in strategies:
                continue
            strategy = strategies[parameter.type]
            if strategy == parameter.update_strategy:
                overrides.pop(parameter.name, None)
            else:
                overrides[parameter.name] = strategy
        apply_parameter_update_overrides(self._active_source_parameters, overrides)
        if overrides == self.overrides:
            return False
        if self._prior_selected:
            self._prior_overrides = overrides
        else:
            self._configured_overrides = overrides
        return True

    def reset_all_overrides(self) -> bool:
        if not self.has_changes:
            return False
        self._configured_overrides.clear()
        self._prior_overrides.clear()
        return True

    @property
    def _active_source_parameters(self) -> list[ParameterConfig]:
        return (
            self._prior_parameters
            if self._prior_selected
            else self._configured_parameters
        )

    @property
    def _active_overrides(self) -> dict[str, LocalizationType]:
        return (
            self._prior_overrides
            if self._prior_selected
            else self._configured_overrides
        )
