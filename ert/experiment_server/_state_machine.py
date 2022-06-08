from collections import defaultdict, Mapping
from email.policy import default
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def nested_dict_keys(tree_structure: Dict) -> set:
    result = set()
    for key, value in tree_structure.items():
        result.add(key)
        if isinstance(value, Mapping):
            # TODO: raise if key exist.
            result.update(nested_dict_keys(value))
        else:
            result.update(set(value))
    return result


class StateMachine:
    def __init__(self, ensemble_structure: Dict) -> None:
        self._ensemble_to_successful_realizations: Dict[int, List[int]] = defaultdict(
            list
        )
        self._realizations_at_ensemble: Dict[int, List[int]] = defaultdict(list)
        self._ensemble_at_iteration: Dict[int, int] = {}
        self._entity_states: Dict[int, Dict] = defaultdict(
            dict, {k: {} for k in nested_dict_keys(ensemble_structure)}
        )
        self._updated_entity_states: Dict[int, Dict] = defaultdict(dict)

        self._ensemble_structure = ensemble_structure

    def successful_realizations(self, iter_: int) -> int:
        """Return an integer indicating the number of successful realizations in an
        ensemble given iter_. Raise IndexError if the ensemble has no successful
        realizations."""
        return len(self._ensemble_to_successful_realizations[iter_])

    def add_successful_realization(self, iter_: int, real: int) -> None:
        logger.debug("adding successful real for iter %d, real: %d", iter_, real)
        if real not in self._ensemble_to_successful_realizations[iter_]:
            self._ensemble_to_successful_realizations[iter_].append(real)

    def add_event(self, event) -> None:
        # Will add the contents of the event to the updated_entity_states.
        # in the scenario that the contents of a new event overlaps with a previous
        # the old content will be overwritten, hence this will also merge redundant information
        pass

    def get_update(self) -> Dict:
        # returns the updated_entity_states, merge with entity_states and then empties the updated_entity_state
        pass

    def get_full_state(self) -> Dict:
        # Returns the state as it is described in `entity_states` (not including the content in `updated_entity_state`)
        pass
