import dataclasses
import os
import pathlib
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas
from typing_extensions import Self


@dataclasses.dataclass
class _SingleRealizationStateDictEntry:
    value: bool = dataclasses.field(default=False)
    timestamp: float = dataclasses.field(default=-1)

    def update(self, value: bool, timestamp: float = -1) -> None:
        if timestamp is None:
            timestamp = time.time()

        self.value = value
        self.timestamp = timestamp

    def copy(self) -> "_SingleRealizationStateDictEntry":
        return _SingleRealizationStateDictEntry(
            value=self.value, timestamp=self.timestamp
        )

    def assign_state(self, src_state: "_SingleRealizationStateDictEntry") -> Self:
        if src_state.timestamp == -1 and self.timestamp != -1:
            return self

        if src_state.timestamp == -1 and self.timestamp == -1:
            # TODO branch may not be needed
            return self

        if src_state.timestamp > self.timestamp:
            self.value = src_state.value
            self.timestamp = src_state.timestamp

        return self


class _SingleRealizationStateDict:
    def __init__(self) -> None:
        self._items_by_kind: Dict[str, Dict[str, _SingleRealizationStateDictEntry]] = {}

    def _set_item(
        self,
        key: str,
        value: bool,
        kind: str,
        source: Optional[pathlib.Path] = None,
    ) -> None:
        if key == kind and kind in self._items_by_kind:
            for k in set(self._items_by_kind[kind]) - {kind}:
                self._set_item(k, value, kind, source)

            return

        if kind not in self._items_by_kind:
            self._items_by_kind[kind] = {}

        items_for_kind = self._items_by_kind[kind]

        timestamp = (
            os.path.getctime(source)
            if (source is not None and os.path.exists(source))
            else -1
        )

        if key not in items_for_kind:
            items_for_kind[key] = _SingleRealizationStateDictEntry(
                value=value, timestamp=timestamp
            )

        items_for_kind[key].update(value, timestamp)

    def set_response(
        self,
        key: str,
        value: bool,
        response_type: str,
        source: Optional[pathlib.Path] = None,
    ) -> None:
        self._set_item(key=key, value=value, kind=response_type, source=source)

    def set_parameter_group(
        self,
        key: str,
        value: bool,
        parameter_group: str,
        source: Optional[pathlib.Path] = None,
    ) -> None:
        self._set_item(key=key, value=value, kind=parameter_group, source=source)

    def _lookup_single_kind_dict_for_key(
        self, key: str
    ) -> Dict[str, _SingleRealizationStateDictEntry]:
        matches = [
            (kind, kind_dict)
            for kind, kind_dict in self._items_by_kind.items()
            if key in kind_dict
        ]

        if len(matches) == 0:
            return {}

        assert len(matches) == 1, (
            f"Expected to find only one matching"
            f" kind for key {key}, but found "
            f"{', '.join([k for k,_ in matches])}"
        )
        return matches[0][1]

    def has_response_key_or_group(self, key: str) -> bool:
        if key in self._items_by_kind:
            # It is a response type
            return any(x.value for x in self._items_by_kind[key].values())

        matching_kind_dict = self._lookup_single_kind_dict_for_key(key)

        return key in matching_kind_dict and matching_kind_dict[key].value

    def has_parameter_key_or_group(self, key: str) -> bool:
        if key in self._items_by_kind:
            # It is a parameter group
            # they are always all written at the same time,
            # question2reviewer: If they are all nan, it means that
            # the parameter was somehow sampled and ended up being all nan
            # does that mean that it still HAS the parameter as in it is
            # "something", or does that mean we should return False here?
            # current assumption: We need at least one non NaN
            return any(x.value for x in self._items_by_kind[key].values())

        matching_kind_dict = self._lookup_single_kind_dict_for_key(key)

        return key in matching_kind_dict and matching_kind_dict[key].value

    def get_response(self, key_or_group: str) -> _SingleRealizationStateDictEntry:
        if key_or_group in self._items_by_kind:
            kind_dicts = self._items_by_kind[key_or_group].values()
            return _SingleRealizationStateDictEntry(
                value=any(x.value for x in kind_dicts),
                timestamp=max(x.timestamp for x in kind_dicts),
            )

        kind_dict = self._lookup_single_kind_dict_for_key(key_or_group)
        entry = kind_dict.get(key_or_group)
        assert entry is not None
        return entry

    def get_parameter(self, key_or_group: str) -> _SingleRealizationStateDictEntry:
        if key_or_group in self._items_by_kind:
            kind_dicts = self._items_by_kind[key_or_group].values()
            return _SingleRealizationStateDictEntry(
                value=any(x.value for x in kind_dicts),
                timestamp=max(x.timestamp for x in kind_dicts),
            )

        kind_dict = self._lookup_single_kind_dict_for_key(key_or_group)

        entry = kind_dict.get(key_or_group)
        assert entry is not None
        return entry

    def copy(self) -> "_SingleRealizationStateDict":
        cpy = _SingleRealizationStateDict()
        cpy._items_by_kind = {
            k: {kind: entry.copy() for kind, entry in kind_to_entries.items()}
            for k, kind_to_entries in self._items_by_kind.items()
        }

        return cpy

    def make_keys_consistent(self, keys_per_kind: Dict[str, Set[str]]) -> None:
        for kind, items in self._items_by_kind.items():
            if set(items) == {kind}:
                entry = items[kind]
                for key in keys_per_kind[kind]:
                    items[key] = entry.copy()

        for kind, items in self._items_by_kind.items():
            if kind in set(items) and set(items) != {kind}:
                del items[kind]

    def assign_state(self, src_state: "_SingleRealizationStateDict") -> Self:
        for src_kind, src_items_by_kind in src_state._items_by_kind.items():
            if set(src_items_by_kind) == {src_kind}:
                # Set all existing keys of this state
                if src_kind not in self._items_by_kind:
                    self._items_by_kind[src_kind] = {
                        src_kind: src_items_by_kind[src_kind].copy()
                    }

                if set(self._items_by_kind[src_kind]) == {src_kind}:
                    self._items_by_kind[src_kind][src_kind] = src_items_by_kind[
                        src_kind
                    ].copy()
                else:
                    for k in self._items_by_kind[src_kind]:
                        self._items_by_kind[src_kind][k] = src_items_by_kind[
                            src_kind
                        ].copy()
                continue

            elif src_kind not in self._items_by_kind:
                self._items_by_kind[src_kind] = {
                    k: v.copy() for k, v in src_items_by_kind.items()
                }
                continue

            src_keys_for_kind = set(src_items_by_kind)
            my_keys_for_kind = set(self._items_by_kind[src_kind])
            all_keys = src_keys_for_kind.union(my_keys_for_kind)

            if src_keys_for_kind == {src_kind}:
                # src has all keys for kind set to the same thing
                state_for_all = src_state._items_by_kind[src_kind][src_kind]
                for k in all_keys - {src_kind}:
                    self._items_by_kind[src_kind][k] = state_for_all.copy()
                continue

            for k in all_keys:
                if k in src_keys_for_kind:
                    src_state_entry = src_items_by_kind[k]
                    if k not in my_keys_for_kind:
                        self._items_by_kind[src_kind][k] = src_state_entry.copy()
                    elif k in my_keys_for_kind:
                        my_state = self._items_by_kind[src_kind][k]
                        self._items_by_kind[src_kind][k] = my_state.copy().assign_state(
                            src_state_entry
                        )

        return self

    def to_tuples(self) -> List[Tuple[str, str, _SingleRealizationStateDictEntry]]:
        tuples = []
        for kind, items_for_kind in self._items_by_kind.items():
            for key, entry in items_for_kind.items():
                tuples.append((kind, key, entry))

        return tuples


class _MultiRealizationStateDict:
    def __init__(self) -> None:
        self._items: Dict[int, _SingleRealizationStateDict] = {}

    def has_response(self, realization: int, key: str) -> bool:
        if realization not in self._items:
            return False

        return self._items[realization].has_response_key_or_group(key)

    def has_parameter_group(self, realization: int, key: str) -> bool:
        if realization not in self._items:
            return False

        return self._items[realization].has_parameter_key_or_group(key)

    def is_empty(self) -> bool:
        return self._items == {}

    def get_single_realization_state(
        self, realization: int
    ) -> _SingleRealizationStateDict:
        if realization not in self._items:
            self._items[realization] = _SingleRealizationStateDict()

        return self._items[realization]

    def copy(self) -> "_MultiRealizationStateDict":
        cpy = _MultiRealizationStateDict()
        cpy._items = {
            realization_index: state.copy()
            for realization_index, state in self._items.items()
        }
        return cpy

    def assign_states(self, source: "_MultiRealizationStateDict") -> Self:
        for realization_index, realization_state in source._items.items():
            if realization_index not in self._items:
                self._items[realization_index] = realization_state.copy()
            else:
                self._items[realization_index].assign_state(realization_state)

        return self

    def make_keys_consistent(self) -> Self:
        keys_per_kind: Dict[str, Set[str]] = {}
        for state in self._items.values():
            for kind, key, _ in state.to_tuples():
                if kind not in keys_per_kind:
                    keys_per_kind[kind] = set()

                keys_per_kind[kind].add(key)

            for kind, keys in keys_per_kind.items():
                if set(keys) != {kind} and kind in keys:
                    keys.remove(kind)

        for state in self._items.values():
            state.make_keys_consistent(keys_per_kind)

        return self

    def to_dataframe(self) -> pandas.DataFrame:
        # One column per realization
        # One row per kind-key
        rows = []
        for real, state in self._items.items():
            for kind, key, entry in state.to_tuples():
                rows.append((real, kind, key, entry))

        return (
            pandas.DataFrame(
                data={
                    "realization": [row[0] for row in rows],
                    "kind": [row[1] for row in rows],
                    "key": [row[2] for row in rows],
                    "value": [row[3].value for row in rows],
                    "timestamp": [row[3].timestamp for row in rows],
                }
            )
            .set_index(["realization", "kind", "key"])
            .sort_values(["realization", "kind", "key"])
        )
