from typing import Dict, Iterable, Tuple, Type

from .response_config import ResponseConfig


class _ResponsesIndex:
    def __init__(self) -> None:
        self._items: Dict[str, Type[ResponseConfig]] = {}

    def add_response_type(self, response_cls: Type[ResponseConfig]) -> None:
        if not issubclass(response_cls, ResponseConfig):
            raise ValueError("Response type must be subclass of ResponseConfig")

        clsname = response_cls.__name__

        if clsname in self._items:
            raise KeyError(
                f"Response type with name {clsname} is already registered. Please "
                f"use another classname to avoid this conflict."
            )

        self._items[clsname] = response_cls

    def values(self) -> Iterable[Type[ResponseConfig]]:
        return self._items.values()

    def items(self) -> Iterable[Tuple[str, Type[ResponseConfig]]]:
        return self._items.items()

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def __getitem__(self, item: str) -> Type[ResponseConfig]:
        return self._items[item]

    def __contains__(self, item: str) -> bool:
        return item in self._items


# Convention for adding responses should be that responses
# import responses_index and do a call to .add_response_type()
# Hence we do not explicitly import GenDataConfig / SummaryConfig and add
# them here.
responses_index: _ResponsesIndex = _ResponsesIndex()
