from __future__ import annotations

from typing import TYPE_CHECKING

from ert.gui.ertwidgets import SelectableListModel

if TYPE_CHECKING:
    from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition


class FilterableKwListModel(SelectableListModel):
    """
    Adds ERT - plotting keyword specific filtering functionality to the general
    SelectableListModel
    """

    def __init__(self, key_defs: list[PlotApiKeyDefinition]) -> None:
        SelectableListModel.__init__(self, [k.key for k in key_defs])
        self._key_defs = key_defs
        self._metadata_filters: dict[str, dict[str, bool]] = {}

    def getList(self) -> list[str]:
        items = []
        for item in self._key_defs:
            add = True
            for meta_key, meta_value in item.metadata.items():
                if (
                    meta_key in self._metadata_filters
                    and not self._metadata_filters[meta_key][meta_value]
                ):
                    add = False

            if add:
                items.append(item.key)
        return items

    def setFilterOnMetadata(self, key: str, value: str, visible: bool) -> None:
        if key not in self._metadata_filters:
            self._metadata_filters[key] = {}

        self._metadata_filters[key][value] = visible
