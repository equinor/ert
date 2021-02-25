from ert_gui.ertwidgets.models.selectable_list_model import SelectableListModel


class FilterableKwListModel(SelectableListModel):
    """
    Adds ERT - plotting keyword specific filtering functionality to the general SelectableListModel
    """

    def __init__(self, key_defs):
        SelectableListModel.__init__(self, [k["key"] for k in key_defs])
        self._key_defs = key_defs
        self._metadata_filters = {}

    def getList(self):
        items = []
        for item in self._key_defs:
            add = True
            for meta_key, meta_value in item["metadata"].items():
                if (
                    meta_key in self._metadata_filters
                    and not self._metadata_filters[meta_key][meta_value]
                ):
                    add = False

            if add:
                items.append(item["key"])
        return items

    def setFilterOnMetadata(self, key, value, visible):
        if key not in self._metadata_filters:
            self._metadata_filters[key] = {}

        self._metadata_filters[key][value] = visible
