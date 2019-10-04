from ert_gui.ertwidgets.models.selectable_list_model import SelectableListModel
from ert_shared import ERT

class FilterableKwListModel(SelectableListModel):
    """
    Adds ERT - plotting keyword specific filtering functionality to the general SelectableListModel
    """
    def __init__(self, selectable_keys):
        SelectableListModel.__init__(self, selectable_keys)
        self._show_summary_keys = True
        self._show_gen_kw_keys = True
        self._show_gen_data_keys = True
        self._show_custom_kw_keys = True

    def getList(self):
        filtered_list = []
        for item in self._items:
            if self._show_summary_keys and self.isSummaryKey(item):
                filtered_list.append(item)
            elif self._show_gen_kw_keys and self.isGenKWKey(item):
                filtered_list.append(item)
            elif self._show_gen_data_keys and self.isGenDataKey(item):
                filtered_list.append(item)
            elif self._show_custom_kw_keys and self.isCustomKwKey(item):
                filtered_list.append(item)

        return filtered_list

    def isSummaryKey(self, key):
        return ERT.enkf_facade.is_summary_key(key)

    def isGenKWKey(self, key):
        return ERT.enkf_facade.is_gen_kw_key(key)

    def isGenDataKey(self, key):
        return ERT.enkf_facade.is_gen_data_key(key)

    def isCustomKwKey(self, key):
        return ERT.enkf_facade.is_custom_kw_key(key)

    def setShowSummaryKeys(self, visible):
        self._show_summary_keys = visible

    def setShowGenKWKeys(self, visible):
        self._show_gen_kw_keys = visible

    def setShowGenDataKeys(self, visible):
        self._show_gen_data_keys = visible

    def setShowCustomKwKeys(self, visible):
        self._show_custom_kw_keys = visible
