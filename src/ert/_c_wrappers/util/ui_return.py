from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype

from .enums import UIReturnStatusEnum


class UIReturn(BaseCClass):
    TYPE_NAME = "ui_return"

    _alloc = ResPrototype("void* ui_return_alloc( ui_return_status )", bind=False)
    _free = ResPrototype("void ui_return_free(ui_return)")
    _get_status = ResPrototype("ui_return_status ui_return_get_status(ui_return)")
    _get_help = ResPrototype("char* ui_return_get_help(ui_return)")
    _add_help = ResPrototype("bool ui_return_add_help(ui_return, char*)")
    _add_error = ResPrototype("bool ui_return_add_error(ui_return, char*)")
    _num_error = ResPrototype("int ui_return_get_error_count(ui_return)")
    _last_error = ResPrototype("char* ui_return_get_last_error(ui_return)")
    _first_error = ResPrototype("char* ui_return_get_first_error(ui_return)")
    _iget_error = ResPrototype("char* ui_return_iget_error(ui_return ,       int)")

    def __init__(self, status):
        c_ptr = self._alloc(status)
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(f"Unable to construct UIReturn with status = {status}")

    # For python 3, corresponds to __nonzero__
    def __bool__(self):
        return self.status() == UIReturnStatusEnum.UI_RETURN_OK

    # For python 2
    def __nonzero__(self):
        return self.__bool__()

    def __len__(self):
        return self._num_error()

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if 0 <= index < len(self):
                return self._iget_error(index)
            else:
                raise IndexError(f"Invalid index.  Valid range: [0, {len(self)})")
        else:
            raise TypeError("Lookup type must be integer")

    def iget_error(self, index):
        return self[index]

    def help_text(self):
        help_text = self._get_help()
        if help_text:
            return help_text
        else:
            return ""

    def add_help(self, help_text):
        self._add_help(help_text)

    def status(self):
        return self._get_status()

    def __assert_error(self):
        if self.status() == UIReturnStatusEnum.UI_RETURN_OK:
            raise ValueError("Can not add error messages to object in state RETURN_OK")

    def add_error(self, error):
        self.__assert_error()
        self._add_error(error)

    def last_error(self):
        self.__assert_error()
        return self._last_error()

    def first_error(self):
        self.__assert_error()
        return self._first_error()

    def free(self):
        self._free()

    def __repr__(self):
        return (
            f"UIReturn(error_count = {len(self)}, status = {self.status()}) "
            f"{self._ad_str()}"
        )
