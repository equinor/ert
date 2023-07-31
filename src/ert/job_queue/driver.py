from typing import List, Optional, Tuple

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert.config import QueueDriverEnum


class Driver(BaseCClass):  # type: ignore
    TYPE_NAME = "driver"
    _alloc = ResPrototype("void* queue_driver_alloc( queue_driver_enum )", bind=False)
    _free = ResPrototype("void queue_driver_free( driver )")
    _set_option = ResPrototype("void queue_driver_set_option( driver , char* , char*)")
    _unset_option = ResPrototype("void queue_driver_unset_option( driver , char*)")
    _get_option = ResPrototype("char* queue_driver_get_option(driver, char*)")
    _get_max_running = ResPrototype("int queue_driver_get_max_running( driver )")
    _set_max_running = ResPrototype("void queue_driver_set_max_running( driver , int)")
    _get_name = ResPrototype("char* queue_driver_get_name( driver )")

    def __init__(
        self,
        driver_type: QueueDriverEnum,
        max_running: int = 0,
        options: Optional[List[Tuple[str, str]]] = None,
    ):
        c_ptr = self._alloc(driver_type)
        super().__init__(c_ptr)
        if options:
            for key, value in options:
                self.set_option(key, value)
        self.set_max_running(max_running)

    def set_option(self, option: str, value: str) -> bool:
        """Set a driver option to a specific value, return False if unknown option."""
        return self._set_option(option, str(value))  # type: ignore

    def unset_option(self, option: str) -> None:
        return self._unset_option(option)  # type: ignore

    def get_option(self, option_key: str) -> str:
        return self._get_option(option_key)  # type: ignore

    def get_max_running(self) -> int:
        return self._get_max_running()  # type: ignore

    def set_max_running(self, max_running: int) -> None:
        self._set_max_running(max_running)

    max_running = property(get_max_running, set_max_running)

    @property
    def name(self) -> str:
        return self._get_name()  # type: ignore

    def free(self) -> None:
        self._free()
