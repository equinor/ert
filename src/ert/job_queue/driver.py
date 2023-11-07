from typing import List, Optional, Tuple

from cwrap import BaseCClass

from ert.config import QueueConfig, QueueSystem

from . import ResPrototype


class Driver(BaseCClass):  # type: ignore
    TYPE_NAME = "driver"
    _alloc = ResPrototype("void* queue_driver_alloc( queue_driver_enum )", bind=False)
    _free = ResPrototype("void queue_driver_free( driver )")
    _set_option = ResPrototype("bool queue_driver_set_option( driver , char* , char*)")
    _get_option = ResPrototype("char* queue_driver_get_option(driver, char*)")

    def __init__(
        self,
        driver_type: QueueSystem,
        options: Optional[List[Tuple[str, str]]] = None,
    ):
        c_ptr = self._alloc(driver_type.to_c_enum())
        super().__init__(c_ptr)
        self._max_running = 0
        self._driver_name = driver_type.name

        if options:
            for key, value in options:
                self.set_option(key, value)

    def set_option(self, option: str, value: str) -> bool:
        if option == "MAX_RUNNING":
            self.set_max_running(int(value) if value else 0)
            return True
        else:
            return self._set_option(option, str(value))

    def get_option(self, option_key: str) -> str:
        if option_key == "MAX_RUNNING":
            return str(self.get_max_running())
        else:
            return self._get_option(option_key)

    def get_max_running(self) -> int:
        return self._max_running

    def set_max_running(self, max_running: int) -> None:
        self._max_running = max_running

    @classmethod
    def create_driver(cls, queue_config: QueueConfig) -> "Driver":
        driver = Driver(queue_config.queue_system)
        if queue_config.queue_system in queue_config.queue_options:
            for setting in queue_config.queue_options[queue_config.queue_system]:
                if not driver.set_option(*setting):
                    raise ValueError(f"Queue option not set {setting}")
        return driver

    @property
    def name(self) -> str:
        return self._driver_name

    def free(self) -> None:
        self._free()
