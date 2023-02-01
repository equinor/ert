from typing import List, Tuple

from cwrap import BaseCClass, BaseCEnum

from ert._c_wrappers import ResPrototype

from .job import Job


class QueueDriverEnum(BaseCEnum):
    TYPE_NAME = "queue_driver_enum"
    NULL_DRIVER = None
    LSF_DRIVER = None
    LOCAL_DRIVER = None
    TORQUE_DRIVER = None
    SLURM_DRIVER = None


QueueDriverEnum.addEnum("NULL_DRIVER", 0)
QueueDriverEnum.addEnum("LSF_DRIVER", 1)
QueueDriverEnum.addEnum("LOCAL_DRIVER", 2)
QueueDriverEnum.addEnum("TORQUE_DRIVER", 4)
QueueDriverEnum.addEnum("SLURM_DRIVER", 5)


LSF_DRIVER = QueueDriverEnum.LSF_DRIVER
LOCAL_DRIVER = QueueDriverEnum.LOCAL_DRIVER
SLURM_DRIVER = QueueDriverEnum.SLURM_DRIVER


class Driver(BaseCClass):
    TYPE_NAME = "driver"
    _alloc = ResPrototype("void* queue_driver_alloc( queue_driver_enum )", bind=False)
    _free = ResPrototype("void queue_driver_free( driver )")
    _set_option = ResPrototype("void queue_driver_set_option( driver , char* , char*)")
    _unset_option = ResPrototype("void queue_driver_unset_option( driver , char*)")
    _get_option = ResPrototype("char* queue_driver_get_option(driver, char*)")
    _free_job = ResPrototype("void   queue_driver_free_job( driver , job )")
    _get_status = ResPrototype(
        "job_status_type_enum queue_driver_get_status(driver, job)"
    )
    _kill_job = ResPrototype("void queue_driver_kill_job( driver , job )")
    _get_max_running = ResPrototype("int queue_driver_get_max_running( driver )")
    _set_max_running = ResPrototype("void queue_driver_set_max_running( driver , int)")
    _get_name = ResPrototype("char* queue_driver_get_name( driver )")

    def __init__(
        self,
        driver_type: QueueDriverEnum,
        max_running: int = 0,
        options: List[Tuple[str, str]] = None,
    ):
        c_ptr = self._alloc(driver_type)
        super().__init__(c_ptr)
        if options:
            for key, value in options:
                self.set_option(key, value)
        self.set_max_running(max_running)

    def set_option(self, option: str, value: str) -> bool:
        """Set a driver option to a specific value, return False if unknown option."""
        return self._set_option(option, str(value))

    def unset_option(self, option):
        return self._unset_option(option)

    def get_option(self, option_key):
        return self._get_option(option_key)

    def is_driver_instance(self):
        return True

    def free_job(self, job: Job):
        self._free_job(job)

    def get_status(self, job):
        return self._get_status(job)

    def kill_job(self, job):
        self._kill_job(job)

    def get_max_running(self):
        return self._get_max_running()

    def set_max_running(self, max_running):
        self._set_max_running(max_running)

    max_running = property(get_max_running, set_max_running)

    @property
    def name(self):
        return self._get_name()

    def free(self):
        self._free()


class LSFDriver(Driver):
    def __init__(
        self, max_running, lsf_server=None, queue="normal", resource_request=None
    ):
        # The strings should match the available keys given in the
        # lsf_driver.h header file.
        options = [
            ("LSF_QUEUE", queue),
            ("LSF_SERVER", lsf_server),
            ("LSF_RESOURCE", resource_request),
        ]
        Driver.__init__(
            self, QueueDriverEnum.LSF_DRIVER, max_running=max_running, options=options
        )


class LocalDriver(Driver):
    def __init__(self, max_running):
        Driver.__init__(self, QueueDriverEnum.LOCAL_DRIVER, max_running, options=[])
