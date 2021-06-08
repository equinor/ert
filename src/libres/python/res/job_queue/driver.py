#  Copyright (C) 2011  Equinor ASA, Norway.
#
#  The file 'driver.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.


import ctypes
from cwrap import BaseCClass, BaseCEnum
from res import ResPrototype
from res.job_queue import Job


class QueueDriverEnum(BaseCEnum):
    TYPE_NAME = "queue_driver_enum"
    NULL_DRIVER = None
    LSF_DRIVER = None
    LOCAL_DRIVER = None
    RSH_DRIVER = None
    TORQUE_DRIVER = None
    SLURM_DRIVER = None


QueueDriverEnum.addEnum("NULL_DRIVER", 0)
QueueDriverEnum.addEnum("LSF_DRIVER", 1)
QueueDriverEnum.addEnum("LOCAL_DRIVER", 2)
QueueDriverEnum.addEnum("RSH_DRIVER", 3)
QueueDriverEnum.addEnum("TORQUE_DRIVER", 4)
QueueDriverEnum.addEnum("SLURM_DRIVER", 5)


LSF_DRIVER = QueueDriverEnum.LSF_DRIVER
RSH_DRIVER = QueueDriverEnum.RSH_DRIVER
LOCAL_DRIVER = QueueDriverEnum.LOCAL_DRIVER
SLURM_DRIVER = QueueDriverEnum.SLURM_DRIVER


class Driver(BaseCClass):
    TYPE_NAME = "driver"
    _alloc = ResPrototype("void* queue_driver_alloc( queue_driver_enum )", bind=False)
    _free = ResPrototype("void queue_driver_free( driver )")
    _set_option = ResPrototype("void queue_driver_set_option( driver , char* , char*)")
    _get_option = ResPrototype("char* queue_driver_get_option(driver, char*)")
    _free_job = ResPrototype("void   queue_driver_free_job( driver , job )")
    _get_status = ResPrototype(
        "job_status_type_enum queue_driver_get_status(driver, job)"
    )
    _kill_job = ResPrototype("void queue_driver_kill_job( driver , job )")
    _get_max_running = ResPrototype("int queue_driver_get_max_running( driver )")
    _set_max_running = ResPrototype("void queue_driver_set_max_running( driver , int)")
    _get_name = ResPrototype("char* queue_driver_get_name( driver )")

    def __init__(self, driver_type, max_running=1, options=None):
        """
        Creates a new driver instance
        """
        c_ptr = self._alloc(driver_type)
        super(Driver, self).__init__(c_ptr)
        if options:
            for (key, value) in options:
                self.set_option(key, value)
        self.set_max_running(max_running)

    def set_option(self, option, value):
        """
        Set the driver option @option to @value.

        If the option is succlessfully set the method will return True,
        otherwise the method will return False. If the @option is not
        recognized the method will return False. The supplied value
        should be a string.
        """
        return self._set_option(option, str(value))

    def get_option(self, option_key):
        return self._get_option(option_key)

    def is_driver_instance(self):
        return True

    def free_job(self, job):
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


class RSHDriver(Driver):
    # Changing shell to bash can come in conflict with running ssh
    # commands.

    def __init__(self, max_running, rsh_host_list, rsh_cmd="/usr/bin/ssh"):
        """
        @rsh_host_list should be a list of tuples like: (hostname , max_running)
        """

        options = [("RSH_CMD", rsh_cmd)]
        for (host, host_max) in rsh_host_list:
            options.append(("RSH_HOST", "%s:%d" % (host, host_max)))
        Driver.__init__(self, QueueDriverEnum.RSH_DRIVER, max_running, options=options)
