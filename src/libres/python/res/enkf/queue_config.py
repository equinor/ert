#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'site_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

from cwrap import BaseCClass

from ecl.util.util import StringList, Hash

from res import ResPrototype
from res.enkf import ConfigKeys
from res.job_queue import JobQueue, ExtJoblist, Driver


class QueueConfig(BaseCClass):

    TYPE_NAME = "queue_config"

    _free = ResPrototype("void queue_config_free( queue_config )")
    _alloc_job_queue = ResPrototype(
        "job_queue_obj queue_config_alloc_job_queue( queue_config )"
    )
    _alloc = ResPrototype("void* queue_config_alloc_load(char*)", bind=False)
    _alloc_full = ResPrototype(
        "void* queue_config_alloc_full(char*, bool, int, int, queue_driver_enum)",
        bind=False,
    )
    _alloc_content = ResPrototype(
        "void* queue_config_alloc(config_content)", bind=False
    )
    _alloc_local_copy = ResPrototype(
        "queue_config_obj queue_config_alloc_local_copy( queue_config )"
    )
    _has_job_script = ResPrototype("bool queue_config_has_job_script( queue_config )")
    _get_job_script = ResPrototype("char* queue_config_get_job_script(queue_config)")
    _max_submit = ResPrototype("int queue_config_get_max_submit(queue_config)")
    _queue_system = ResPrototype("char* queue_config_get_queue_system(queue_config)")
    _queue_driver = ResPrototype(
        "driver_ref queue_config_get_queue_driver(queue_config, char*)"
    )
    _get_num_cpu = ResPrototype("int queue_config_get_num_cpu(queue_config)")

    _lsf_queue_opt = ResPrototype("char* queue_config_lsf_queue_name()", bind=False)
    _lsf_server_opt = ResPrototype("char* queue_config_lsf_server()", bind=False)
    _lsf_resource_opt = ResPrototype("char* queue_config_lsf_resource()", bind=False)
    _lsf_driver_opt = ResPrototype("char* queue_config_lsf_driver_name()", bind=False)

    def __init__(self, user_config_file=None, config_content=None, config_dict=None):
        configs = sum(
            [
                1
                for x in [user_config_file, config_content, config_dict]
                if x is not None
            ]
        )

        if configs > 1:
            raise ValueError(
                "Attempting to create QueueConfig object with multiple config objects"
            )

        if configs == 0:
            raise ValueError(
                "Attempting to create QueueConfig object with no config objects"
            )

        c_ptr = None
        if user_config_file is not None:
            c_ptr = self._alloc(user_config_file)

        if config_content is not None:
            c_ptr = self._alloc_content(config_content)

        if config_dict is not None:
            c_ptr = self._alloc_full(
                config_dict[ConfigKeys.JOB_SCRIPT],
                config_dict[ConfigKeys.USER_MODE],
                config_dict[ConfigKeys.MAX_SUBMIT],
                config_dict[ConfigKeys.NUM_CPU],
                config_dict[ConfigKeys.QUEUE_SYSTEM],
            )
        if not c_ptr:
            raise ValueError("Unable to create QueueConfig instance")

        super(QueueConfig, self).__init__(c_ptr)

        # Need to create
        if config_dict is not None:
            queue_options = config_dict.get(ConfigKeys.QUEUE_OPTION)
            for option in queue_options:
                self.driver.set_option(
                    option[ConfigKeys.NAME], option[ConfigKeys.VALUE]
                )

    def create_job_queue(self):
        queue = JobQueue(self.driver, max_submit=self.max_submit)
        return queue

    def create_local_copy(self):
        return self._alloc_local_copy()

    def has_job_script(self):
        return self._has_job_script()

    def free(self):
        self._free()

    @property
    def max_submit(self):
        return self._max_submit()

    @property
    def queue_name(self):
        return self.driver.get_option(QueueConfig.LSF_QUEUE_NAME_KEY)

    @property
    def queue_system(self):
        """The queue system in use, e.g. LSF or LOCAL"""
        return self._queue_system()

    @property
    def job_script(self):
        return self._get_job_script()

    @property
    def driver(self):
        return self._queue_driver(self.queue_system).setParent(self)

    def _assert_lsf(self, key="driver"):
        sys = self.queue_system
        if sys != QueueConfig.LSF_KEY:
            fmt = "Cannot fetch LSF {key}, current queue is {system}"
            raise ValueError(fmt.format(key=key, system=self.queue_system))

    @property
    def _lsf_driver(self):
        self._assert_lsf()
        driver = self._queue_driver(self.LSF_KEY)
        return driver.setParent(self)

    @property
    def lsf_resource(self):
        self._assert_lsf(key=QueueConfig.LSF_RESOURCE_KEY)
        return self._lsf_driver.get_option(self.LSF_RESOURCE_KEY)

    @property
    def lsf_server(self):
        self._assert_lsf(key=QueueConfig.LSF_SERVER_KEY)
        return self._lsf_driver.get_option(self.LSF_SERVER_KEY)

    @property
    def num_cpu(self):
        return self._get_num_cpu()

    def __eq__(self, other):

        if self.max_submit != other.max_submit:
            return False
        if self.queue_system != other.queue_system:
            return False
        if self.num_cpu != other.num_cpu:
            return False
        if self.job_script != other.job_script:
            return False

        if self.queue_system != "LOCAL":
            if self.queue_name != other.queue_name:
                return False
            if self.lsf_resource != other.lsf_resource:
                return False
            if self.lsf_server != other.lsf_server:
                return False

        return True

    LSF_KEY = _lsf_driver_opt()
    LSF_QUEUE_NAME_KEY = _lsf_queue_opt()
    LSF_RESOURCE_KEY = _lsf_resource_opt()
    LSF_SERVER_KEY = _lsf_server_opt()
