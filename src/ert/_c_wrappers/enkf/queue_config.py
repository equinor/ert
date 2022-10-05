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

import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from ert._c_wrappers.job_queue import Driver, JobQueue, QueueDriverEnum


@dataclass
class QueueConfig:
    job_script: str = shutil.which("job_dispatch.py") or "job_dispatch.py"
    max_submit: int = 2
    num_cpu: int = 0
    queue_system: QueueDriverEnum = QueueDriverEnum.NULL_DRIVER
    queue_options: Dict[QueueDriverEnum, List[Union[Tuple[str, str], str]]] = field(
        default_factory=dict
    )

    def create_driver(self) -> Driver:
        driver = Driver(self.queue_system)
        if self.queue_system in self.queue_options:
            for setting in self.queue_options[self.queue_system]:
                if isinstance(setting, Tuple):
                    driver.set_option(*setting)
                else:
                    driver.unset_option(setting)
        return driver

    def create_job_queue(self) -> JobQueue:
        queue = JobQueue(self.create_driver(), max_submit=self.max_submit)
        return queue

    def create_local_copy(self):
        return QueueConfig(
            self.job_script,
            self.max_submit,
            self.num_cpu,
            QueueDriverEnum.LOCAL_DRIVER,
            self.queue_options,
        )
