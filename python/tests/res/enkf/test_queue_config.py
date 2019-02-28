#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_summary_obs.py' is part of ERT - Ensemble based Reservoir Tool.
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

import os

from tests import ResTest
from res.enkf import QueueConfig

class QueueConfigTest(ResTest):

    def test_get_queue_config(self):
        queue_config = QueueConfig(None)
        job_queue = queue_config.create_job_queue()
        queue_config_copy = queue_config.create_local_copy()

        self.assertEqual(
                queue_config.has_job_script(),
                queue_config_copy.has_job_script()
                )
