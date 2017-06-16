#  Copyright (C) 2017  Statoil ASA, Norway.
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

from ecl.test import ExtendedTestCase

from res.enkf import QueueConfig

class QueueConfigTest(ExtendedTestCase):

    def test_get_queue_config(self):
        QC = QueueConfig()
        JobQueue = QC.create_job_queue()
        QC_copy = QC.create_local_copy()
        self.assertFalse(QC_copy.has_job_script())

    
