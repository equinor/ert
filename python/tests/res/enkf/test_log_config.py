#  Copyright (C) 2017  Statoil ASA, Norway.
#
#  The file 'test_log_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

import os, itertools

from ecl.test import TestAreaContext
from tests import ResTest
from res.util.enums import MessageLevelEnum

from res.enkf import LogConfig, ResConfig

class LogConfigTest(ResTest):

    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")
        self.config_file = "simple_config/minimum_config"

        self.log_files = [
                            (None, "simple_config/log"),
                            ("file_loglog", "simple_config/file_loglog"),
                            ("this/is/../my/log/file.loglog", "simple_config/this/my/log/file.loglog")
                         ]

        self.log_levels = [(None, MessageLevelEnum.LOG_ERROR)]
        for message_level in MessageLevelEnum.enums():
            # Add new log level
            self.log_levels.append((message_level.name.split("_")[1], message_level))

            # Add old log level
            self.log_levels.append((str(message_level.value), message_level))

    def assert_log_config_load(
            self,
            log_file, exp_log_file,
            log_level, exp_log_level,
            write_abs_path=False
            ):


        with TestAreaContext("log_config_test") as work_area:
            work_area.copy_directory(self.case_directory)

            # Append config file
            with open(self.config_file, 'a') as cf:
                if log_file:
                    if write_abs_path:
                        log_file = os.path.join(
                                os.path.abspath(os.path.split(self.config_file)[0]),
                                log_file
                                )

                    cf.write("\nLOG_FILE %s\n" % log_file)

                if log_level:
                    cf.write("\nLOG_LEVEL %s\n" % log_level)

            log_config = LogConfig(self.config_file)

            self.assertTrue(os.path.isabs(log_config.log_file))

            self.assertEqual(
                    os.path.normpath(log_config.log_file),
                    os.path.normpath(os.path.abspath(exp_log_file))
                    )

            self.assertEqual(
                    log_config.log_level,
                    exp_log_level
                    )


    def test_log_config(self):
        test_cases = itertools.product(self.log_files, self. log_levels)

        for log_file_data, log_level_data in test_cases:
            self.assert_log_config_load(
                    log_file_data[0], log_file_data[1],
                    log_level_data[0], log_level_data[1]
                    )

            if log_file_data[0]:
                self.assert_log_config_load(
                        log_file_data[0], log_file_data[1],
                        log_level_data[0], log_level_data[1],
                        write_abs_path=True
                        )
