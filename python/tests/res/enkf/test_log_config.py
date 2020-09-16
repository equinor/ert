#  Copyright (C) 2017  Equinor ASA, Norway.
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

import itertools
import os
import sys

from ecl.util.test import TestAreaContext

from res.enkf import LogConfig, ResConfig, ConfigKeys
from res.util.enums import MessageLevelEnum
from tests import ResTest


class LogConfigTest(ResTest):

    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")
        self.config_file = "simple_config/minimum_config"

        self.log_files = [
            (None, "simple_config/log.txt"),
            ("file_loglog", "simple_config/file_loglog"),
            ("this/is/../my/log/file.loglog", "simple_config/this/my/log/file.loglog")
        ]

        self.log_levels = [(None, MessageLevelEnum.LOG_WARNING)]
        for message_level in [lev for lev in MessageLevelEnum.enums() if lev.value]:
            self.log_levels.append((message_level.name.split("_")[1], message_level))

    def assert_log_config_load(
            self,
            log_file, exp_log_file,
            log_level, exp_log_level,
            write_abs_path=False
            ):

        with TestAreaContext("log_config_test") as work_area:
            work_area.copy_directory(self.case_directory)

            config_dict = {}
            # Append config file
            with open(self.config_file, 'a') as cf:
                if log_file:
                    config_dict[ConfigKeys.LOG_FILE] = os.path.realpath(
                        os.path.join(
                            os.path.abspath(
                                os.path.split(
                                    self.config_file)[0]
                            ), log_file
                        )
                    )

                    if write_abs_path:
                        log_file = config_dict[ConfigKeys.LOG_FILE]

                    cf.write("\nLOG_FILE %s\n" % log_file)

                else:
                    config_dict[ConfigKeys.LOG_FILE] = os.path.realpath(
                        os.path.join(
                            os.path.abspath(
                                os.path.split(
                                    self.config_file)[0]
                            ), "log.txt"
                        )
                    )

                if log_level:
                    level = log_level
                    if sys.version_info[0] >= 3:
                        if not log_level.isalpha():
                            level = int(float(level))
                    cf.write("\nLOG_LEVEL %s\n" % level)
                    if type(level) is str and level.isdigit():
                        config_dict[ConfigKeys.LOG_LEVEL] = MessageLevelEnum.to_enum(eval(level))
                    elif type(level) is str:
                        config_dict[ConfigKeys.LOG_LEVEL] = MessageLevelEnum.from_string("LOG_" + level)
                    else:
                        config_dict[ConfigKeys.LOG_LEVEL] = MessageLevelEnum.to_enum(level)
                else:
                    config_dict[ConfigKeys.LOG_LEVEL] = MessageLevelEnum.LOG_WARNING

            log_config = LogConfig(user_config_file=self.config_file)
            res_config = ResConfig(self.config_file)
            log_config_dict = LogConfig(config_dict=config_dict)
            self.assertEqual(log_config, log_config_dict)
            self.assertEqual(log_config, res_config.log_config)

            self.assertTrue(os.path.isabs(log_config.log_file))

            self.assertEqual(
                    os.path.normpath(log_config.log_file),
                    os.path.normpath(os.path.abspath(exp_log_file))
                    )

            if isinstance(log_config.log_level, int):
                level = MessageLevelEnum.to_enum(log_config.log_level)
            else:
                level = log_config.log_level

            self.assertEqual(
                    level,
                    exp_log_level
            )

    def test_log_config(self):
        test_cases = itertools.product(self.log_files, self.log_levels)

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
