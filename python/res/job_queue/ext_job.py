#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'ext_job.py' is part of ERT - Ensemble based Reservoir Tool.
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
import os.path

from cwrap import BaseCClass
from res import ResPrototype
from ecl.util.util import StringList, Hash
from res.config import ContentTypeEnum


class ExtJob(BaseCClass):
    TYPE_NAME = "ext_job"
    _alloc = ResPrototype("void* ext_job_alloc(char*,char*, bool)", bind=False)
    _fscanf_alloc = ResPrototype(
        "void* ext_job_fscanf_alloc(char*, char*, bool, char* , bool)", bind=False
    )
    _free = ResPrototype("void ext_job_free( ext_job )")
    _get_help_text = ResPrototype("char* ext_job_get_help_text(ext_job)")
    _get_name = ResPrototype("char* ext_job_get_name(ext_job)")
    _set_private_args_as_string = ResPrototype(
        "void ext_job_set_private_args_from_string(ext_job, char*)"
    )
    _is_private = ResPrototype("int ext_job_is_private(ext_job)")
    _get_config_file = ResPrototype("char* ext_job_get_config_file(ext_job)")
    _set_config_file = ResPrototype("void ext_job_set_config_file(ext_job, char*)")
    _get_stdin_file = ResPrototype("char* ext_job_get_stdin_file(ext_job)")
    _set_stdin_file = ResPrototype("void ext_job_set_stdin_file(ext_job, char*)")
    _get_stdout_file = ResPrototype("char* ext_job_get_stdout_file(ext_job)")
    _set_stdout_file = ResPrototype("void ext_job_set_stdout_file(ext_job, char*)")
    _get_stderr_file = ResPrototype("char* ext_job_get_stderr_file(ext_job)")
    _set_stderr_file = ResPrototype("void ext_job_set_stderr_file(ext_job, char*)")
    _get_target_file = ResPrototype("char* ext_job_get_target_file(ext_job)")
    _set_target_file = ResPrototype("void ext_job_set_target_file(ext_job, char*)")
    _get_executable = ResPrototype("char* ext_job_get_executable(ext_job)")
    _set_executable = ResPrototype("void ext_job_set_executable(ext_job, char*)")
    _get_error_file = ResPrototype("char* ext_job_get_error_file(ext_job)")
    _get_start_file = ResPrototype("char* ext_job_get_start_file(ext_job)")
    _get_max_running = ResPrototype("int ext_job_get_max_running(ext_job)")
    _set_max_running = ResPrototype("void ext_job_set_max_running(ext_job, int)")
    _get_max_running_minutes = ResPrototype(
        "int ext_job_get_max_running_minutes(ext_job)"
    )
    _set_max_running_minutes = ResPrototype(
        "void ext_job_set_max_running_minutes(ext_job, int)"
    )

    _min_arg = ResPrototype("int ext_job_get_min_arg(ext_job)")
    _max_arg = ResPrototype("int ext_job_get_max_arg(ext_job)")
    _arg_type = ResPrototype(
        "config_content_type_enum ext_job_iget_argtype(ext_job, int)"
    )

    _get_environment = ResPrototype("string_hash_ref ext_job_get_environment(ext_job)")
    _set_environment = ResPrototype(
        "void ext_job_add_environment(ext_job, char*, char*)"
    )
    _get_license_path = ResPrototype("char* ext_job_get_license_path(ext_job)")
    _get_arglist = ResPrototype("stringlist_ref ext_job_get_arglist(ext_job)")
    _set_arglist = ResPrototype("void ext_job_set_args(ext_job, stringlist)")
    _get_argvalues = ResPrototype("stringlist_ref ext_job_get_argvalues(ext_job)")
    _clear_environment = ResPrototype("void ext_job_clear_environment(ext_job)")
    _save = ResPrototype("void ext_job_save(ext_job)")

    def __init__(
        self, config_file, private, name=None, license_root_path=None, search_PATH=True
    ):
        if os.path.isfile(config_file):
            if name is None:
                name = os.path.basename(config_file)

            c_ptr = self._fscanf_alloc(
                name, license_root_path, private, config_file, search_PATH
            )
            if c_ptr:
                super(ExtJob, self).__init__(c_ptr)
            else:
                raise ValueError(
                    "Unable to construct ExtJob(name=%s, config_file=%s, private=%s)"
                    % (name, config_file, private)
                )
        else:
            raise IOError('No such config file "%s".' % config_file)

    def __repr__(self):
        if self._address():
            return self._create_repr(
                "{}, config_file = {}".format(self.name(), self.get_config_file())
            )
        else:
            return "UNINITIALIZED ExtJob"

    def set_private_args_as_string(self, args):
        self._set_private_args_as_string(args)

    def get_help_text(self):
        return self._get_help_text()

    def is_private(self):
        return self._is_private()

    def get_config_file(self):
        return self._get_config_file()

    def set_config_file(self, config_file):
        self._set_config_file(config_file)

    def get_stdin_file(self):
        return self._get_stdin_file()

    def set_stdin_file(self, filename):
        self._set_stdin_file(filename)

    def get_stdout_file(self):
        return self._get_stdout_file()

    def set_stdout_file(self, filename):
        self._set_stdout_file(filename)

    def get_stderr_file(self):
        return self._get_stderr_file()

    def set_stderr_file(self, filename):
        self._set_stderr_file(filename)

    def get_target_file(self):
        return self._get_target_file()

    def set_target_file(self, filename):
        self._set_target_file(filename)

    def get_executable(self):
        return self._get_executable()

    def set_executable(self, executable):
        self._set_executable(executable)

    def get_max_running(self):
        return self._get_max_running()

    def set_max_running(self, max_running):
        self._set_max_running(max_running)

    def get_error_file(self):
        return self._get_error_file()

    def get_start_file(self):
        return self._get_start_file()

    def get_max_running_minutes(self):
        return self._get_max_running_minutes()

    def set_max_running_minutes(self, min_value):
        self._set_max_running_minutes(min_value)

    @property
    def min_arg(self):
        return self._min_arg()

    @property
    def max_arg(self):
        return self._max_arg()

    @property
    def arg_types(self):
        """ @rtype: list of type """

        result = []
        for index in range(self.max_arg):
            result.append(self._arg_type(index))

        return result

    @staticmethod
    def valid_args(arg_types, arg_list, runtime=False):
        for index, arg_type in enumerate(arg_types):
            arg = arg_list[index]
            if not arg_type.valid_string(arg, runtime):
                return False
            return True

    def get_environment(self):
        return self._get_environment()

    def set_environment(self, key, value):
        self._set_environment(key, value)

    def get_license_path(self):
        return self._get_license_path()

    def get_arglist(self):
        # The return type is cast from a <StringList> to a regular Python list.
        return list(self._get_arglist())

    def get_argvalues(self):
        # The return type is cast from a <StringList> to a regular Python list.
        return list(self._get_argvalues())

    def set_arglist(self, args):
        return self._set_arglist(args)

    def clear_environment(self):
        self._clear_environment()

    def save(self):
        self._save()

    def free(self):
        self._free()

    def name(self):
        return self._get_name()

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        if self.name() != other.name():
            return False

        if [x for x in self.get_arglist()] != [x for x in other.get_arglist()]:
            return False

        if self.get_config_file() != other.get_config_file():
            return False

        if self.get_stderr_file() != other.get_stderr_file():
            return False

        if self.get_stdout_file() != other.get_stdout_file():
            return False

        return True
