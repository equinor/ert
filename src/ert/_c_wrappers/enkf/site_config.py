#  Copyright (C) 2012  Equinor ASA, Norway.
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

import os

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.job_queue import EnvironmentVarlist, ExtJob, ExtJoblist


class SiteConfig(BaseCClass):
    TYPE_NAME = "site_config"
    _alloc = ResPrototype("void* site_config_alloc(config_content)", bind=False)
    _alloc_full = ResPrototype(
        "void* site_config_alloc_full(ext_joblist, env_varlist)", bind=False
    )
    _free = ResPrototype("void site_config_free( site_config )")
    _get_installed_jobs = ResPrototype(
        "ext_joblist_ref site_config_get_installed_jobs(site_config)"
    )
    _get_license_root_path = ResPrototype(
        "char* site_config_get_license_root_path(site_config)"
    )
    _set_license_root_path = ResPrototype(
        "void site_config_set_license_root_path(site_config, char*)"
    )
    _get_location = ResPrototype("char* site_config_get_location()", bind=False)
    _get_config_file = ResPrototype("char* site_config_get_config_file(site_config)")
    _get_env_var_list = ResPrototype(
        "env_varlist_ref site_config_get_env_varlist(site_config)"
    )

    def __init__(self, config_content=None, config_dict=None):

        configs = sum(1 for x in [config_content, config_dict] if x is not None)

        if configs > 1:
            raise ValueError(
                "Attempting to construct SiteConfig with multiple config objects"
            )

        if configs == 0:
            raise ValueError(
                "Attempting to construct SiteConfig with no config objects"
            )

        c_ptr = None
        if config_content is not None:
            c_ptr = self._alloc(config_content)

        elif config_dict is not None:
            __license_root_path = None
            if ConfigKeys.LICENSE_PATH in config_dict:
                license_root_path = config_dict.get(ConfigKeys.LICENSE_PATH)
                license_root_path_site = os.path.realpath(license_root_path)
                __license_root_path = os.path.join(
                    license_root_path_site, os.getenv("USER"), str(os.getpid())
                )

            # Create joblist
            ext_job_list = ExtJoblist()
            for job in config_dict.get(ConfigKeys.INSTALL_JOB, []):
                if not os.path.isfile(job[ConfigKeys.PATH]):
                    print(f"WARNING: Unable to locate job file {job[ConfigKeys.PATH]}")
                    continue
                try:
                    new_job = ExtJob(
                        config_file=job[ConfigKeys.PATH],
                        private=False,
                        name=job[ConfigKeys.NAME],
                        license_root_path=__license_root_path,
                    )
                    new_job.convertToCReference(None)
                    ext_job_list.add_job(job[ConfigKeys.NAME], new_job)
                except (ValueError, OSError):
                    print(f"WARNING: Unable to create job from {job[ConfigKeys.PATH]}")

            for job_path in config_dict.get(ConfigKeys.INSTALL_JOB_DIRECTORY, []):
                if not os.path.isdir(job_path):
                    print(f"WARNING: Unable to locate job directory {job_path}")
                    continue
                files = os.listdir(job_path)
                for file_name in files:
                    full_path = os.path.join(job_path, file_name)
                    if os.path.isfile(full_path):
                        try:
                            new_job = ExtJob(
                                config_file=full_path,
                                private=False,
                                license_root_path=__license_root_path,
                            )
                            new_job.convertToCReference(None)
                            ext_job_list.add_job(new_job.name(), new_job)
                        except (ValueError, OSError):
                            print(f"WARNING: Unable to create job from {full_path}")

            ext_job_list.convertToCReference(None)

            # Create varlist
            environment_vars = config_dict.get(ConfigKeys.SETENV, [])
            env_var_list = EnvironmentVarlist(
                vars={
                    elem[ConfigKeys.NAME]: elem[ConfigKeys.VALUE]
                    for elem in environment_vars
                }
            )

            env_var_list.convertToCReference(None)

            c_ptr = self._alloc_full(ext_job_list, env_var_list)

        if c_ptr is None:
            raise ValueError("Failed to construct SiteConfig instance.")

        super().__init__(c_ptr)

    def __repr__(self):
        return (
            "SiteConfig(config_dict={"
            + f"{ConfigKeys.INSTALL_JOB}: ["
            + ", ".join(str(job) for job in self.get_installed_jobs())
            + "],"
            + f"{ConfigKeys.SETENV}: {self._get_env_var_list()}"
            + "})"
        )

    @property
    def config_file(self):
        return self._get_config_file()

    def get_installed_jobs(self):
        return self._get_installed_jobs().setParent(self)

    def get_license_root_path(self) -> str:
        return self._get_license_root_path()

    def set_license_root_pathmax_submit(self, path):
        self._set_license_root_path(path)

    @classmethod
    def getLocation(cls) -> str:
        return cls._get_location()

    def free(self):
        self._free()

    def __eq__(self, other):
        self_job_list = self.get_installed_jobs()
        other_job_list = other.get_installed_jobs()

        if self._get_env_var_list() != other._get_env_var_list():
            return False

        if set(other_job_list.getAvailableJobNames()) != set(
            self_job_list.getAvailableJobNames()
        ):
            return False

        if len(other_job_list.getAvailableJobNames()) != len(
            self_job_list.getAvailableJobNames()
        ):
            return False

        for job_name in other_job_list.getAvailableJobNames():

            if (
                other_job_list[job_name].get_config_file()
                != self_job_list[job_name].get_config_file()
            ):
                return False

            if (
                other_job_list[job_name].get_stderr_file()
                != self_job_list[job_name].get_stderr_file()
            ):
                return False

            if (
                other_job_list[job_name].get_stdout_file()
                != self_job_list[job_name].get_stdout_file()
            ):
                return False
        return True
