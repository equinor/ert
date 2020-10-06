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
from res import ResPrototype
from res.enkf import ConfigKeys
from res.job_queue import ExtJob, ExtJoblist, EnvironmentVarlist


class SiteConfig(BaseCClass):
    TYPE_NAME = "site_config"
    _alloc = ResPrototype("void* site_config_alloc(config_content)", bind=False)
    _alloc_full = ResPrototype(
        "void* site_config_alloc_full(ext_joblist, env_varlist, int)", bind=False
    )
    _alloc_load_user_config = ResPrototype(
        "void* site_config_alloc_load_user_config(char*)", bind=False
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
    _get_umask = ResPrototype("int site_config_get_umask(site_config)")

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
                "Attempting to construct SiteConfig with multiple config objects"
            )

        if configs == 0:
            raise ValueError(
                "Attempting to construct SiteConfig with no config objects"
            )

        c_ptr = None
        if user_config_file is not None:
            if not os.path.isfile(user_config_file):
                raise IOError('No such configuration file "%s".' % user_config_file)
            c_ptr = self._alloc_load_user_config(user_config_file)

        elif config_content is not None:
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
                    print(
                        "WARNING: Unable to locate job file {}".format(
                            job[ConfigKeys.PATH]
                        )
                    )
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
                except:
                    print(
                        "WARNING: Unable to create job from {}".format(
                            job[ConfigKeys.PATH]
                        )
                    )

            for job_path in config_dict.get(ConfigKeys.INSTALL_JOB_DIRECTORY, []):
                if not os.path.isdir(job_path):
                    print("WARNING: Unable to locate job directory {}".format(job_path))
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
                        except:
                            print(
                                "WARNING: Unable to create job from {}".format(
                                    full_path
                                )
                            )

            ext_job_list.convertToCReference(None)

            # Create varlist)
            env_var_list = EnvironmentVarlist()
            for (var, value) in config_dict.get(ConfigKeys.SETENV, []):
                env_var_list[var] = value

            env_var_list.convertToCReference(None)
            umask = config_dict.get(ConfigKeys.UMASK)

            c_ptr = self._alloc_full(ext_job_list, env_var_list, umask)

        if c_ptr is None:
            raise ValueError("Failed to construct SiteConfig instance.")

        super(SiteConfig, self).__init__(c_ptr)

    def __repr__(self):
        return "Site Config {}".format(SiteConfig.getLocation())

    @property
    def config_file(self):
        return self._get_config_file()

    def get_installed_jobs(self):
        """ @rtype: ExtJoblist """
        return self._get_installed_jobs().setParent(self)

    def get_license_root_path(self):
        """ @rtype: str """
        return self._get_license_root_path()

    def set_license_root_pathmax_submit(self, path):
        self._set_license_root_path(path)

    @classmethod
    def getLocation(cls):
        """ @rtype: str """
        return cls._get_location()

    def free(self):
        self._free()

    @property
    def umask(self):
        return self._get_umask()

    def __eq__(self, other):
        if self.umask != other.umask:
            return False

        self_job_list = self.get_installed_jobs()
        other_job_list = other.get_installed_jobs()

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
