import os
from dataclasses import dataclass

from ert._c_wrappers.config import ConfigContent
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.job_queue import EnvironmentVarlist, ExtJob, ExtJoblist


@dataclass
class SiteConfig:
    job_list: ExtJoblist
    var_list: EnvironmentVarlist

    @classmethod
    def _add_config_content(
        cls,
        license_root_path: str,
        site_config_content: ConfigContent,
        job_list: ExtJoblist,
        var_list: EnvironmentVarlist,
    ):

        if site_config_content.hasKey(ConfigKeys.INSTALL_JOB):
            for args in iter(site_config_content[ConfigKeys.INSTALL_JOB]):
                if not os.path.isfile(args[1]):
                    print(f"WARNING: Unable to locate job file {args}")
                    continue
                try:
                    new_job = ExtJob(
                        config_file=args[1],
                        private=False,
                        name=args[0],
                        license_root_path=license_root_path,
                    )
                    new_job.convertToCReference(None)
                    job_list.add_job(new_job.name(), new_job)
                except (ValueError, OSError):
                    print(f"WARNING: Unable to create job from {args}")

        if site_config_content.hasKey(ConfigKeys.INSTALL_JOB_DIRECTORY):
            for args in iter(site_config_content[ConfigKeys.INSTALL_JOB_DIRECTORY]):
                job_path = args[0]
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
                                license_root_path=license_root_path,
                            )
                            new_job.convertToCReference(None)
                            job_list.add_job(new_job.name(), new_job)
                        except (ValueError, OSError):
                            print(
                                f"WARNING: Unable to create job from {full_path} "
                                f"in directory {job_path}"
                            )

        site_environment_vars = site_config_content[ConfigKeys.SETENV]

        for elem in iter(site_environment_vars):
            var_list.setenv(elem[0], elem[1])

        site_paths = site_config_content["UPDATE_PATH"]

        for elem in iter(site_paths):
            var_list.update_path(elem[0], elem[1])

    @classmethod
    def from_config_content(
        cls, config_content: ConfigContent, site_config_content: ConfigContent
    ):
        license_root_path = None
        if site_config_content.hasKey(ConfigKeys.LICENSE_PATH):
            license_root_path = site_config_content.getValue(ConfigKeys.LICENSE_PATH)
        if config_content.hasKey(ConfigKeys.LICENSE_PATH):
            user_license_root_path = config_content.getValue(ConfigKeys.LICENSE_PATH)
            license_root_path_site = os.path.realpath(user_license_root_path)
            license_root_path = os.path.join(
                license_root_path_site, os.getenv("USER"), str(os.getpid())
            )

        ext_job_list = ExtJoblist()
        env_vars = EnvironmentVarlist()

        SiteConfig._add_config_content(
            license_root_path, config_content, ext_job_list, env_vars
        )

        SiteConfig._add_config_content(
            license_root_path, site_config_content, ext_job_list, env_vars
        )

        return SiteConfig(ext_job_list, env_vars)

    @classmethod
    def from_config_dict(cls, config_dict, site_config_content: ConfigContent):
        license_root_path = None
        if site_config_content.hasKey(ConfigKeys.LICENSE_PATH):
            license_root_path = site_config_content.getValue(ConfigKeys.LICENSE_PATH)
        if ConfigKeys.LICENSE_PATH in config_dict:
            user_license_root_path = config_dict.get(ConfigKeys.LICENSE_PATH)
            license_root_path_site = os.path.realpath(user_license_root_path)
            license_root_path = os.path.join(
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
                    license_root_path=license_root_path,
                )
                new_job.convertToCReference(None)
                ext_job_list.add_job(job[ConfigKeys.NAME], new_job)
            except (ValueError, OSError) as e:
                print(f"WARNING: Unable to create job from {job[ConfigKeys.PATH]}: {e}")

        for job_path in config_dict.get(ConfigKeys.INSTALL_JOB_DIRECTORY, []):
            if not os.path.isdir(job_path):
                print(f"WARNING: Unable to locate job directory {job_path}")
                continue
            files = os.listdir(job_path)
            print(files)
            for file_name in files:
                full_path = os.path.join(job_path, file_name)
                if os.path.isfile(full_path):
                    try:
                        new_job = ExtJob(
                            config_file=full_path,
                            private=False,
                            license_root_path=license_root_path,
                        )
                        new_job.convertToCReference(None)
                        ext_job_list.add_job(new_job.name(), new_job)
                    except (ValueError, OSError):
                        print(
                            f"WARNING: Unable to create job from {full_path}"
                            f" in directory {job_path}"
                        )

        ext_job_list.convertToCReference(None)

        # Create varlist
        env_vars = EnvironmentVarlist()
        dict_vars = config_dict.get(ConfigKeys.SETENV, [])

        for elem in dict_vars:
            env_vars.setenv(elem[ConfigKeys.NAME], elem[ConfigKeys.VALUE])

        dict_paths = config_dict.get("UPDATE_PATH", [])

        for elem in dict_paths:
            env_vars.update_path(elem[ConfigKeys.NAME], elem[ConfigKeys.VALUE])

        SiteConfig._add_config_content(
            license_root_path, site_config_content, ext_job_list, env_vars
        )

        return SiteConfig(ext_job_list, env_vars)
