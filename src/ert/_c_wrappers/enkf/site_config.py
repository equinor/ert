import logging
import os
from dataclasses import dataclass

from ert._c_wrappers.config import ConfigContent
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.job_queue import EnvironmentVarlist, ExtJob, ExtJoblist

logger = logging.getLogger(__name__)


@dataclass
class SiteConfig:
    job_list: ExtJoblist
    env_vars: EnvironmentVarlist

    @classmethod
    def _add_job(cls, job_list, license_root_path, job_path, job_name=None):
        if not os.path.isfile(job_path):
            logger.warning(f"Unable to locate job file {job_path}")
            return
        new_job = ExtJob(
            config_file=job_path,
            private=False,
            name=job_name,
            license_root_path=license_root_path,
        )
        job_list.add_job(new_job.name(), new_job)

    @classmethod
    def _add_config_content(
        cls,
        license_root_path: str,
        config_content: ConfigContent,
        job_list: ExtJoblist,
        env_vars: EnvironmentVarlist,
    ):

        if config_content.hasKey(ConfigKeys.INSTALL_JOB):
            for name, config in iter(config_content[ConfigKeys.INSTALL_JOB]):
                cls._add_job(job_list, license_root_path, config, name)

        if config_content.hasKey(ConfigKeys.INSTALL_JOB_DIRECTORY):
            for args in iter(config_content[ConfigKeys.INSTALL_JOB_DIRECTORY]):
                job_path = args[0]
                if not os.path.isdir(job_path):
                    logger.warning(f"Unable to locate job directory {job_path}")
                    continue
                files = os.listdir(job_path)
                for file_name in files:
                    full_path = os.path.join(job_path, file_name)
                    if os.path.isfile(full_path):
                        cls._add_job(job_list, license_root_path, full_path)

        environment_vars = config_content[ConfigKeys.SETENV]

        for key, value in iter(environment_vars):
            env_vars.setenv(key, value)

        paths = config_content["UPDATE_PATH"]

        for key, value in iter(paths):
            env_vars.update_path(key, value)

    @classmethod
    def from_config_content(
        cls, user_config_content: ConfigContent, site_config_content: ConfigContent
    ):
        license_root_path = None
        if site_config_content.hasKey(ConfigKeys.LICENSE_PATH):
            license_root_path = site_config_content.getValue(ConfigKeys.LICENSE_PATH)
        if user_config_content.hasKey(ConfigKeys.LICENSE_PATH):
            user_license_root_path = user_config_content.getValue(
                ConfigKeys.LICENSE_PATH
            )
            license_root_path_site = os.path.realpath(user_license_root_path)
            license_root_path = os.path.join(
                license_root_path_site, os.getenv("USER"), str(os.getpid())
            )

        ext_job_list = ExtJoblist()
        env_vars = EnvironmentVarlist()

        SiteConfig._add_config_content(
            license_root_path, site_config_content, ext_job_list, env_vars
        )

        SiteConfig._add_config_content(
            license_root_path, user_config_content, ext_job_list, env_vars
        )

        return SiteConfig(ext_job_list, env_vars)

    @classmethod
    def from_config_dict(cls, config_dict, site_config_content: ConfigContent = None):
        license_root_path = None
        if site_config_content is not None and site_config_content.hasKey(
            ConfigKeys.LICENSE_PATH
        ):
            license_root_path = site_config_content.getValue(ConfigKeys.LICENSE_PATH)
        if ConfigKeys.LICENSE_PATH in config_dict:
            user_license_root_path = config_dict.get(ConfigKeys.LICENSE_PATH)
            license_root_path_site = os.path.realpath(user_license_root_path)
            license_root_path = os.path.join(
                license_root_path_site, os.getenv("USER"), str(os.getpid())
            )

        job_list = ExtJoblist()
        env_vars = EnvironmentVarlist()

        if site_config_content is not None:
            SiteConfig._add_config_content(
                license_root_path, site_config_content, job_list, env_vars
            )

        # fill in joblist
        for job in config_dict.get(ConfigKeys.INSTALL_JOB, []):
            cls._add_job(
                job_list,
                license_root_path,
                os.path.abspath(job[ConfigKeys.PATH]),
                job[ConfigKeys.NAME],
            )

        for job_path in config_dict.get(ConfigKeys.INSTALL_JOB_DIRECTORY, []):
            if not os.path.isdir(job_path):
                logger.warning(f"Unable to locate job directory {job_path}")
                continue
            files = os.listdir(job_path)
            for file_name in files:
                full_path = os.path.abspath(os.path.join(job_path, file_name))
                cls._add_job(job_list, license_root_path, full_path)

        # fill in varlist
        dict_vars = config_dict.get(ConfigKeys.SETENV, [])

        for elem in dict_vars:
            env_vars.setenv(elem[ConfigKeys.NAME], elem[ConfigKeys.VALUE])

        dict_paths = config_dict.get("UPDATE_PATH", [])

        for elem in dict_paths:
            env_vars.update_path(elem[ConfigKeys.NAME], elem[ConfigKeys.VALUE])

        return SiteConfig(job_list, env_vars)
