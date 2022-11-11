import logging
from dataclasses import dataclass

from ert._c_wrappers.config import ConfigContent
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.job_queue import EnvironmentVarlist

logger = logging.getLogger(__name__)


@dataclass
class SiteConfig:
    env_vars: EnvironmentVarlist

    @classmethod
    def _add_config_content(
        cls,
        config_content: ConfigContent,
        env_vars: EnvironmentVarlist,
    ):
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
        env_vars = EnvironmentVarlist()

        SiteConfig._add_config_content(site_config_content, env_vars)
        SiteConfig._add_config_content(user_config_content, env_vars)

        return SiteConfig(env_vars)

    @classmethod
    def from_config_dict(cls, config_dict, site_config_content: ConfigContent = None):
        env_vars = EnvironmentVarlist()

        if site_config_content is not None:
            SiteConfig._add_config_content(site_config_content, env_vars)

        # fill in varlist
        dict_vars = config_dict.get(ConfigKeys.SETENV, [])

        for elem in dict_vars:
            env_vars.setenv(elem[ConfigKeys.NAME], elem[ConfigKeys.VALUE])

        dict_paths = config_dict.get("UPDATE_PATH", [])

        for elem in dict_paths:
            env_vars.update_path(elem[ConfigKeys.NAME], elem[ConfigKeys.VALUE])

        return SiteConfig(env_vars)
