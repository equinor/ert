import yaml
import collections

from res.enkf import ErtScript
from res.enkf.jobs.scaling import scaling_job


class EnkfScalingJob(ErtScript):

    def run(self, job_config_file):
        user_config = load_yaml(job_config_file)
        user_config = _insert_default_group(user_config)
        for job_config in user_config:
            scaling_job.scaling_job(self.ert(), job_config)


def load_yaml(fname):
    with open(fname, "r") as fin:
        config = yaml.safe_load(fin)
    return config


def _insert_default_group(value):
    if isinstance(value, collections.Mapping):
        return [value]
    else:
        return value
