import logging
import logging.config
import pathlib

import yaml

LOGGING_CONFIG = pathlib.Path(__file__).parent.resolve() / "logger.conf"

with open(LOGGING_CONFIG) as conf_file:
    logging.config.dictConfig(yaml.safe_load(conf_file))


def get_logger(name):
    return logging.getLogger(name)
