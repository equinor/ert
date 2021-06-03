import logging
import logging.config
import yaml
import os

LOGGING_CONFIG = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "logger.conf")
)
with open(LOGGING_CONFIG) as conf_file:
    logging.config.dictConfig(yaml.safe_load(conf_file))


def get_logger(name):
    return logging.getLogger(name)
