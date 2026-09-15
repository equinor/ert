import logging.config
from logging import Logger


def log_and_print(msg: str, logger: Logger, level: int = logging.INFO) -> None:
    logger.log(level, msg)
    print(msg)
