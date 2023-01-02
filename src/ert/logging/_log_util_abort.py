import logging
import traceback


def _log_util_abort(
    filename: str, lineno: int, function: str, message: str, backtrace: str
) -> None:
    """
    When this function is called we are in util_abort, so will not be able to recover
    so we flush the logs to make sure we are able to propagate them.
    """

    class UtilAbort(Exception):
        pass

    logger = logging.getLogger(__name__)
    try:
        # This might deserve a comment. The reason for raising an exception and
        # immediately catching it is because logger.exception expects to be called
        # inside a try .. except block, however no python exception has occured. Instead
        # of manually creating the traceback a custom exception is raised.
        raise UtilAbort(message)
    except UtilAbort:
        logger.exception(
            f"C trace:\n{backtrace} \nwith message: {message} \nfrom file: {filename} "
            f"in {function} at line: {lineno}"
            f"\n\nPython backtrace:\n{''.join(traceback.format_stack())}"
        )
    for handle in logger.handlers:
        handle.flush()
    logging.shutdown()
