import logging
from typing import Any, Dict, Optional

from ert._c_wrappers.enkf.ert_config import ErtConfig
from ert._c_wrappers.enkf.ert_config import site_config_location as site_loc

logger = logging.getLogger(__name__)


def site_config_location():
    return site_loc()


class ResConfig(ErtConfig):
    def __new__(
        cls, user_config_file: Optional[str] = None, config_dict: Dict[str, Any] = None
    ):
        if user_config_file is not None:
            logger.warning(
                "Old ResConfig __init__ called. "
                "Should use ResConfig.from_file() instead"
            )
            return ErtConfig.from_file(user_config_file)
        else:
            logger.warning(
                "Old ResConfig __init__ called. "
                "Should use ResConfig.from_dict() instead"
            )
            return ErtConfig.from_dict(config_dict)
