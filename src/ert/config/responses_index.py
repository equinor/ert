from typing import Dict, Type

from .gen_data_config import GenDataConfig
from .response_config import ResponseConfig
from .summary_config import SummaryConfig

responses_index: Dict[str, Type[ResponseConfig]] = {
    "SummaryConfig": SummaryConfig,
    "GenDataConfig": GenDataConfig,
}
