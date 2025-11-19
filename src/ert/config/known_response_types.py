from .gen_data_config import GenDataConfig
from .summary_config import SummaryConfig

KnownResponseTypes = SummaryConfig | GenDataConfig
KNOWN_RESPONSE_TYPES = (
    SummaryConfig,
    GenDataConfig,
)
