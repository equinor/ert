from .everest_config import EverestConstraintsConfig, EverestObjectivesConfig
from .gen_data_config import GenDataConfig
from .summary_config import SummaryConfig

KnownErtResponseTypes = SummaryConfig | GenDataConfig
KNOWN_ERT_RESPONSE_TYPES = (
    SummaryConfig,
    GenDataConfig,
)
KnownResponseTypes = (
    KnownErtResponseTypes | EverestConstraintsConfig | EverestObjectivesConfig
)
