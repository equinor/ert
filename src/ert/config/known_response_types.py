from ert.config.seismic_config import SeismicConfig

from .everest_response import EverestConstraintsConfig, EverestObjectivesConfig
from .gen_data_config import GenDataConfig
from .rft_config import RFTConfig
from .summary_config import SummaryConfig

KnownErtResponseTypes = SummaryConfig | GenDataConfig | RFTConfig | SeismicConfig
KNOWN_ERT_RESPONSE_TYPES = (
    SummaryConfig,
    GenDataConfig,
    RFTConfig,
    SeismicConfig,
)
KnownResponseTypes = (
    KnownErtResponseTypes | EverestConstraintsConfig | EverestObjectivesConfig
)
