from .everest_response import EverestConstraintsConfig, EverestObjectivesConfig
from .gen_data_config import GenDataConfig
from .rft_config import RFTConfig
from .seismic_attribute_config import SeismicAttributeConfig
from .summary_config import SummaryConfig

KnownErtResponseTypes = (
    SummaryConfig | GenDataConfig | RFTConfig | SeismicAttributeConfig
)
KNOWN_ERT_RESPONSE_TYPES = (
    SummaryConfig,
    GenDataConfig,
    RFTConfig,
    SeismicAttributeConfig,
)
KnownResponseTypes = (
    KnownErtResponseTypes | EverestConstraintsConfig | EverestObjectivesConfig
)
