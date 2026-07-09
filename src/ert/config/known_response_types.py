from ert.config.seismic_config import SeismicConfig

from .breakthrough_config import BreakthroughConfig
from .everest_response import EverestConstraintsConfig, EverestObjectivesConfig
from .gen_data_config import GenDataConfig
from .rft_config import RFTConfig
from .summary_config import SummaryConfig

KnownErtSimulationResponseTypes = (
    SummaryConfig | GenDataConfig | RFTConfig | SeismicConfig
)
KnownErtDerivedResponseTypes = BreakthroughConfig
KnownErtResponseTypes = KnownErtSimulationResponseTypes | KnownErtDerivedResponseTypes
KNOWN_ERT_SIMULATION_RESPONSE_TYPES = (
    SummaryConfig,
    GenDataConfig,
    RFTConfig,
    SeismicConfig,
)
KNOWN_ERT_DERIVED_RESPONSE_TYPES = (BreakthroughConfig,)
KNOWN_ERT_RESPONSE_TYPES = (
    KNOWN_ERT_SIMULATION_RESPONSE_TYPES + KNOWN_ERT_DERIVED_RESPONSE_TYPES
)
KnownResponseTypes = (
    KnownErtResponseTypes | EverestConstraintsConfig | EverestObjectivesConfig
)
