from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ert.config.breakthrough_config import BreakthroughConfig
from ert.config.gen_data_config import GenDataConfig
from ert.config.response_config import ResponseConfig
from ert.config.rft_config import RFTConfig
from ert.config.seismic_config import SeismicConfig
from ert.config.summary_config import SummaryConfig

if TYPE_CHECKING:
    from .local_ensemble import LocalEnsemble

logger = logging.getLogger(__name__)


@dataclass
class GenDataMatchConfig:
    pass


@dataclass
class SummaryMatchConfig:
    pass


@dataclass
class BreakthroughMatchConfig:
    pass


@dataclass
class RFTMatchConfig:
    pass


@dataclass
class SeismicMatchConfig:
    pass


@dataclass
class MatchConfig:
    rft: RFTMatchConfig = field(default_factory=RFTMatchConfig)
    seismic: SeismicMatchConfig = field(default_factory=SeismicMatchConfig)
    gen_data: GenDataMatchConfig = field(default_factory=GenDataMatchConfig)
    summary: SummaryMatchConfig = field(default_factory=SummaryMatchConfig)
    breakthrough: BreakthroughMatchConfig = field(
        default_factory=BreakthroughMatchConfig
    )

    special_match_keys: list[str] | None = None


class ReservoirDataMatchStrategy[ConcreteResponseConfig: ResponseConfig](ABC):
    ensemble: LocalEnsemble
    response_cls: ConcreteResponseConfig
    match_config: MatchConfig
    match_keys: list[str]

    def __init__(
        self,
        ensemble: LocalEnsemble,
        response_cls: ConcreteResponseConfig,
        match_config: MatchConfig,
    ) -> None:
        self.ensemble = ensemble
        self.response_cls = response_cls
        self.match_config = match_config
        if self.match_config.special_match_keys is not None:
            self.match_keys = self.match_config.special_match_keys
        else:
            self.match_keys = ["response_key", *response_cls.match_key]


class GenDataMatchStrategy(ReservoirDataMatchStrategy[GenDataConfig]):
    pass


class SummaryMatchStrategy(ReservoirDataMatchStrategy[SummaryConfig]):
    pass


class BreakthroughMatchStrategy(ReservoirDataMatchStrategy[BreakthroughConfig]):
    pass


class RFTMatchStrategy(ReservoirDataMatchStrategy[RFTConfig]):
    pass


class SeismicMatchStrategy(ReservoirDataMatchStrategy[SeismicConfig]):
    pass


MatchStrategyType = (
    GenDataMatchStrategy
    | SummaryMatchStrategy
    | BreakthroughMatchStrategy
    | RFTMatchStrategy
    | SeismicMatchStrategy
)


def matching_strategy(
    ensemble: LocalEnsemble,
    response_cls: ResponseConfig,
    match_config: MatchConfig,
) -> MatchStrategyType:
    """Get strategy for reservoir data matching for provided ResponseConfig."""
    data_map: dict[str, type] = {
        "gen_data": GenDataMatchStrategy,
        "summary": SummaryMatchStrategy,
        "breakthrough": BreakthroughMatchStrategy,
        "rft": RFTMatchStrategy,
        "seismic": SeismicMatchStrategy,
    }
    strategy_cls = data_map[response_cls.type]
    return strategy_cls(ensemble, response_cls, match_config)
