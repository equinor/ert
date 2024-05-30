from typing import List, Tuple

from .response_config import ResponseConfigWithLifecycleHooks


class SummaryConfigWithHooks(ResponseConfigWithLifecycleHooks):
    @classmethod
    def ert_config_response_keyword(cls) -> str:
        return "SUMMARY"

    @classmethod
    def ert_config_observation_keyword(cls) -> str:
        pass

    def parse_response_from_config(self, config_list: List[Tuple[str, str]]) -> None:
        pass

    def parse_observation_from_config(self, config_list: List[Tuple[str, str]]) -> None:
        pass

    def parse_response_from_runpath(self, run_path: str) -> str:
        pass
