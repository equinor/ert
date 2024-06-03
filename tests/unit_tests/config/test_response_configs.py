from typing import List, Tuple

from ert.config import ErtConfig
from ert.config.responses.response_config import (
    ResponseConfigWithLifecycleHooks,
)


def test_that_custom_response_type_is_parsed_into_config_when_invoked(tmp_path):
    (tmp_path / "test.ert").write_text(
        """
        NUM_REALIZATIONS 1
        QUEUE_SYSTEM LOCAL

        DUMMY_RESPONSE(<SRC>=DUMMY)
        DUMMY_OBSERVATION(<SRC>=DUMMY.csv,<RESPONSE_TYPE>=DUMMY)
    """
    )

    class DummyResponseConfig(ResponseConfigWithLifecycleHooks):
        @classmethod
        def response_type(cls) -> str:
            return "DUMMY"

        @classmethod
        def ert_config_response_keyword(cls) -> str:
            return "DUMMY_RESPONSE"

        @classmethod
        def ert_config_observation_keyword(cls) -> str:
            return "DUMMY_OBSERVATION"

    config_dict = ErtConfig.with_plugins(
        response_type_classes=[DummyResponseConfig]
    ).read_user_config(tmp_path / "test.ert")

    assert "DUMMY_RESPONSE" in config_dict
    assert "DUMMY_OBSERVATION" in config_dict

    assert config_dict["DUMMY_RESPONSE"] == [["(<SRC>=DUMMY)"]]

    observation_config_list = config_dict["DUMMY_OBSERVATION"][0][0]
    parsed_kwargs = DummyResponseConfig.parse_kwargs_from_config_list(
        observation_config_list
    )
    assert parsed_kwargs == {"<SRC>": "DUMMY.csv", "<RESPONSE_TYPE>": "DUMMY"}


def test_that_custom_response_type_is_not_parsed_into_config_when_not_invoked(tmp_path):
    (tmp_path / "test.ert").write_text(
        """
        NUM_REALIZATIONS 1
        QUEUE_SYSTEM LOCAL

    """
    )

    class DummyResponseConfig(ResponseConfigWithLifecycleHooks):
        @classmethod
        def response_type(cls) -> str:
            return "DUMMY"

        @classmethod
        def ert_config_response_keyword(cls) -> str:
            return "DUMMY_RESPONSE"

        @classmethod
        def ert_config_observation_keyword(cls) -> str:
            return "DUMMY_OBSERVATION"

    config_dict = ErtConfig.with_plugins(
        response_type_classes=[DummyResponseConfig]
    ).read_user_config(tmp_path / "test.ert")

    assert "DUMMY_RESPONSE" not in config_dict
    assert "DUMMY_OBSERVATION" not in config_dict


def test_that_custom_response_appears_in_ensemble_config(tmp_path):
    (tmp_path / "test.ert").write_text(
        """
        NUM_REALIZATIONS 1
        QUEUE_SYSTEM LOCAL

        DUMMY_RESPONSE(<SRC>=DUMMY)
        DUMMY_RESPONSE(<SRC>=DUMMY2,<NAME>=D2)
        DUMMY_OBSERVATION(<SRC>=DUMMY0.csv,<RESPONSE_TYPE>=DUMMY)
        DUMMY_OBSERVATION(<SRC>=DUMMY1.csv,<RESPONSE_TYPE>=DUMMY,<OBS_NAME>=DMY1)
        DUMMY_OBSERVATION(<SRC>=DUMMY2.csv,<RESPONSE_NAME>=D2,<OBS_NAME>=DMY2)
    """
    )

    class DummyResponseConfig(ResponseConfigWithLifecycleHooks):
        def parse_response_from_config(
            self, config_list: List[Tuple[str, str]]
        ) -> None:
            pass

        def parse_observation_from_config(
            self, config_list: List[Tuple[str, str]]
        ) -> None:
            pass

        def parse_response_from_runpath(self, run_path: str) -> str:
            pass

        @classmethod
        def response_type(cls) -> str:
            return "DUMMY"

        @classmethod
        def ert_config_response_keyword(cls) -> str:
            return "DUMMY_RESPONSE"

        @classmethod
        def ert_config_observation_keyword(cls) -> str:
            return "DUMMY_OBSERVATION"

    ert_config = ErtConfig.with_plugins(response_types=[DummyResponseConfig]).from_file(
        tmp_path / "test.ert"
    )

    ens_config = ert_config.ensemble_config
    obs_configs = ens_config.observation_configs
    resp_configs = ens_config.response_configs

    assert set(obs_configs.keys()) == {
        "DMY1",
        "DMY2",
        "obs(<SRC>=DUMMY0.csv,<RESPONSE_TYPE>=DUMMY)",
    }

    assert set(resp_configs.keys()) == {"D2", "response(<SRC>=DUMMY)"}

    d1c = resp_configs["response(<SRC>=DUMMY)"]
    assert d1c.name == "response(<SRC>=DUMMY)"
    assert d1c.line_from_ert_config == ["(<SRC>=DUMMY)"]
    assert d1c.src == "DUMMY"

    d2c = resp_configs["D2"]
    assert d2c.line_from_ert_config == ["(<SRC>=DUMMY2,<NAME>=D2)"]
    assert d2c.name == "D2"
    assert d2c.src == "DUMMY2"

    o0c = obs_configs["obs(<SRC>=DUMMY0.csv,<RESPONSE_TYPE>=DUMMY)"]
    assert o0c.obs_name == "obs(<SRC>=DUMMY0.csv,<RESPONSE_TYPE>=DUMMY)"
    assert o0c.response_type == "DUMMY"
    assert o0c.response_name is None
    assert o0c.src == "DUMMY0.csv"

    o1c = obs_configs["DMY1"]
    assert o1c.obs_name == "DMY1"
    assert o1c.response_type == "DUMMY"
    assert o1c.response_name is None
    assert o1c.src == "DUMMY1.csv"

    o2c = obs_configs["DMY2"]
    assert o2c.obs_name == "DMY2"
    assert o2c.response_type is None
    assert o2c.response_name == "D2"
    assert o2c.src == "DUMMY2.csv"
