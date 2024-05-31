from ert.config import ErtConfig
from ert.config.responses.response_config import ResponseConfigWithLifecycleHooks


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
