import logging

from ert.config import ExtParamConfig


def test_ext_param_logs_parameters_on_instantiation(caplog):
    caplog.set_level(logging.INFO)
    ExtParamConfig(
        name="name",
        input_keys=["keys"],
        output_file="foo.json",
    )
    print(caplog.text)
