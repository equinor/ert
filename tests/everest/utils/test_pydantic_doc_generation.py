import sys

import pytest

from everest.docs.generate_docs_from_config_spec import generate_docs_pydantic_to_rst
from tests.everest.utils import relpath


@pytest.mark.skipif(sys.version_info < (3, 10), reason="python version less than 3.11")
def test_generated_doc():
    """Python 3.9 and below interprets annotations differently than 3.11 and above.
    3.8 will also soon be dropped so it's better to support only the updated interpretation.
    thus I limit this test to run for versions with the updated annotations handler
    """
    error_msg = """
    The generated documentation for the configuration file is
    out of date. Run `python -m everest.docs` to re-generate.
    """

    committed_file = relpath("..", "..", "docs", "everest", "config_generated.rst")
    with open(committed_file, "r", encoding="utf-8") as fp:
        committed_text = fp.read()

    generated_rst = generate_docs_pydantic_to_rst()

    assert committed_text.strip() == generated_rst.strip(), error_msg
