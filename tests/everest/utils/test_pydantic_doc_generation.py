from everest.docs.generate_docs_from_config_spec import generate_docs_pydantic_to_rst
from tests.everest.utils import relpath


def test_generated_doc():
    error_msg = """
    The generated documentation for the configuration file is
    out of date. Run `python -m everest.docs` to re-generate.
    """

    committed_file = relpath("..", "..", "docs", "everest", "config_generated.rst")
    with open(committed_file, "r", encoding="utf-8") as fp:
        committed_text = fp.read()

    generated_rst = generate_docs_pydantic_to_rst()

    assert committed_text.strip() == generated_rst.strip(), error_msg
