import os

from tests.everest.utils import relpath

from everest.docs.generate_docs_from_config_spec import generate_docs_pydantic_to_rst

committed_file = os.path.abspath(
    relpath("..", "..", "docs", "everest", "config_generated.rst")
)

print(f"Writing new docs contents to {committed_file}")
docs_content = generate_docs_pydantic_to_rst()
with open(committed_file, "w", encoding="utf-8") as fp:
    fp.write(docs_content)

print(f"Writing updated docs to {committed_file} complete, you may now commit it.")
