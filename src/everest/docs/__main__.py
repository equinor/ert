from pathlib import Path

from everest.docs.generate_docs_from_config_spec import generate_docs_pydantic_to_rst

committed_file = (
    Path(__file__).parents[3] / "docs" / "everest" / "config_generated.rst"
).resolve()


print(f"Writing new docs contents to {committed_file}")
docs_content = generate_docs_pydantic_to_rst()
with open(committed_file, "w", encoding="utf-8") as fp:
    fp.write(docs_content)

print(f"Writing updated docs to {committed_file} complete, you may now commit it.")
