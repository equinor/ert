import sys
from pathlib import Path

from everest.docs.generate_docs_from_config_spec import generate_docs_pydantic_to_rst

GENERATED_FILE = Path("docs") / "everest" / "config_generated.rst"

parts = GENERATED_FILE.parts
while parts and not Path(*parts).exists():
    parts = parts[1:]
if not parts:
    print(
        f"""The file to update ('{GENERATED_FILE}') was not found.
The generator must be run from one of the parent directories of
'{GENERATED_FILE.parts[-1]}' within the ERT source directory.
Please change the current directory accordingly and re-run."""
    )
    sys.exit()

docs_content = generate_docs_pydantic_to_rst()
with Path(*parts).open("w", encoding="utf-8") as fp:
    fp.write(docs_content)

print(f"Generated docs written to {GENERATED_FILE}")
print("Please commit the generated file.")
