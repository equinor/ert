from textwrap import dedent

from pydantic import BaseModel, Field


class InstallTemplateConfig(BaseModel, extra="forbid"):
    template: str = Field(
        description=dedent(
            """
            The jinja2 template file.
            """
        )
    )
    output_file: str = Field(
        description=dedent(
            """
            The name of the output file.
            """
        )
    )
    extra_data: str | None = Field(
        description=dedent(
            """
            Extra input files.

            The content of each extra JSON or YAML file is exposed to the jinja2
            renderer, using the name of the file as a variable name.
            """
        ),
        default=None,
    )
