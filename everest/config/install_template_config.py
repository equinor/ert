from typing import Optional

from pydantic import BaseModel, Field


class InstallTemplateConfig(BaseModel, extra="forbid"):  # type: ignore
    template: str = Field()  # existing file
    output_file: str = Field()  # path
    extra_data: Optional[str] = Field(default=None)  # path
