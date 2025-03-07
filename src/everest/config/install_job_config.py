from pydantic import BaseModel, Field


class InstallJobConfig(BaseModel, extra="forbid"):
    name: str = Field(description="name of the installed job")
    source: str = Field(description="source file of the ert job")
