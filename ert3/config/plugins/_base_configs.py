from pydantic import BaseModel


class _BasePluginConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


class TransformationConfigBase(_BasePluginConfig):
    """Common config for all Transformations"""

    location: str
