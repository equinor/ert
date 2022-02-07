from typing import List, Optional

from ert3.config.plugins import TransformationConfigBase


class FileTransformationConfig(TransformationConfigBase):
    mime: str = ""


class SummaryTransformationConfig(TransformationConfigBase):
    smry_keys: Optional[List[str]] = None


class DirectoryTransformationConfig(TransformationConfigBase):
    pass

