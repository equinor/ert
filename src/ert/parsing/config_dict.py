from typing import Dict, List, Optional, Union

from .error_info import WarningInfo


class ConfigDict(Dict[str, Union[str, List[str], List[List[str]]]]):
    warning_infos: Optional[List[WarningInfo]] = None
