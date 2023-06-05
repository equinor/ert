from typing import Dict, List, Optional, Union

from .context_values import ContextString, ContextValue
from .error_info import WarningInfo


class ConfigDict(
    Dict[
        ContextString,
        Union[ContextValue, List[ContextValue], List[List[ContextValue]]],
    ]
):
    warning_infos: Optional[List[WarningInfo]] = None
