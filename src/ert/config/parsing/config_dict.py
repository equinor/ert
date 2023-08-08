from typing import Dict, List, Union

from .context_values import ContextString, ContextValue

ConfigDict = Dict[
    ContextString,
    Union[ContextValue, List[ContextValue], List[List[ContextValue]]],
]
