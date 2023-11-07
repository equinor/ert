from enum import Enum
from typing import Any, List, Tuple, Union

from .context_values import ContextValue
from .file_context_token import FileContextToken

# The type of the leaf nodes in the Tree after transformation is done
Instruction = List[
    List[Union[FileContextToken, List[Tuple[FileContextToken, FileContextToken]]]]
]

Defines = List[List[str]]

Primitives = Union[float, bool, str, int, Enum]

MaybeWithContext = Union[ContextValue, Primitives, FileContextToken, List[Any]]
