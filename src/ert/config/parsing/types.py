from enum import Enum
from typing import Any

from .context_values import ContextValue
from .file_context_token import FileContextToken

# The type of the leaf nodes in the Tree after transformation is done
Instruction = list[
    list[FileContextToken | list[tuple[FileContextToken, FileContextToken]]]
]

Defines = list[list[str]]

Primitives = float | bool | str | int | Enum

MaybeWithContext = ContextValue | Primitives | FileContextToken | list[Any]
