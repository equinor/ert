from typing import List, Tuple, Union

from .lark_parser_file_context_token import FileContextToken
from .lark_parser_primitive_tokens import PrimitiveWithContext

# The type of the leaf nodes in the Tree after transformation is done
Instruction = List[
    List[Union[FileContextToken, List[Tuple[FileContextToken, FileContextToken]]]]
]

Defines = List[List[str]]


Primitives = Union[float, bool, str, int]
MaybeWithToken = MaybeWithKeywordToken = Union[PrimitiveWithContext, Primitives]
