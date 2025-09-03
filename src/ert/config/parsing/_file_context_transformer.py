from __future__ import annotations

from lark import (
    Token,
    Transformer,
    Tree,
)

from .types import FileContextToken


class FileContextTransformer(Transformer[Token, Tree[FileContextToken]]):
    """Adds filename to each token,
    to ensure we have enough context for error messages"""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(visit_tokens=True)

    def __default_token__(self, token: Token) -> FileContextToken:
        return FileContextToken(token, self.filename)
