import mimetypes
from typing import Any, Callable

import ert

DEFAULT_RECORD_MIME_TYPE: str = "application/octet-stream"


def _ensure_mime(val: str, **kwargs: Any) -> str:
    """Guess on mime based on the value of basis_for_guess."""
    if val:
        if not ert.serialization.has_serializer(val):
            raise ValueError(f"{val} is not a valid mime.")
        return val

    basis = kwargs["basis_for_guess"]
    values = kwargs["values"]
    if basis is None:
        raise ValueError("no basis for guess")
    guess = mimetypes.guess_type(str(values.get(basis, "")))[0]
    if guess:
        if not ert.serialization.has_serializer(guess):
            return DEFAULT_RECORD_MIME_TYPE
        return guess
    return DEFAULT_RECORD_MIME_TYPE


def ensure_mime(basis_for_guess: str) -> Callable[[Any, Any], str]:
    """Create a mime validator that will guess a mime, or use a default one, based on
    basis_for_guess."""

    def wrap(*args: Any, **kwargs: Any) -> str:
        kwargs["basis_for_guess"] = basis_for_guess
        return _ensure_mime(*args, **kwargs)

    return wrap
