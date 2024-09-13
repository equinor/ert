from typing import List, Optional, Tuple

from .rangestring import mask_to_rangestring, rangestring_to_mask


class ActiveRange:
    def __init__(
        self,
        mask: Optional[List[bool]] = None,
        rangestring: Optional[str] = None,
        length: Optional[int] = None,
    ):
        if mask is None and rangestring is None and length is None:
            raise ValueError("Supply mask or rangestring and length to IndexRange.")
        if mask is None:
            if length is not None and rangestring is not None:
                self._mask = rangestring_to_mask(rangestring=rangestring, length=length)
            else:
                raise ValueError("Must supply both rangestring and length")
        else:
            if rangestring is not None:
                raise ValueError("Can't supply both mask and rangestring")
            if length is not None and length != len(mask):
                raise ValueError(
                    f"Explicit length {length} not equal to mask length {len(mask)}"
                )
            self._mask = mask

    @property
    def mask(self) -> List[bool]:
        return list(self._mask)

    @property
    def rangestring(self) -> str:
        return mask_to_rangestring(self._mask)

    def __len__(self) -> int:
        return len(self._mask)

    def __repr__(self) -> str:
        return mask_to_rangestring(self._mask) + f" length={len(self._mask)}"

    @classmethod
    def validate_rangestring(cls, rangestring: str) -> str:
        if not rangestring.strip():
            return rangestring
        if not set(rangestring).issubset("0123456789-, "):
            raise ValueError(
                f"Only digits, commas, dashes and spaces are allowed, got {rangestring}"
            )
        for _range in rangestring.split(","):
            if "-" in _range:
                if len(_range.split("-")) != 2:
                    raise ValueError(f"Invalid range specified, got {_range}")
                realization_bounds = _range.split("-")
                start = int(realization_bounds[0])
                end = int(realization_bounds[1])
                if end < start:
                    raise ValueError(
                        f"Invalid direction in range specified, got {_range}"
                    )
            else:
                int(_range)
        return rangestring

    @classmethod
    def validate_rangestring_vs_length(
        cls, rangestring: str, length: int
    ) -> Tuple[str, int]:
        for realization_index in (
            rangestring.replace("-", " ").replace(",", " ").split(" ")
        ):
            if realization_index and int(realization_index) >= length:
                raise ValueError(
                    f"Realization out of ensemble bounds in {rangestring} "
                    f"for size {length}"
                )
        return (rangestring, length)
