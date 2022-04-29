"""This module contains functionality of representing a list of booleans as
strings denoting consecutive ranges of boolean True states (a form of run-length
encoding), where length of the list is predetermined. The ranges can
overlap.

The end of each range is inclusive.

Examples `[True, True, True]` is "0-2", `[True, False, True, True]` is "0,2-3"

"""
from typing import Collection, List, Optional, Tuple, Union


def mask_to_rangestring(mask: Collection[Union[bool, int]]) -> str:
    """Convert a mask (ordered collection of booleans or int) into a rangestring.
    For instance, `0 1 0 1 1 1` would be converted to `1, 3-5`.

    The length of the collection is not encoded in the resulting string and
    must be stored elsewhere.
    """
    ranges: List[str] = []

    def store_range(begin: int, end: int) -> None:
        if end - begin == 1:
            ranges.append(f"{begin}")
        else:
            ranges.append(f"{begin}-{end-1}")

    start: Optional[int] = None
    for i, is_active in enumerate(mask):
        if is_active:
            if start is None:  # begin tracking a range
                start = i
            assert start is not None
        else:
            if start is not None:  # store the range and stop tracking
                store_range(start, i)
                start = None
            assert start is None
    if start is not None:  # complete the last range if any
        store_range(start, len(mask))
    return ", ".join(ranges)


def rangestring_to_mask(rangestring: str, length: int) -> List[bool]:
    """Convert a string specifying ranges of elements, and the number of elements,
    into a list of booleans. The ranges are end-inclusive."""
    mask = [False] * length
    if rangestring == "":
        # An empty string means all realizations deactivated. Note that an
        # ActiveRange-typed instance being None means the opposite
        return mask
    for _range in rangestring.split(","):
        if "-" in _range:
            if len(_range.strip().split("-")) != 2:
                raise ValueError(f"Wrong range syntax {_range}")
            start, end = map(int, _range.strip().split("-"))
            if end < start:
                raise ValueError(f"Range {start}-{end} has invalid direction")
            if end + 1 > length:
                raise ValueError(
                    f"Range endpoint {end} is beyond the mask length {length} "
                )
            mask[start : end + 1] = [True] * (end + 1 - start)
        elif _range:
            if int(_range) + 1 > length:
                raise ValueError(
                    f"Realization index {_range} is beyond the mask length {length} "
                )
            mask[int(_range)] = True
    return mask


class ActiveRange:
    def __init__(
        self,
        mask: Optional[List[bool]] = None,
        rangestring: Optional[str] = None,
        length: Optional[int] = None,
    ):
        if mask is None and rangestring is None and length is None:
            raise ValueError("Supply mask or rangestring and length to ActiveRange.")
        if mask is None:
            if length is not None and rangestring is not None:
                self._mask = rangestring_to_mask(rangestring=rangestring, length=length)
            else:
                raise ValueError("Must supply both rangestring and length")
        else:
            if rangestring is not None:
                raise ValueError("Can't supply both mask and rangestring")
            if length is not None:
                if length != len(mask):
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
        if rangestring.strip() == "":
            return rangestring
        if not set(rangestring).issubset("0123456789-, "):
            raise ValueError(
                f"Only digits, commas, dashes and spaces are allowed, got {rangestring}"
            )
        for _range in rangestring.split(","):
            if "-" in _range:
                if len(_range.split("-")) != 2:
                    raise ValueError(f"Invalid range specified, got {_range} ")
                realization_bounds = _range.split("-")
                start = int(realization_bounds[0])
                end = int(realization_bounds[1])
                if end < start:
                    raise ValueError(
                        f"Invalid direction in range specified, got {_range} "
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
            if realization_index != "" and int(realization_index) >= length:
                raise ValueError(
                    f"Realization out of ensemble bounds in {rangestring} "
                    f"for size {length}"
                )
        return (rangestring, length)
