"""This module converts between range strings and boolean masks.

Index range strings occur frequently in the ert config as e.g., the index
of active realizations. The are of the form "0, 2-4" meaning, for instance,
that realization 0, 2, 3, and 4 are active.

The ranges can overlap. The end of each range is inclusive.

Example:

    >>> mask_to_rangestring([True, True, True])
    "0-2"
    >>> mask_to_rangestring([True, False, True, True])
    "0,2-3"
"""
from typing import Collection, List, Optional, Union


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
        # An empty string means no active indecies. Note that an
        # IndexRange-typed instance being None means the opposite
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
