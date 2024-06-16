from __future__ import annotations

from typing import Iterator, Optional, TypeVar, overload

from .parsing import ConfigWarning


def get_num_cpu_from_data_file(data_file: str) -> Optional[int]:
    """Reads the number of cpus required from the reservoir simulator .data file.

    Works similarly to resdata.util.get_num_cpu

    This does not attempt to parse the .data file completely, although
    that could be done using opm.io.Parser. The file format is context
    sensitive and contains ~2000 keywords many of which needs to be parsed
    in a unique way or changes the context.

    Instead, we keep backwards compatibility with the files that we can
    parse using the following heuristic method:

    1. the first word on any line is the keyword;
    2. A line is separated into words by splitting
        with space, quotations and comments (see _split_line);
    3. A sequence is consecutive words ended by "/".
        The PARALLEL keyword is followed by one sequence.
        The SLAVES keyword is followed by several sequences, and ends by a single "/".
    4. Keywords that are not "PARALLEL" or "SLAVES" are ignored, except TITLE where
        the next two lines have to be skipped.

    To disambiguate what is correct behavior, the following implementation using
    opm.io shows how this is interpreted by opm flow:

    .. code-block:: python

        from __future__ import annotations

        from typing import Any, Optional, Tuple

        import opm.io

        from .parsing import ConfigWarning

        OPMIOPARSER_RECOVERY: list[Tuple[str, Any]] = [
            ("PARSE_EXTRA_DATA", opm.io.action.ignore),
            ("PARSE_EXTRA_RECORDS", opm.io.action.ignore),
            ("PARSE_INVALID_KEYWORD_COMBINATION", opm.io.action.ignore),
            ("PARSE_MISSING_DIMS_KEYWORD", opm.io.action.ignore),
            ("PARSE_MISSING_INCLUDE", opm.io.action.ignore),
            ("PARSE_MISSING_SECTIONS", opm.io.action.ignore),
            ("PARSE_RANDOM_SLASH", opm.io.action.ignore),
            ("PARSE_RANDOM_TEXT", opm.io.action.ignore),
            ("PARSE_UNKNOWN_KEYWORD", opm.io.action.ignore),
            ("SUMMARY_UNKNOWN_GROUP", opm.io.action.ignore),
            ("UNSUPPORTED_*", opm.io.action.ignore),
        ]


        def get_num_cpu_from_data_file(data_file: str) -> Optional[int]:
            try:
                parsecontext = opm.io.ParseContext(OPMIOPARSER_RECOVERY)
                deck = opm.io.Parser().parse(data_file, parsecontext)
                for _, kword in enumerate(deck):
                    if kword.name in ["PARALLEL"]:
                        return kword[0][0].get_int(0)
                    if kword.name in ["SLAVES"]:
                        num_cpu = 1
                        for rec in kword:
                            num_cpu += rec.get_int(1)
                        return num_cpu
            except Exception as err:
                ConfigWarning.ert_context_warn(
                    f"Failed to read NUM_CPU from {data_file}: {err}",
                    data_file,
                )
            return None

    """
    try:
        with open(data_file, "r") as file:
            return _get_num_cpu(iter(file), data_file)
    except OSError as err:
        ConfigWarning.ert_context_warn(
            f"Failed to read from DATA_FILE {data_file}: {err}", data_file
        )
    return None


def _get_num_cpu(
    lines_iter: Iterator[str], data_file_name: Optional[str] = None
) -> Optional[int]:
    """Handles reading the lines in the data file and returns the num_cpu

    TITLE keyword requires skipping two non-empty lines

    >>> _get_num_cpu(iter(["TITLE", "", "", "PARALLEL", "3 / -- skipped", "PARALLEL", "4 /"]))
    4

    PARALLEL takes presedence even when SLAVES comes first:

    >>> _get_num_cpu(iter(["SLAVES", "/", "PARALLEL", "10 /"]))
    10

    """
    parser = _Parser(lines_iter)
    try:
        slaves_num_cpu = None
        while (words := parser.next_line(None)) is not None:
            if not words:
                continue
            keyword = next(words, None)
            keyword = keyword[0 : min(len(keyword), 8)] if keyword is not None else None
            if keyword == "TITLE":
                # Skip two non-blank lines following a TITLE
                for _ in range(2):
                    line: list[str] = []
                    while line == []:
                        nline = parser.next_line(None)
                        if nline is None:
                            break
                        line = list(nline)
            if keyword == "PARALLEL":
                while (word := next(words, None)) is None:
                    words = parser.next_line(None)
                    if words is None:
                        return None
                if word is not None:
                    return int(word)
                else:
                    return None
            if keyword == "SLAVES" and slaves_num_cpu is None:
                slaves_num_cpu = 1
                while (line_iter := parser.next_line(None)) is not None:
                    parameters = list(line_iter)
                    if not parameters:
                        continue
                    if parameters[0] == "/":
                        break
                    if len(parameters) != 6:
                        slaves_num_cpu += 1
                    else:
                        slaves_num_cpu += int(parameters[4])
    except Exception as err:
        ConfigWarning.ert_context_warn(
            f"Failed to read NUM_CPU from {data_file_name} Line {parser.line_number}: {err}",
            data_file_name if data_file_name else "",
        )

    return slaves_num_cpu


T = TypeVar("T")


class _Parser:
    def __init__(self, line_iterator: Iterator[str]) -> None:
        self._line_iterator = line_iterator
        self.line_number = 1

    @overload
    def next_line(self) -> Iterator[str]: ...

    @overload
    def next_line(self, __default: T) -> Iterator[str] | T: ...

    def next_line(self, *args: T) -> Iterator[str] | T:
        self.line_number += 1
        words = next(self._line_iterator, *args)
        if isinstance(words, str):
            return _split_line(words)
        return words


def _split_line(line: str) -> Iterator[str]:
    """
    split a keyword line inside a .data file. This splits the values of a
    'simple' keyword into tokens. ie.

    >>> list(_split_line("3 1.0 3*4 PORO 3*INC 'HELLO WORLD ' 3*'NAME'"))
    ['3', '1.0', '3*4', 'PORO', '3*INC', 'HELLO WORLD ', '3*', 'NAME']
    """
    value = ""
    inside_str = None
    for char in line:
        if char == "'":
            # end of str
            if inside_str:
                yield value
                value = ""
                inside_str = False
            # start of str
            else:
                if value:
                    yield value
                value = ""
                inside_str = char
        elif inside_str:
            value += char
        elif value and value[-1] == "-" and char == "-":
            # a comment
            value = value[0:-1]
            break
        elif char.isspace():
            # delimiting space
            if value:
                yield value
                value = ""
        else:
            value += char
    if value:
        yield value
