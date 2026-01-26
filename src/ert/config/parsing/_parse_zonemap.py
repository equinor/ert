from .config_errors import ConfigValidationError


def parse_zonemap(filename: str, content: str) -> dict[int, list[str]]:
    """A zonemap is a map from a simulation grids layers to a list of zone names
    that the k-layers belongs to.

    Note that the map uses 1-indexing of layers.
    """
    zones_at_k_value: dict[int, list[str]] = {}

    base_err_msg = "On Line {line_number} in the ZONEMAP file: "
    for idx, line in enumerate(content.splitlines()):
        line_number = idx + 1
        zonemap_line = _strip_comments(line).split()

        if not zonemap_line:
            continue

        if len(zonemap_line) < 2:
            raise ConfigValidationError.with_context(
                "Number of zonenames must be 1 or more.",
                filename,
            )
        try:
            k = int(zonemap_line[0])
        except ValueError as err:
            raise ConfigValidationError.with_context(
                base_err_msg.format(line_number=line_number)
                + f"k must be an integer, was {zonemap_line[0]}.",
                filename,
            ) from err
        if k <= 0:
            raise ConfigValidationError.with_context(
                base_err_msg.format(line_number=line_number)
                + "k must be at least 1. Layers are 1-indexed.",
                filename,
            )
        zones_at_k_value[k] = [zone.strip() for zone in zonemap_line[1:]]

    return zones_at_k_value


def _strip_comments(line: str) -> str:
    return line.partition("--")[0].rstrip()
