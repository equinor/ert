from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import starmap
from pathlib import Path
from typing import Any, Self, assert_never, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from resfo_utilities import history_key

from ._observations import (
    ErrorModes,
    _missing_value_error,
    _parse_date,
    _unknown_key_error,
    validate_error_mode,
    validate_float,
    validate_int,
    validate_positive_float,
    validate_positive_int,
)
from .ert_config import ErtConfig, logger
from .parsing import (
    HistorySource,
    ObservationConfigError,
    ObservationDict,
    read_file,
)
from .parsing.config_errors import ConfigValidationError, ConfigWarning
from .refcase import Refcase

DEFAULT_TIME_DELTA = timedelta(seconds=30)


@dataclass
class ObservationError:
    error_mode: ErrorModes
    error: float
    error_min: float


@dataclass
class Segment(ObservationError):
    name: str
    start: int
    stop: int


def _validate_segment_dict(name_token: str, inp: dict[str, Any]) -> Segment:
    start = None
    stop = None
    error_mode = ErrorModes.RELMIN
    error = 0.1
    error_min = 0.1
    for key, value in inp.items():
        match key:
            case "START":
                start = validate_int(value, key)
            case "STOP":
                stop = validate_int(value, key)
            case "ERROR":
                error = validate_positive_float(value, key, strictly_positive=True)
            case "ERROR_MIN":
                error_min = validate_positive_float(value, key, strictly_positive=True)
            case "ERROR_MODE":
                error_mode = validate_error_mode(value)
            case _:
                raise _unknown_key_error(key, name_token)

    if start is None:
        raise _missing_value_error(name_token, "START")
    if stop is None:
        raise _missing_value_error(name_token, "STOP")
    return Segment(
        name=name_token,
        start=start,
        stop=stop,
        error_mode=error_mode,
        error=error,
        error_min=error_min,
    )


@dataclass
class HistoryObservation(ObservationError):
    name: str
    segments: list[Segment]

    @property
    def key(self) -> str:
        """The :term:`summary key` to be fetched from :ref:`refcase`."""
        # For history observations the key is also the name, ie.
        # "HISTORY_OBSERVATION FOPR" means to add the values from
        # the summary vector FOPRH in refcase as observations.
        return self.name

    @classmethod
    def from_obs_dict(
        cls, directory: str, observation_dict: ObservationDict
    ) -> list[Self]:
        error_mode = ErrorModes.RELMIN
        error = 0.1
        error_min = 0.1
        segments = []
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "ERROR":
                    error = validate_positive_float(value, key, strictly_positive=True)
                case "ERROR_MIN":
                    error_min = validate_positive_float(
                        value, key, strictly_positive=True
                    )
                case "ERROR_MODE":
                    error_mode = validate_error_mode(value)
                case "segments":
                    segments = list(starmap(_validate_segment_dict, value))
                case _:
                    raise _unknown_key_error(str(key), observation_dict["name"])

        instance = cls(
            name=observation_dict["name"],
            error_mode=error_mode,
            error=error,
            error_min=error_min,
            segments=segments,
        )
        return [instance]


@dataclass(frozen=True)
class TextEdit:
    """
    Represents a replacement of a declaration block in a file.
    Line numbers are 1-based and inclusive.
    """

    start_line: int
    end_line: int
    replacement: str


def extract_declaration_block(lines: list[str], start_line: int) -> tuple[int, int]:
    i = start_line - 1
    line = lines[i]
    line_without_comment = line.split("--", 1)[0]

    # Single-line declaration
    if ";" in line_without_comment and "{" not in line_without_comment:
        return i, i

    brace_depth = 0
    has_opening_brace = False
    for j in range(i, len(lines)):
        line_to_process = lines[j].split("--", 1)[0]

        if "{" in line_to_process:
            has_opening_brace = True

        brace_depth += line_to_process.count("{")
        brace_depth -= line_to_process.count("}")

        if has_opening_brace and brace_depth == 0:
            return i, j

    raise ValueError(f"Unterminated declaration at line {start_line}")


@dataclass(frozen=True)
class _SummaryFromHistoryChange:
    source_observation: HistoryObservation
    summary_obs_declarations: list[str]
    lines: list[str]

    def edits(self) -> list[TextEdit]:
        start, end = extract_declaration_block(
            self.lines,
            self.source_observation.name.line,  # type: ignore
        )

        replacement = "\n\n".join(self.summary_obs_declarations) + "\n"

        return [
            TextEdit(
                start_line=start + 1,
                end_line=end + 1,
                replacement=replacement,
            )
        ]


@dataclass(frozen=True)
class _GeneralObservationChange:
    source_observation: LegacyGeneralObservation
    declaration: str
    restart: int
    lines: list[str]

    def edits(self) -> list[TextEdit]:
        start, end = extract_declaration_block(
            self.lines,
            self.source_observation.name.line,  # type: ignore
        )

        return [
            TextEdit(
                start_line=start + 1,
                end_line=end + 1,
                replacement=self.declaration + "\n",
            )
        ]


@dataclass(frozen=True)
class _SummaryObservationChange:
    source_observation: LegacySummaryObservation
    declaration: str
    date: datetime
    lines: list[str]

    def edits(self) -> list[TextEdit]:
        start, end = extract_declaration_block(
            self.lines,
            self.source_observation.name.line,  # type: ignore
        )

        return [
            TextEdit(
                start_line=start + 1,
                end_line=end + 1,
                replacement=self.declaration + "\n",
            )
        ]


@dataclass(frozen=True)
class _TimeMapAndRefcaseRemovalInfo:
    obs_config_path: str
    refcase_path: str | None
    time_map_path: str | None
    history_changes: list[_SummaryFromHistoryChange]
    general_obs_changes: list[_GeneralObservationChange]
    summary_obs_changes: list[_SummaryObservationChange]

    def is_empty(self) -> bool:
        return (
            len(self.history_changes)
            + len(self.general_obs_changes)
            + len(self.summary_obs_changes)
        ) == 0

    def collect_edits(self) -> list[TextEdit]:
        edits: list[TextEdit] = []

        for history_change in self.history_changes:
            edits.extend(history_change.edits())

        for gen_change in self.general_obs_changes:
            edits.extend(gen_change.edits())

        for summary_change in self.summary_obs_changes:
            edits.extend(summary_change.edits())

        return edits

    def apply_to_file(self, path: Path) -> None:
        edits = self.collect_edits()

        # Sort edits bottom-up, so that line numbers remain valid for subsequent edits
        edits.sort(key=lambda e: e.start_line, reverse=True)

        lines = path.read_text(encoding="utf-8").splitlines()

        for edit in edits:
            # Line numbers are 1-based, inclusive. Convert to 0-based slice.
            start_index = edit.start_line - 1
            end_index = edit.end_line

            replacement_lines = edit.replacement.splitlines()
            lines[start_index:end_index] = replacement_lines

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class LegacyObservationDate:
    days: float | None = None
    hours: float | None = None
    date: str | None = None
    restart: int | None = None


@dataclass
class _LegacySummaryValues:
    name: str
    value: float
    key: str  #: The :term:`summary key` in the summary response
    location_x: float | None = None
    location_y: float | None = None
    location_range: float | None = None


@dataclass
class LegacySummaryObservation(
    LegacyObservationDate, _LegacySummaryValues, ObservationError
):
    @classmethod
    def from_obs_dict(
        cls, directory: str, observation_dict: ObservationDict
    ) -> list[Self]:
        error_mode = ErrorModes.ABS
        summary_key = None

        date_dict: LegacyObservationDate = LegacyObservationDate()
        float_values: dict[str, float] = {"ERROR_MIN": 0.1}
        localization_values: dict[str, float] = {}
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "RESTART":
                    date_dict.restart = validate_positive_int(value, key)
                case "ERROR" | "ERROR_MIN":
                    float_values[str(key)] = validate_positive_float(
                        value, key, strictly_positive=True
                    )
                case "DAYS" | "HOURS":
                    setattr(
                        date_dict, str(key).lower(), validate_positive_float(value, key)
                    )
                case "VALUE":
                    float_values[str(key)] = validate_float(value, key)
                case "ERROR_MODE":
                    error_mode = validate_error_mode(value)
                case "KEY":
                    summary_key = value
                case "DATE":
                    date_dict.date = value
                case "LOCATION_X":
                    localization_values["x"] = validate_float(value, key)
                case "LOCATION_Y":
                    localization_values["y"] = validate_float(value, key)
                case "LOCATION_RANGE":
                    localization_values["range"] = validate_float(value, key)
                case _:
                    raise _unknown_key_error(str(key), observation_dict["name"])
        if "VALUE" not in float_values:
            raise _missing_value_error(observation_dict["name"], "VALUE")
        if summary_key is None:
            raise _missing_value_error(observation_dict["name"], "KEY")
        if "ERROR" not in float_values:
            raise _missing_value_error(observation_dict["name"], "ERROR")

        return [
            cls(
                name=observation_dict["name"],
                error_mode=error_mode,
                error=float_values["ERROR"],
                error_min=float_values["ERROR_MIN"],
                key=summary_key,
                value=float_values["VALUE"],
                location_x=localization_values.get("x"),
                location_y=localization_values.get("y"),
                location_range=localization_values.get("range"),
                **date_dict.__dict__,
            )
        ]


@dataclass
class _LegacyGeneralObservation:
    name: str
    data: str
    value: float | None = None
    error: float | None = None
    index_list: str | None = None
    index_file: str | None = None
    obs_file: str | None = None
    days: float | None = None
    hours: float | None = None
    date: str | None = None
    restart: int | None = None


@dataclass
class LegacyGeneralObservation(_LegacyGeneralObservation):
    @classmethod
    def from_obs_dict(
        cls, directory: str, observation_dict: ObservationDict
    ) -> list[Self]:
        try:
            data = observation_dict["DATA"]
        except KeyError as err:
            raise _missing_value_error(observation_dict["name"], "DATA") from err

        output = cls(name=observation_dict["name"], data=data)
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "RESTART":
                    output.restart = validate_positive_int(value, key)
                case "VALUE":
                    output.value = validate_float(value, key)
                case "ERROR" | "DAYS" | "HOURS":
                    setattr(
                        output, str(key).lower(), validate_positive_float(value, key)
                    )
                case "DATE" | "INDEX_LIST":
                    setattr(output, str(key).lower(), value)
                case "OBS_FILE" | "INDEX_FILE":
                    assert not isinstance(key, tuple)
                    filename = value
                    if not os.path.isabs(filename):
                        filename = os.path.join(directory, filename)
                    if not os.path.exists(filename):
                        raise ObservationConfigError.with_context(
                            "The following keywords did not"
                            f" resolve to a valid path:\n {key}",
                            value,
                        )
                    setattr(output, str(key).lower(), filename)
                case "DATA":
                    output.data = value
                case _:
                    raise _unknown_key_error(str(key), observation_dict["name"])
        if output.value is not None and output.error is None:
            raise ObservationConfigError.with_context(
                f"For GENERAL_OBSERVATION {observation_dict['name']}, with"
                f" VALUE = {output.value}, ERROR must also be given.",
                observation_dict["name"],
            )
        return [output]


def remove_refcase_and_time_map_dependence_from_obs_config(
    config_path: str,
) -> _TimeMapAndRefcaseRemovalInfo | None:
    """
    Parses an ERT configuration to find observation declarations that depend on
    REFCASE or TIME_MAP and generates a set of proposed changes to remove these
    dependencies.

    The function reads the main ERT config and identifies the observation
    configuration file. It then processes three types of observation declarations:

    1.  HISTORY_OBSERVATION: These are converted into one or more
        SUMMARY_OBSERVATION declarations, with dates and values extracted
        from the REFCASE.
    2.  GENERAL_OBSERVATION: Declarations using the DATE keyword are updated
        to use the corresponding RESTART (report step) from the REFCASE or TIME_MAP.
    3.  SUMMARY_OBSERVATION: Declarations using the RESTART (report step) keyword
        are updated to use the corresponding DATE from the REFCASE or TIME_MAP.

    All proposed modifications are collected and returned in a
    _TimeMapAndRefcaseRemovalInfo object, which can be used to apply the
    changes to the observation file. This function does not modify any files itself.

    """
    user_config_contents = read_file(config_path)
    config_dict = ErtConfig._config_dict_from_contents(
        user_config_contents,
        config_path,
    )

    refcase = (
        Refcase.from_config_dict(config_dict) if "REFCASE" in config_dict else None
    )

    obs_config_file, obs_config_entries = config_dict.get("OBS_CONFIG", (None, None))
    if obs_config_file is None:
        return None

    time_map = None
    time_map_path = None
    if time_map_args := config_dict.get("TIME_MAP"):
        time_map_file, time_map_contents = time_map_args
        try:
            time_map = _read_time_map(time_map_contents)
        except ValueError as err:
            raise ConfigValidationError.with_context(
                f"Could not read timemap file {time_map_file}: {err}",
                time_map_file,
            ) from err

    obs_config_lines = read_file(str(obs_config_file)).splitlines()
    config_dir = Path(obs_config_file).parent

    history_source = config_dict.get("HISTORY_SOURCE", HistorySource.REFCASE_HISTORY)
    obs_time_list: list[datetime] = []
    if refcase is not None:
        obs_time_list = refcase.all_dates
    elif time_map is not None:
        obs_time_list = time_map

    # Create observation objects from the configuration
    history_observations: list[HistoryObservation] = [
        obs
        for obs_dict in obs_config_entries
        if obs_dict.get("type") == "HISTORY_OBSERVATION"
        for obs in HistoryObservation.from_obs_dict("", obs_dict)
    ]

    genobs_deprecated_keys = {"DATE", "DAYS", "HOURS"}
    general_observations: list[LegacyGeneralObservation] = [
        obs
        for obs_dict in obs_config_entries
        if obs_dict.get("type") == "GENERAL_OBSERVATION"
        and (len(genobs_deprecated_keys.intersection(set(obs_dict))) > 0)
        for obs in LegacyGeneralObservation.from_obs_dict(str(config_dir), obs_dict)
    ]

    summary_deprecated_keys = {"RESTART", "DAYS", "HOURS"}
    summary_observations: list[LegacySummaryObservation] = [
        obs
        for obs_dict in obs_config_entries
        if obs_dict.get("type") == "SUMMARY_OBSERVATION"
        and (len(summary_deprecated_keys.intersection(set(obs_dict))) > 0)
        for obs in LegacySummaryObservation.from_obs_dict(str(config_dir), obs_dict)
    ]
    # Process history observations, which generate summary observation declarations
    history_changes = []
    for history_obs in history_observations:
        history_obs_df = _handle_history_observation(
            refcase, history_obs, history_obs.name, history_source, len(obs_time_list)
        )
        declarations = []
        for obs_row in history_obs_df.to_dicts():
            declaration = (
                f"SUMMARY_OBSERVATION "
                f"{obs_row['observation_key']} {{\n"
                f"   VALUE    = {obs_row['observations']};\n"
                f"   ERROR    = {obs_row['std']};\n"
                f"   DATE     = {obs_row['time'].strftime('%Y-%m-%d')};\n"
                f"   KEY      = {obs_row['observation_key']};\n"
                "};"
            )
            declarations.append(declaration)

        history_changes.append(
            _SummaryFromHistoryChange(
                source_observation=history_obs,
                summary_obs_declarations=declarations,
                lines=obs_config_lines,
            )
        )

    # Process general observations
    general_obs_changes = []
    for gen_obs in general_observations:
        restart = _get_restart(
            cast(LegacyObservationDate, gen_obs),
            gen_obs.name,
            obs_time_list,
            refcase is not None,
        )

        index_list_or_file_declaration = ""
        if gen_obs.index_list is not None:
            index_list_or_file_declaration = f"   INDEX_LIST = {gen_obs.index_list};\n"
        elif gen_obs.index_file is not None:
            index_list_or_file_declaration = f"   INDEX_FILE = {gen_obs.index_file};\n"

        obs_file_or_value_declaration = ""
        if gen_obs.value is not None:
            obs_file_or_value_declaration = f"   VALUE      = {gen_obs.value};\n"
            obs_file_or_value_declaration += f"   ERROR      = {gen_obs.error};\n"
        if gen_obs.obs_file is not None:
            obs_file_or_value_declaration = (
                f"   OBS_FILE   = {Path(gen_obs.obs_file).relative_to(config_dir)};\n"
            )

        declaration = (
            f"GENERAL_OBSERVATION {gen_obs.name} {{\n"
            f"   DATA       = {gen_obs.data};\n"
            f"{index_list_or_file_declaration}"
            f"   RESTART    = {restart};\n"
            f"{obs_file_or_value_declaration}"
            "};"
        )
        general_obs_changes.append(
            _GeneralObservationChange(
                source_observation=gen_obs,
                declaration=declaration,
                restart=restart,
                lines=obs_config_lines,
            )
        )

    # Process summary observations
    summary_obs_changes = []
    for smry_obs in summary_observations:
        restart = _get_restart(
            smry_obs, smry_obs.name, obs_time_list, refcase is not None
        )
        date = obs_time_list[restart]

        declaration = (
            f"SUMMARY_OBSERVATION {smry_obs.name} {{\n"
            f"   VALUE    = {smry_obs.value};\n"
            f"   ERROR    = {smry_obs.error};\n"
            f"   DATE     = {date.strftime('%Y-%m-%d')};\n"
            f"   KEY      = {smry_obs.key};\n"
            + (
                f"   LOCATION_X={smry_obs.location_x};\n"
                if smry_obs.location_x is not None
                else ""
            )
            + (
                f"   LOCATION_Y={smry_obs.location_y};\n"
                if smry_obs.location_y is not None
                else ""
            )
            + (
                f"   LOCATION_RANGE={smry_obs.location_range};\n"
                if smry_obs.location_range is not None
                else ""
            )
            + "};"
        )
        summary_obs_changes.append(
            _SummaryObservationChange(
                source_observation=smry_obs,
                declaration=declaration,
                date=date,
                lines=obs_config_lines,
            )
        )

    return _TimeMapAndRefcaseRemovalInfo(
        obs_config_path=str(obs_config_file),
        refcase_path=config_dict.get("REFCASE", None),
        time_map_path=time_map_path,
        history_changes=history_changes,
        general_obs_changes=general_obs_changes,
        summary_obs_changes=summary_obs_changes,
    )


def _find_nearest(
    time_map: list[datetime],
    time: datetime,
    threshold: timedelta = DEFAULT_TIME_DELTA,
) -> int:
    nearest_index = -1
    nearest_diff = None
    for i, t in enumerate(time_map):
        diff = abs(time - t)
        if diff < threshold and (nearest_diff is None or nearest_diff > diff):
            nearest_diff = diff
            nearest_index = i
    if nearest_diff is None:
        raise IndexError(f"{time} is not in the time map")
    return nearest_index


def _get_time(
    date_dict: LegacyObservationDate, start_time: datetime, context: Any = None
) -> tuple[datetime, str]:
    if date_dict.date is not None:
        return _parse_date(date_dict.date), f"DATE={date_dict.date}"
    if date_dict.days is not None:
        days = date_dict.days
        return start_time + timedelta(days=days), f"DAYS={days}"
    if date_dict.hours is not None:
        hours = date_dict.hours
        return start_time + timedelta(hours=hours), f"HOURS={hours}"
    raise ObservationConfigError.with_context("Missing time specifier", context=context)


def _get_restart(
    date_dict: LegacyObservationDate,
    obs_name: str,
    time_map: list[datetime],
    has_refcase: bool,
) -> int:
    if date_dict.restart is not None:
        return date_dict.restart
    if not time_map:
        raise ObservationConfigError.with_context(
            f"Missing REFCASE or TIME_MAP for observations: {obs_name}",
            obs_name,
        )

    time, date_str = _get_time(date_dict, time_map[0], context=obs_name)

    try:
        return _find_nearest(time_map, time)
    except IndexError as err:
        raise ObservationConfigError.with_context(
            f"Could not find {time} ({date_str}) in "
            f"the time map for observations {obs_name}. "
            + (
                "The time map is set from the REFCASE keyword. Either "
                "the REFCASE has an incorrect/missing date, or the observation "
                "is given an incorrect date.)"
                if has_refcase
                else "(The time map is set from the TIME_MAP "
                "keyword. Either the time map file has an "
                "incorrect/missing date, or the observation is given an "
                "incorrect date."
            ),
            obs_name,
        ) from err


def _legacy_handle_error_mode(
    values: npt.ArrayLike,
    error_dict: ObservationError,
) -> npt.NDArray[np.double]:
    values = np.asarray(values)
    error_mode = error_dict.error_mode
    error_min = error_dict.error_min
    error = error_dict.error
    match error_mode:
        case ErrorModes.ABS:
            return np.full(values.shape, error)
        case ErrorModes.REL:
            return np.abs(values) * error
        case ErrorModes.RELMIN:
            return np.maximum(np.abs(values) * error, np.full(values.shape, error_min))
        case default:
            assert_never(default)


def _handle_history_observation(
    refcase: Refcase | None,
    history_observation: HistoryObservation,
    summary_key: str,
    history_type: HistorySource,
    time_len: int,
) -> pl.DataFrame:
    if refcase is None:
        raise ObservationConfigError.with_context(
            "REFCASE is required for HISTORY_OBSERVATION", summary_key
        )

    if history_type == HistorySource.REFCASE_HISTORY:
        local_key = history_key(summary_key)
    else:
        local_key = summary_key
    if local_key not in refcase.keys:
        raise ObservationConfigError.with_context(
            f"Key {local_key!r} is not present in refcase", summary_key
        )
    values = refcase.values[refcase.keys.index(local_key)]
    std_dev = _legacy_handle_error_mode(values, history_observation)
    for segment in history_observation.segments:
        start = segment.start
        stop = segment.stop
        if start < 0:
            ConfigWarning.warn(
                f"Segment {segment.name} out of bounds."
                " Truncating start of segment to 0.",
                segment.name,
            )
            start = 0
        if stop >= time_len:
            ConfigWarning.warn(
                f"Segment {segment.name} out of bounds. Truncating"
                f" end of segment to {time_len - 1}.",
                segment.name,
            )
            stop = time_len - 1
        if start > stop:
            ConfigWarning.warn(
                f"Segment {segment.name} start after stop. Truncating"
                f" end of segment to {start}.",
                segment.name,
            )
            stop = start
        if np.size(std_dev[start:stop]) == 0:
            ConfigWarning.warn(
                f"Segment {segment.name} does not"
                " contain any time steps. The interval "
                f"[{start}, {stop}) does not intersect with steps in the"
                "time map.",
                segment.name,
            )
        std_dev[start:stop] = _legacy_handle_error_mode(values[start:stop], segment)
    dates_series = pl.Series(refcase.dates).dt.cast_time_unit("ms")
    if (std_dev <= 0).any():
        raise ObservationConfigError.with_context(
            "Observation uncertainty must be given a strictly positive value",
            summary_key,
        ) from None

    return pl.DataFrame(
        {
            "response_key": summary_key,
            "observation_key": summary_key,
            "time": dates_series,
            "observations": pl.Series(values, dtype=pl.Float32),
            "std": pl.Series(std_dev, dtype=pl.Float32),
            "location_x": pl.Series([None] * len(values), dtype=pl.Float32),
            "location_y": pl.Series([None] * len(values), dtype=pl.Float32),
            "location_range": pl.Series([None] * len(values), dtype=pl.Float32),
        }
    )


def _read_time_map(file_contents: str) -> list[datetime]:
    def str_to_datetime(date_str: str) -> datetime:
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            logger.warning(
                "DD/MM/YYYY date format is deprecated"
                ", please use ISO date format YYYY-MM-DD."
            )
            return datetime.strptime(date_str, "%d/%m/%Y")

    dates = []
    for line in file_contents.splitlines():
        dates.append(str_to_datetime(line.strip()))
    return dates
