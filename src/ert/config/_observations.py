from __future__ import annotations

import inspect
import logging
import os
import pathlib
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, assert_never

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from resfo_utilities import make_summary_key

from ert.validation import rangestring_to_list

from ._shapes import CircleShapeConfig, ShapeConfig, ShapeRegistry
from .parsing import (
    ConfigWarning,
    ErrorInfo,
    ObservationConfigError,
    ObservationDict,
    ObservationType,
)
from .parsing.file_context_token import FileContextToken

logger = logging.getLogger(__name__)

DEFAULT_LOCALIZATION_RADIUS = 2000


class ErrorModes(StrEnum):
    REL = "REL"
    ABS = "ABS"
    RELMIN = "RELMIN"


class BaseObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _SummaryValues(BaseObservation):
    type: Literal["summary_observation"] = "summary_observation"
    name: str
    value: float
    error: float
    key: str  #: The :term:`summary key` in the summary response
    date: str
    shape_id: int | None = None


def _cast_optional_csv_value(value: str | None, annotation: str) -> int | str | None:
    if value is None:
        return None
    if "int" in annotation:
        return int(value)
    return value


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        try:
            date = datetime.strptime(date_str, "%d/%m/%Y")  # noqa: DTZ007
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Unsupported date format {date_str}. Please use ISO date format",
                date_str,
            ) from err
        else:
            ConfigWarning.warn(
                f"Deprecated time format {date_str}."
                " Please use ISO date format YYYY-MM-DD",
                date_str,
            )
            return date


class SummaryObservation(_SummaryValues):
    error_mode: ErrorModes | None = Field(default=None, exclude=True)
    error_min: float | None = Field(default=None, exclude=True)

    @classmethod
    def from_obs_dict(
        cls,
        directory: str,
        observation_dict: ObservationDict,
        shape_registry: ShapeRegistry,
    ) -> list[Self | BreakthroughObservation]:
        if observation_dict["type"] is ObservationType.SUMMARY_COMMON_CONFIG:
            return cls.from_common_config_dict(
                directory, observation_dict, shape_registry
            )

        error_mode = ErrorModes.ABS
        summary_key = None

        date: str | None = None
        float_values: dict[str, float] = {"ERROR_MIN": 0.1}
        east = None
        north = None
        radius = None
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "ERROR" | "ERROR_MIN":
                    float_values[str(key)] = validate_positive_float(
                        value, key, strictly_positive=True
                    )
                case "DAYS" | "HOURS" | "RESTART":
                    raise ObservationConfigError.with_context(
                        (
                            "SUMMARY_OBSERVATION must use DATE to specify "
                            "date, DAYS | HOURS is no longer allowed. "
                            "Please run:\n ert convert_observations "
                            "<your_ert_config.ert>\nto migrate the observation config "
                            "to use the correct format."
                        ),
                        key,
                    )
                case "VALUE":
                    float_values[str(key)] = validate_float(value, key)
                case "ERROR_MODE":
                    error_mode = validate_error_mode(value)
                case "KEY":
                    summary_key = value
                case "DATE":
                    date = value
                case "LOCALIZATION":
                    validate_localization(value, observation_dict.context)
                    east, north, radius = extract_localization_values(value)
                case _:
                    raise _unknown_key_error(str(key), observation_dict.context)
        if "VALUE" not in float_values:
            raise _missing_value_error(observation_dict.context, "VALUE")
        if summary_key is None:
            raise _missing_value_error(observation_dict.context, "KEY")
        if "ERROR" not in float_values:
            raise _missing_value_error(observation_dict.context, "ERROR")

        assert date is not None
        # Raise errors if the date is off
        standardized_date: str = _parse_date(date).isoformat()

        value = float_values["VALUE"]
        input_error = float_values["ERROR"]
        error_min = float_values["ERROR_MIN"]

        error = input_error
        match error_mode:
            case ErrorModes.ABS:
                error = validate_positive_float(
                    np.abs(input_error), summary_key, strictly_positive=True
                )
            case ErrorModes.REL:
                error = validate_positive_float(
                    np.abs(value) * input_error, summary_key, strictly_positive=True
                )
            case ErrorModes.RELMIN:
                error = validate_positive_float(
                    np.maximum(np.abs(value) * input_error, error_min),
                    summary_key,
                    strictly_positive=True,
                )
            case default:
                assert_never(default)

        shape_id = cls.get_shape_id(east, north, radius, shape_registry)

        name = (
            observation_dict["name"]
            if observation_dict["name"] is not None
            else summary_key
        )

        obs_instance = cls(
            name=name,
            error=error,
            key=summary_key,
            value=value,
            shape_id=shape_id,
            date=standardized_date,
        )
        # Bypass pydantic discarding context
        # only relevant for ERT config surfacing validation errors
        # irrelevant for runmodels etc.
        obs_instance.name = name

        return [obs_instance]

    @classmethod
    def from_common_config_dict(
        cls,
        directory: str,
        observation_dict: ObservationDict,
        shape_registry: ShapeRegistry,
    ) -> list[Self | BreakthroughObservation]:
        obs_instances: list[Self | BreakthroughObservation] = []
        context: FileContextToken = observation_dict.context

        required_csv_columns = ["keyword", "value", "error", "date"]

        # Optional arguments for resfo_utilities.make_summary_key
        optional_csv_columns = {}
        for name, param in inspect.signature(make_summary_key).parameters.items():
            if param.default is not inspect.Parameter.empty:
                optional_csv_columns[name] = param.annotation

        csv_file = observation_dict.get("VALUES")
        if csv_file is None:
            raise _missing_value_error(context, "VALUES")
        csv_path = _resolve_path(directory, csv_file)
        csv_df = pl.read_csv(
            csv_path,
            encoding="utf-8",
        )
        # Strip all whitespaces from dataframe columns and values
        csv_df = csv_df.rename({col: col.strip() for col in csv_df.columns}).select(
            pl.all().str.strip_chars()
        )
        # Rename 'well' column to 'name' to match make_summary_key parameter
        if "well" not in csv_df.columns:
            raise _missing_csv_column(context, "well", csv_file)
        csv_df = csv_df.rename({"well": "name"})

        for col in csv_df.columns:
            if col not in {
                *required_csv_columns,
                *optional_csv_columns,
            }:
                raise _unknown_csv_column(context, col, csv_path)

        for required_col in required_csv_columns:
            if required_col not in csv_df.columns:
                raise _missing_csv_column(context, required_col, csv_file)

        well_localization = {}

        for key, value in observation_dict.items():
            match key:
                case "name" | "VALUES" | "type":
                    pass
                case _:
                    if match := re.match(r"WELL (.+)", key):
                        well_name = match.group(1)
                        for key_, value_ in value.items():
                            match key_:
                                case "LOCALIZATION":
                                    validate_localization(value_, context)
                                    east, north, radius = extract_localization_values(
                                        value_
                                    )
                                    well_localization[well_name] = {
                                        "east": east,
                                        "north": north,
                                        "radius": radius,
                                    }
                                case "BREAKTHROUGH":
                                    breakthrough_context = context.update(
                                        value="BREAKTHROUGH"
                                    )
                                    validate_populated_string(
                                        value_.get("KEY"), "KEY", breakthrough_context
                                    )
                                    breakthrough_key = value_["KEY"]
                                    breakthrough_key = (
                                        breakthrough_key
                                        if breakthrough_key.endswith(f":{well_name}")
                                        else f"{breakthrough_key}:{well_name}"
                                    )
                                    breakthrough_obs_dict = ObservationDict(
                                        {
                                            **value_,
                                        }
                                        | {"KEY": breakthrough_key}
                                        | (
                                            {"LOCALIZATION": value["LOCALIZATION"]}
                                            if "LOCALIZATION" in value
                                            else {}
                                        ),
                                        context=breakthrough_context,
                                    )
                                    obs_instances.extend(
                                        BreakthroughObservation.from_obs_dict(
                                            directory,
                                            breakthrough_obs_dict,
                                            shape_registry,
                                        )
                                    )
                    else:
                        raise _unknown_key_error(str(key), observation_dict.context)

        for row in csv_df.iter_rows(named=True):
            well = row.get("name")
            key = validate_populated_string(row["keyword"], "key", context)

            summary_key = make_summary_key(
                keyword=key,
                **{  # type: ignore[arg-type]
                    col: _cast_optional_csv_value(row.get(col), annotation)
                    for col, annotation in optional_csv_columns.items()
                },
            )
            key = f"{key}{f':{well}' if well else ''}"

            standardized_date = _parse_date(row["date"]).isoformat()
            validated_value = validate_float(row["value"], f"({key}) VALUE")
            validated_error = validate_positive_float(
                row["error"], f"({key}) ERROR", strictly_positive=True
            )

            loc_values = well_localization.get(well, {})
            shape_id = cls.get_shape_id(
                loc_values.get("east"),
                loc_values.get("north"),
                loc_values.get("radius"),
                shape_registry,
            )

            obs_instances.append(
                cls(
                    name=summary_key,
                    error=validated_error,
                    key=key,
                    value=validated_value,
                    shape_id=shape_id,
                    date=standardized_date,
                )
            )

        return obs_instances

    @classmethod
    def get_shape_id(
        cls,
        east: float | None,
        north: float | None,
        radius: float | None,
        shape_registry: ShapeRegistry,
    ) -> int | None:
        # Register geometry if localization is present
        shape_id = None
        if east is not None and north is not None:
            radius = radius if radius is not None else DEFAULT_LOCALIZATION_RADIUS
            shape_id = shape_registry.register(
                CircleShapeConfig(
                    east=east,
                    north=north,
                    radius=radius,
                )
            )
        return shape_id

    def shape(self, shape_registry: ShapeRegistry) -> ShapeConfig | None:
        if self.shape_id is not None:
            return shape_registry.get(self.shape_id)
        return None


class _GeneralObservation(BaseObservation):
    type: Literal["general_observation"] = "general_observation"
    name: str
    data: str
    value: float
    error: float
    restart: int
    index: int
    shape_id: int | None = None


class GeneralObservation(_GeneralObservation):
    @classmethod
    def from_obs_dict(
        cls,
        directory: str,
        observation_dict: ObservationDict,
        shape_registry: ShapeRegistry,
    ) -> list[Self]:
        try:
            data = observation_dict["DATA"]
        except KeyError as err:
            raise _missing_value_error(observation_dict.context, "DATA") from err

        allowed = {
            "type",
            "name",
            "RESTART",
            "VALUE",
            "ERROR",
            "DATE",
            "DAYS",
            "HOURS",
            "INDEX_LIST",
            "OBS_FILE",
            "INDEX_FILE",
            "DATA",
        }

        extra = set(observation_dict.keys()) - allowed
        if extra:
            raise _unknown_key_error(str(next(iter(extra))), observation_dict.context)

        if any(k in observation_dict for k in ("DATE", "DAYS", "HOURS")):
            bad_key = next(
                k for k in ("DATE", "DAYS", "HOURS") if k in observation_dict
            )
            raise ObservationConfigError.with_context(
                (
                    "GENERAL_OBSERVATION must use RESTART to specify "
                    "report step. Please run:\n ert convert_observations "
                    "<your_ert_config.ert>\nto migrate the observation config "
                    "to use the correct format."
                ),
                bad_key,
            )

        if "OBS_FILE" in observation_dict and (
            "VALUE" in observation_dict or "ERROR" in observation_dict
        ):
            raise ObservationConfigError.with_context(
                "GENERAL_OBSERVATION cannot contain both VALUE/ERROR and OBS_FILE",
                observation_dict.context,
            )

        if "INDEX_FILE" in observation_dict and "INDEX_LIST" in observation_dict:
            raise ObservationConfigError.with_context(
                (
                    "GENERAL_OBSERVATION "
                    f"{observation_dict['name']} has both INDEX_FILE and INDEX_LIST."
                ),
                observation_dict.context,
            )

        restart = (
            validate_positive_int(observation_dict["RESTART"], "RESTART")
            if "RESTART" in observation_dict
            else 0
        )

        if "OBS_FILE" not in observation_dict:
            if "VALUE" in observation_dict and "ERROR" not in observation_dict:
                raise ObservationConfigError.with_context(
                    f"For GENERAL_OBSERVATION {observation_dict['name']}, with"
                    f" VALUE = {observation_dict['VALUE']}, ERROR must also be given.",
                    observation_dict.context,
                )

            if "VALUE" not in observation_dict and "ERROR" not in observation_dict:
                raise ObservationConfigError.with_context(
                    "GENERAL_OBSERVATION must contain either VALUE "
                    "and ERROR or OBS_FILE",
                    context=observation_dict.context,
                )

            index = 0
            name = (
                observation_dict["name"]
                if observation_dict["name"] is not None
                else f"{data}:{restart}:{index}"
            )

            obs_instance = cls(
                name=name,
                data=data,
                value=validate_float(observation_dict["VALUE"], "VALUE"),
                error=validate_positive_float(
                    observation_dict["ERROR"], "ERROR", strictly_positive=True
                ),
                restart=restart,
                index=index,
                shape_id=None,
            )
            # Bypass pydantic discarding context
            # only relevant for ERT config surfacing validation errors
            # irrelevant for runmodels etc.
            obs_instance.name = name
            return [obs_instance]

        obs_filename = _resolve_path(directory, observation_dict["OBS_FILE"])
        try:
            file_values = np.loadtxt(obs_filename, delimiter=None).ravel()
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Failed to read OBS_FILE {obs_filename}: {err}", obs_filename
            ) from err
        if len(file_values) % 2 != 0:
            raise ObservationConfigError.with_context(
                "Expected even number of values in GENERAL_OBSERVATION",
                obs_filename,
            )
        values = file_values[::2]
        stds = file_values[1::2]

        if "INDEX_FILE" in observation_dict:
            idx_file = _resolve_path(directory, observation_dict["INDEX_FILE"])
            indices = np.loadtxt(idx_file, delimiter=None, dtype=np.int32).ravel()
        elif "INDEX_LIST" in observation_dict:
            indices = np.array(
                sorted(rangestring_to_list(observation_dict["INDEX_LIST"])),
                dtype=np.int32,
            )
        else:
            indices = np.arange(len(values), dtype=np.int32)

        if len({len(stds), len(values), len(indices)}) != 1:
            raise ObservationConfigError.with_context(
                (
                    "Values ("
                    f"{values}), error ({stds}) and index list ({indices}) "
                    "must be of equal length"
                ),
                observation_dict.context,
            )

        if np.any(stds <= 0):
            raise ObservationConfigError.with_context(
                "Observation uncertainty must be strictly > 0",
                observation_dict.context,
            )

        obs_instances: list[Self] = []
        for _pos, (val, std, idx) in enumerate(zip(values, stds, indices, strict=True)):
            # index should reflect the index provided by INDEX_FILE / INDEX_LIST
            name = (
                observation_dict["name"]
                if observation_dict["name"] is not None
                else f"{data}:{restart}:{int(idx)}"
            )
            inst = cls(
                name=name,
                data=data,
                value=float(val),
                error=float(std),
                restart=restart,
                index=int(idx),
                shape_id=None,
            )
            # Bypass pydantic discarding context
            # only relevant for ERT config surfacing validation errors
            # irrelevant for runmodels etc.
            inst.name = name
            obs_instances.append(inst)
        return obs_instances

    def shape(self, shape_registry: ShapeRegistry) -> ShapeConfig | None:
        # General observations do not support localization
        return None


class RFTObservation(BaseObservation):
    """Represents an RFT (Repeat Formation Tester) observation.

    RFT observations are used to condition on pressure, saturation, or other
    properties measured at specific well locations and times.
    """

    type: Literal["rft_observation"] = "rft_observation"
    name: str
    well: str
    date: str
    property: str
    value: float
    error: float
    east: float
    north: float
    tvd: float
    md: float | None = None
    shape_id: int | None = None
    zone: str | None = None

    @classmethod
    def from_csv(
        cls,
        directory: str,
        observation_dict: ObservationDict,
        filename: str,
        shape_registry: ShapeRegistry,
        observed_property: str = "PRESSURE",
        radius: float | None = None,
    ) -> list[Self]:
        """Create RFT observations from a CSV file.

        The CSV file must contain the following columns: WELL_NAME, DATE,
        ERROR, NORTH, EAST, TVD, and a column for the observed property
        (e.g., PRESSURE, SWAT). An optional ZONE column may also be present.

        Args:
            directory: Base directory for resolving relative file paths.
            observation_dict: Dictionary containing the observation configuration.
            filename: Path to the CSV file containing RFT observations.
            observed_property: Property to observe (default: PRESSURE).
            radius: Localization radius defined in observation config - outside CSV
                file.

        Returns:
            List of RFTObservation instances created from the CSV file.

        Raises:
            ObservationConfigError: If the file is missing, inaccessible,
                lacks required columns, or contains invalid observation values
                (value=-1 and error=0).
        """
        if not os.path.isabs(filename):
            filename = os.path.join(directory, filename)
        if not pathlib.Path(filename).exists():
            raise ObservationConfigError.with_context(
                f"The CSV file ({filename}) does not exist or is not accessible.",
                filename,
            )
        csv_file = pd.read_csv(
            filename,
            encoding="utf-8",
            on_bad_lines="error",
        )

        required_columns = {
            "WELL_NAME",
            "DATE",
            observed_property,
            "ERROR",
            "NORTH",
            "EAST",
            "TVD",
        }
        missing_required_columns = required_columns - set(csv_file.keys())
        if missing_required_columns:
            raise ObservationConfigError.with_context(
                f"The rft observations file {filename} is missing required column(s) "
                f"{', '.join(sorted(missing_required_columns))}.",
                filename,
            )

        rft_observations = []
        invalid_observations = []
        for row in csv_file.itertuples(index=True):
            east_val = validate_float(str(row.EAST), "EAST")
            north_val = validate_float(str(row.NORTH), "NORTH")
            radius = radius if radius is not None else DEFAULT_LOCALIZATION_RADIUS

            shape_id = shape_registry.register(
                CircleShapeConfig(
                    east=east_val,
                    north=north_val,
                    radius=radius,
                )
            )

            rft_observation = cls(
                name=f"{observation_dict['name']}[{row.Index}]",
                well=str(row.WELL_NAME),
                date=str(row.DATE),
                property=observed_property,
                value=validate_float(
                    str(getattr(row, observed_property)), observed_property
                ),
                error=validate_float(str(row.ERROR), "ERROR"),
                east=east_val,
                north=north_val,
                shape_id=shape_id,
                tvd=validate_float(str(row.TVD), "TVD"),
                md=validate_float(str(row.MD), "MD") if "MD" in csv_file else None,
                zone=(
                    str(row.ZONE)
                    if "ZONE" in csv_file and row.ZONE is not None
                    else None
                ),
            )
            # A value of -1 and error of 0 is used by fmu.tools.rms create_rft_ertobs to
            # indicate missing data. If encountered in an rft observations csv file
            # it should raise an error and ask the user to remove invalid observations.
            if rft_observation.value == -1 and rft_observation.error == 0:
                invalid_observations.append(rft_observation)
            else:
                rft_observations.append(rft_observation)

        if invalid_observations:
            well_list = "\n - ".join(
                [
                    f"{observation.well} at date {observation.date}"
                    for observation in invalid_observations
                ]
            )
            raise ObservationConfigError.with_context(
                (
                    f"Invalid value=-1 and error=0 detected in {filename} for "
                    f"well(s):\n - {well_list}\n"
                    "The invalid observation(s) must be removed from the file."
                ),
                filename,
            )
        return rft_observations

    @classmethod
    def from_obs_dict(
        cls,
        directory: str,
        observation_dict: ObservationDict,
        shape_registry: ShapeRegistry,
    ) -> list[Self]:
        """Create RFT observations from an observation dictionary.

        Supports two modes:
        1. CSV mode: Load observations from a CSV file specified by the CSV key.
        2. Direct mode: Create a single observation from individual keys
           (WELL, PROPERTY, VALUE, ERROR, DATE, NORTH, EAST, TVD, MD, ZONE).

        Args:
            directory: Base directory for resolving relative file paths.
            observation_dict: Dictionary containing the observation configuration.
            shape_registry: ShapeRegistry for storing geometry.

        Returns:
            List of RFTObservation instances. Returns multiple observations when
            loading from CSV, or a single observation when using direct mode.

        Raises:
            ObservationConfigError: If required keys are missing or invalid.
        """
        csv_filename = None
        well = None
        observed_property = None
        observed_value = None
        error = None
        date = None
        north = None
        east = None
        radius = None
        tvd = None
        md = None
        zone = None
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "WELL":
                    well = value
                case "PROPERTY":
                    observed_property = value
                case "VALUE":
                    observed_value = validate_float(value, key)
                case "ERROR":
                    error = validate_float(value, key)
                case "DATE":
                    date = value
                case "NORTH":
                    north = validate_float(value, key)
                case "EAST":
                    east = validate_float(value, key)
                case "TVD":
                    tvd = validate_float(value, key)
                case "CSV":
                    csv_filename = value
                case "ZONE":
                    zone = value
                case "MD":
                    md = validate_float(value, key)
                case "LOCALIZATION":
                    validate_rft_localization(value, observation_dict.context)
                    east, north, radius = extract_localization_values(value)
                    radius = (
                        radius if radius is not None else DEFAULT_LOCALIZATION_RADIUS
                    )
                case _:
                    raise _unknown_key_error(str(key), observation_dict.context)
        if csv_filename is not None:
            return cls.from_csv(
                directory,
                observation_dict,
                csv_filename,
                shape_registry,
                observed_property or "PRESSURE",
                radius=radius,
            )
        if well is None:
            raise _missing_value_error(observation_dict.context, "WELL")
        if observed_value is None:
            raise _missing_value_error(observation_dict.context, "VALUE")
        if observed_property is None:
            raise _missing_value_error(observation_dict.context, "PROPERTY")
        if error is None:
            raise _missing_value_error(observation_dict.context, "ERROR")
        if date is None:
            raise _missing_value_error(observation_dict.context, "DATE")
        if north is None:
            raise _missing_value_error(observation_dict.context, "NORTH")
        if east is None:
            raise _missing_value_error(observation_dict.context, "EAST")
        if tvd is None:
            raise _missing_value_error(observation_dict.context, "TVD")

        name = (
            observation_dict["name"]
            if observation_dict["name"] is not None
            else (
                f"{well}:{date}:{observed_property}:{east}:{north}:{tvd}"
                + (f":{zone}" if zone is not None else "")
            )
        )

        radius = radius if radius is not None else DEFAULT_LOCALIZATION_RADIUS
        shape_id = shape_registry.register(
            CircleShapeConfig(
                east=east,
                north=north,
                radius=radius,
            )
        )

        obs_instance = cls(
            name=name,
            well=well,
            property=observed_property,
            value=observed_value,
            error=error,
            date=date,
            north=north,
            east=east,
            tvd=tvd,
            md=md,
            zone=zone,
            shape_id=shape_id,
        )

        # Bypass pydantic discarding context
        # only relevant for ERT config surfacing validation errors
        # irrelevant for runmodels etc.
        obs_instance.name = name

        return [obs_instance]

    def shape(self, shape_registry: ShapeRegistry) -> ShapeConfig | None:
        if self.shape_id is not None:
            return shape_registry.get(self.shape_id)
        return None


class BreakthroughObservation(BaseObservation):
    type: Literal["breakthrough"] = "breakthrough"
    name: str
    key: str
    date: datetime
    error: float
    threshold: float
    shape_id: int | None = None

    @classmethod
    def from_obs_dict(
        cls,
        directory: str,
        obs_dict: ObservationDict,
        shape_registry: ShapeRegistry,
    ) -> list[Self]:
        summary_key = None
        date = None
        error = None
        threshold = None
        east = None
        north = None
        radius = None
        for key, value in obs_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "KEY":
                    summary_key = value
                case "DATE":
                    date = value
                case "ERROR":
                    error = validate_float(value, key)
                case "THRESHOLD":
                    threshold = validate_float(value, key)
                case "LOCALIZATION":
                    validate_localization(value, obs_dict.context)
                    east, north, radius = extract_localization_values(value)
                case _:
                    raise _unknown_key_error(str(key), obs_dict.context)

        if summary_key is None:
            raise _missing_value_error(obs_dict.context, "KEY")
        if date is None:
            raise _missing_value_error(obs_dict.context, "DATE")
        if error is None:
            raise _missing_value_error(obs_dict.context, "ERROR")
        if threshold is None:
            raise _missing_value_error(obs_dict.context, "THRESHOLD")

        # Register shape if localization is present
        shape_id = None
        if east is not None and north is not None:
            radius = radius if radius is not None else DEFAULT_LOCALIZATION_RADIUS
            shape_id = shape_registry.register(
                CircleShapeConfig(
                    east=east,
                    north=north,
                    radius=radius,
                )
            )

        name = (
            obs_dict.get("name")
            if obs_dict.get("name") is not None
            else f"BREAKTHROUGH:{summary_key}:{threshold}"
        )
        return [
            cls(
                name=name,
                key=summary_key,
                date=date,
                error=error,
                threshold=threshold,
                shape_id=shape_id,
            )
        ]

    def shape(self, shape_registry: ShapeRegistry) -> ShapeConfig | None:
        if self.shape_id is not None:
            return shape_registry.get(self.shape_id)
        return None


Observation = Annotated[
    (
        SummaryObservation
        | GeneralObservation
        | RFTObservation
        | BreakthroughObservation
    ),
    Field(discriminator="type"),
]

_TYPE_TO_CLASS: dict[ObservationType, type[Observation]] = {
    ObservationType.SUMMARY: SummaryObservation,
    ObservationType.GENERAL: GeneralObservation,
    ObservationType.RFT: RFTObservation,
    ObservationType.BREAKTHROUGH: BreakthroughObservation,
    ObservationType.SUMMARY_COMMON_CONFIG: SummaryObservation,
}

LOCALIZATION_KEYS = Literal["east", "north", "radius"]


def make_observations(
    directory: str,
    observation_dicts: Sequence[ObservationDict],
    shape_registry: ShapeRegistry,
) -> list[Observation]:
    """Takes observation dicts and returns validated observations.

    Param:
        directory: The name of the directory the observation config is located in.
            Used to disambiguate relative paths.
        observation_dicts: The collection of observation dictionaries to validate.
        shape_registry: ShapeRegistry for storing localization geometries.

    """
    result: list[Observation] = []
    error_list: list[ErrorInfo | ObservationConfigError] = []
    for obs_dict in observation_dicts:
        if obs_dict["type"] == ObservationType.HISTORY:
            msg = (
                "HISTORY_OBSERVATION is deprecated, and must be specified "
                "as SUMMARY_OBSERVATION. Run"
                " ert convert_observations <ert_config.ert> to convert your "
                "observations automatically"
            )
            logger.error(msg)
            error_list.append(ObservationConfigError(msg))
            continue
        try:
            result.extend(
                _TYPE_TO_CLASS[obs_dict["type"]].from_obs_dict(
                    directory, obs_dict, shape_registry=shape_registry
                )
            )
        except KeyError as err:
            raise _unknown_observation_type_error(obs_dict) from err
        except ObservationConfigError as err:
            error_list.extend(err.errors)

    if error_list:
        raise ObservationConfigError.from_collected(error_list)

    return result


def validate_error_mode(inp: str) -> ErrorModes:
    if inp == "REL":
        return ErrorModes.REL
    if inp == "ABS":
        return ErrorModes.ABS
    if inp == "RELMIN":
        return ErrorModes.RELMIN
    raise ObservationConfigError.with_context(
        f'Unexpected ERROR_MODE {inp}. Failed to validate "{inp}"', inp
    )


def validate_populated_string(val: str, key: str, context: FileContextToken) -> str:
    if not val:
        raise ObservationConfigError.with_context(
            f'Required string value for key "{key}" is empty for {context!s}.',
            context,
        )
    try:
        return str(val)
    except ValueError as err:
        raise _conversion_error(key, val, "str") from err


def validate_float(val: str, key: str) -> float:
    try:
        return float(val)
    except ValueError as err:
        raise _conversion_error(key, val, "float") from err


def validate_int(val: str, key: str) -> int:
    try:
        return int(val)
    except ValueError as err:
        raise _conversion_error(key, val, "int") from err


def validate_positive_float(
    val: str, key: str, strictly_positive: bool = False
) -> float:
    v = validate_float(val, key)
    if v < 0 or (v <= 0 and strictly_positive):
        raise ObservationConfigError.with_context(
            f'Failed to validate "{val}" in {key}={val}.'
            f" {key} must be given a "
            f"{'strictly ' if strictly_positive else ''}positive value.",
            val,
        )
    return v


def validate_rft_localization(val: dict[str, Any], context: FileContextToken) -> None:
    errors = []
    if "EAST" in val:
        errors.append(_invalid_rft_localization_key_error("EAST", context))
    if "NORTH" in val:
        errors.append(_invalid_rft_localization_key_error("NORTH", context))
    errors.extend(
        _unknown_key_error(key, context)
        for key in val
        if key not in {"EAST", "NORTH", "RADIUS"}
    )
    if errors:
        raise ObservationConfigError.from_collected(errors)


def validate_localization(val: dict[str, Any], context: FileContextToken) -> None:
    errors = []
    localization_context = context.update(value=f"LOCALIZATION for {context!s}")
    if "EAST" not in val:
        errors.append(_missing_value_error(localization_context, "EAST"))
    if "NORTH" not in val:
        errors.append(_missing_value_error(localization_context, "NORTH"))
    errors.extend(
        _unknown_key_error(key, localization_context)
        for key in val
        if key not in {"EAST", "NORTH", "RADIUS"}
    )
    if errors:
        raise ObservationConfigError.from_collected(errors)


def extract_localization_values(value: Mapping[str, Any]) -> tuple[float | None, ...]:
    east = value.get("EAST")
    north = value.get("NORTH")
    radius = value.get("RADIUS")
    east = validate_float(east, "EAST") if east is not None else None
    north = validate_float(north, "NORTH") if north is not None else None
    radius = validate_float(radius, "RADIUS") if radius is not None else None
    return east, north, radius


def validate_positive_int(val: str, key: str) -> int:
    try:
        v = int(val)
    except ValueError as err:
        raise _conversion_error(key, val, "int") from err
    if v < 0:
        raise ObservationConfigError.with_context(
            f'Failed to validate "{val}" in {key}={val}.'
            f" {key} must be given a positive value.",
            val,
        )
    return v


def _missing_value_error(
    context: FileContextToken, value_key: str
) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f'Missing item "{value_key}" in {context!s}', context
    )


def _unknown_csv_column(
    context: FileContextToken, column_name: str, filename: str | None
) -> ObservationConfigError:
    if column_name == "name":
        column_name = "well"
    return ObservationConfigError.with_context(
        f"Unrecognized column '{column_name}' in CSV file '{filename}' for {context!s}",
        context,
    )


def _missing_csv_column(
    context: FileContextToken, value_key: str, filename: str
) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f'Missing column "{value_key}" in csv file "{filename}"', context
    )


def _resolve_path(directory: str, filename: str) -> str:
    if not os.path.isabs(filename):
        filename = os.path.join(directory, filename)
    if not pathlib.Path(filename).exists():
        raise ObservationConfigError.with_context(
            f"The following keywords did not resolve to a valid path:\n {filename}",
            filename,
        )
    return filename


def _conversion_error(token: str, value: Any, type_name: str) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f'Could not convert {value} to {type_name}. Failed to validate "{value}"',
        token,
    )


def _unknown_key_error(key: str, context: FileContextToken) -> ObservationConfigError:
    raise ObservationConfigError.with_context(f"Unknown {key} in {context!s}", context)


def _unknown_observation_type_error(obs: ObservationDict) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f"Unexpected type in observations {obs}", obs.context
    )


def _invalid_rft_localization_key_error(
    key: str, context: FileContextToken
) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f"Invalid key: '{key}' in LOCALIZATION for RFT_OBSERVATION. "
        f"The '{key}' keyword must be defined outside the LOCALIZATION section for "
        f"RFT observations - or in the CSV RFT configuration file.",
        context,
    )
