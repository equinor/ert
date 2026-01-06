import json
import shutil
from pathlib import Path
from typing import Any, Literal, Self

import polars as pl
from polars.datatypes import DataTypeClass
from pydantic import BaseModel

info = (
    "Move parameters.json, responses.json, observations contents to be in "
    "experiment index.json .experiment field"
)

# https://github.com/pola-rs/polars/issues/13152#issuecomment-1864600078
# PS: Serializing/deserializing schema is scheduled to be added to polars core,
# ref https://github.com/pola-rs/polars/issues/20426
# then this workaround can be omitted.


def str_to_dtype(dtype_str: str) -> pl.DataType:
    dtype = eval(f"pl.{dtype_str}")
    if isinstance(dtype, DataTypeClass):
        dtype = dtype()
    return dtype


class DictEncodedDataFrame(BaseModel):
    type: Literal["dicts"]
    data: list[dict[str, Any]]
    datatypes: dict[str, str]

    @classmethod
    def from_polars(cls, data: pl.DataFrame) -> Self:
        str_schema = {k: str(dtype) for k, dtype in data.schema.items()}
        return cls(type="dicts", data=data.to_dicts(), datatypes=str_schema)

    def to_polars(self) -> pl.DataFrame:
        final_schema = {
            col: str_to_dtype(dtype_str) for col, dtype_str in self.datatypes.items()
        }

        # For the initial read, any column destined to be a date or datetime
        # is read as a string (Utf8) to handle parsing. Others are read directly.
        read_schema = {
            col: pl.Utf8 if dtype.base_type() in {pl.Date, pl.Datetime} else dtype
            for col, dtype in final_schema.items()
        }

        df = pl.from_dicts(self.data, schema=read_schema, infer_schema_length=28)

        # Build a list of expressions to parse the string columns into their
        # proper date/datetime types.
        date_casts = [
            pl.col(col).str.to_datetime(
                time_unit=dtype.time_unit,  # type: ignore
                time_zone=dtype.time_zone,  # type: ignore
            )
            if dtype == pl.Datetime
            else pl.col(col).str.to_date()
            for col, dtype in final_schema.items()
            if dtype.base_type() in {pl.Date, pl.Datetime}
        ]

        # Apply the casting expressions if any were generated.
        if date_casts:
            df = df.with_columns(date_casts)

        return df


def migrate_parameters_responses_and_observations_into_experiment_index(
    path: Path,
) -> None:
    for experiment_path in path.glob("experiments/*"):
        experiment_json = {}
        with open(experiment_path / "metadata.json", encoding="utf-8") as fin:
            metadata_json = json.load(fin)
            if "weights" in metadata_json:
                experiment_json["weights"] = metadata_json["weights"]

        responses_contents = json.loads(
            (experiment_path / "responses.json").read_text(encoding="utf-8")
        )
        parameters_contents = json.loads(
            (experiment_path / "parameter.json").read_text(encoding="utf-8")
        )

        experiment_json["response_configuration"] = list(responses_contents.values())
        experiment_json["parameter_configuration"] = list(parameters_contents.values())

        # PS: This may be super slow for large observation datasets
        # Revisit later and consider keeping the files
        observations_dict = {}
        if (experiment_path / "observations").exists():
            for path_to_obs_file in (experiment_path / "observations").glob("*"):
                encoded_obs = DictEncodedDataFrame.from_polars(
                    pl.read_parquet(path_to_obs_file)
                ).model_dump(mode="json")
                response_type = path_to_obs_file.stem
                observations_dict[response_type] = encoded_obs

            experiment_json["observations"] = observations_dict

        with open(experiment_path / "index.json", encoding="utf-8") as fin:
            index_json = json.load(fin)
            index_json["experiment"] = experiment_json

            Path(experiment_path / "index.json").write_text(
                json.dumps(index_json, indent=2), encoding="utf-8"
            )

        (experiment_path / "metadata.json").unlink()
        (experiment_path / "responses.json").unlink()
        (experiment_path / "parameter.json").unlink()

        if (experiment_path / "observations").exists():
            shutil.rmtree(experiment_path / "observations")


def migrate(path: Path) -> None:
    migrate_parameters_responses_and_observations_into_experiment_index(path)
