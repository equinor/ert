from typing import Any, Literal, Self

import polars as pl
from polars.datatypes import DataTypeClass
from pydantic import BaseModel

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
