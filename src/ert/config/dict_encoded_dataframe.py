from __future__ import annotations

from typing import Annotated, Any, Literal, Self

import polars as pl
from polars.datatypes import DataTypeClass
from pydantic import BaseModel, PlainSerializer, PlainValidator


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
        return pl.from_dicts(
            self.data,
            schema={
                col: str_to_dtype(dtype_str)
                for col, dtype_str in self.datatypes.items()
            },
        )


def _validate_dataframe(value: Any) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value
    return DictEncodedDataFrame.model_validate(value).to_polars()


def _serialize_dataframe(value: pl.DataFrame) -> dict[str, Any]:
    return DictEncodedDataFrame.from_polars(value).model_dump(mode="json")


SerializableDataFrame = Annotated[
    pl.DataFrame,
    PlainValidator(_validate_dataframe),
    PlainSerializer(_serialize_dataframe),
]
