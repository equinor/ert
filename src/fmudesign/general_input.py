from pathlib import Path
from typing import Literal, Self

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveInt,
    field_serializer,
)

from .config_validation import SeedStrategy
from .utils import resolve_path


class GeneralInput(BaseModel):
    designtype: Literal["onebyone"]
    repeats: PositiveInt
    distribution_seed: NonNegativeInt | None
    rms_seeds: FilePath | Literal["default"] | None
    correlation_iterations: NonNegativeInt = 0
    seed_strategy: SeedStrategy = Field(
        default=SeedStrategy.JOINT, validate_default=True
    )
    background: FilePath | str | None = None

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    @staticmethod
    def _read_general_input(
        input_filename: str, general_input_sheet: str
    ) -> dict[str, str | None]:
        df = pl.read_excel(
            input_filename,
            sheet_name=general_input_sheet,
            has_header=False,
            read_options={"dtypes": "string"},
            columns=[0, 1],
        )
        df = df.with_columns(
            pl.col(col).str.strip_chars().alias(col) for col in df.columns
        ).with_columns(
            pl.when(
                pl.col(df.columns[1])
                .str.to_lowercase()
                .is_in({"none", "null", "na", "nan"})
            )
            .then(None)
            .otherwise(pl.col(df.columns[1]))
            .alias(df.columns[1])
        )
        df = df.filter(pl.any_horizontal(pl.all().is_not_null()))
        df = df.with_columns(pl.col(df.columns[0]).fill_null(""))
        return dict(df.rows())

    @classmethod
    def from_xlsx(cls, input_filename: str, general_input_sheet: str) -> Self:
        input_dict = cls._read_general_input(input_filename, general_input_sheet)
        return cls.from_dict(
            input_dict,
            input_filename,
        )

    @classmethod
    def from_dict(
        cls, input_dict: dict[str, str | None], input_filename: str = ""
    ) -> Self:
        general_input: dict[str, str | Path | None] = dict(input_dict.items())

        for key in ["seed_strategy", "correlation_iterations"]:
            if general_input.get(key) is None:
                print(
                    f"'{key}' not set in general input sheet. "
                    f"Setting to default "
                    f"{GeneralInput.model_fields[key].default}."
                )
                general_input.pop(key, None)

        for key in ["rms_seeds", "background"]:
            if isinstance((val := general_input.get(key)), str):
                resolved = resolve_path(val, base_file=input_filename)
                assert isinstance(resolved, str)
                general_input[key] = (
                    Path(resolved) if Path(resolved).is_file() else resolved
                )

        return cls(**general_input)

    @field_serializer("rms_seeds", "background")
    def serialize_paths(self, field: Path | None) -> str | None:
        if field is None:
            return None
        return str(field)
