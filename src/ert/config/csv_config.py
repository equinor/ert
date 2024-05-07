import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import xarray as xr
from typing_extensions import Self

from ._option_dict import option_dict
from .parsing import ConfigValidationError
from .response_config import ResponseConfig


@dataclass
class CSVConfig(ResponseConfig):
    input_file: str = ""

    @classmethod
    def from_config_list(cls, gen_data: List[str]) -> Self:
        options = option_dict(gen_data, 1)
        name = gen_data[0]
        res_file = options.get("RESULT_FILE")

        if res_file is None:
            raise ConfigValidationError.with_context(
                f"Missing or unsupported RESULT_FILE for CSV_RESPONSE key {name!r}",
                name,
            )
        if os.path.isabs(res_file):
            result_file_context = next(
                x for x in gen_data if x.startswith("RESULT_FILE:")
            )
            raise ConfigValidationError.with_context(
                f"The RESULT_FILE:{res_file} setting for {name} is "
                f"invalid - must be a relative path",
                result_file_context,
            )
        return cls(name=name, input_file=res_file)

    def read_from_file(self, run_path: str, _: int) -> xr.Dataset:
        filename = Path(run_path) / self.input_file
        if not filename.exists():
            raise ValueError(f"Missing output file: {filename}")
        data = pd.read_csv(filename)
        return xr.Dataset.from_dataframe(data)
