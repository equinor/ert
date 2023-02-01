from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import xtgeo

from ert._c_wrappers.enkf.enums import EnkfTruncationType

if TYPE_CHECKING:
    import numpy.typing as npt


def _field_transform(data: npt.ArrayLike, transform_name: str) -> Any:
    if not transform_name:
        return data

    def f(x: float) -> float:  # pylint: disable=too-many-return-statements
        if transform_name in ("LN", "LOG"):
            return math.log(x, math.e)
        if transform_name == "LN0":
            return math.log(x, math.e) + 0.000001
        if transform_name == "LOG10":
            return math.log(x, 10)
        if transform_name == "EXP":
            return math.exp(x)
        if transform_name == "EXP0":
            return math.exp(x) + 0.000001
        if transform_name == "POW10":
            return math.pow(x, 10)
        if transform_name == "TRUNC_POW10":
            return math.pow(max(x, 0.001), 10)
        return x

    vfunc = np.vectorize(f)

    return vfunc(data)


def _field_truncate(
    data: npt.ArrayLike, truncation_mode: EnkfTruncationType, min_: float, max_: float
) -> Any:
    if truncation_mode == EnkfTruncationType.TRUNCATE_MIN:
        vfunc = np.vectorize(lambda x: max(x, min_))
        return vfunc(data)
    if truncation_mode == EnkfTruncationType.TRUNCATE_MAX:
        vfunc = np.vectorize(lambda x: min(x, max_))
        return vfunc(data)
    if (
        truncation_mode
        == EnkfTruncationType.TRUNCATE_MAX | EnkfTruncationType.TRUNCATE_MIN
    ):
        vfunc = np.vectorize(lambda x: max(min(x, max_), min_))
        return vfunc(data)
    return data


class Storage:
    def __init__(self, mount_point: Path) -> None:
        self.mount_point = mount_point

    def has_parameters(self) -> bool:
        """
        Checks if a parameter folder has been created
        """
        if Path(self.mount_point / "gen-kw.nc").exists():
            return True

        return False

    def save_gen_kw(  # pylint: disable=R0913
        self,
        parameter_name: str,
        parameter_keys: List[str],
        parameter_transfer_functions: List[Dict[str, Union[str, Dict[str, float]]]],
        realizations: List[int],
        data: npt.ArrayLike,
    ) -> None:
        ds = xr.Dataset(
            {
                parameter_name: ((f"{parameter_name}_keys", "iens"), data),
            },
            coords={f"{parameter_name}_keys": parameter_keys, "iens": realizations},
        )
        mode: Literal["a", "w"] = (
            "a" if Path.exists(self.mount_point / "gen-kw.nc") else "w"
        )

        ds.to_netcdf(self.mount_point / "gen-kw.nc", mode=mode, engine="scipy")
        priors = {}
        if Path.exists(self.mount_point / "gen-kw-priors.json"):
            with open(
                self.mount_point / "gen-kw-priors.json", "r", encoding="utf-8"
            ) as f:
                priors = json.load(f)
        priors.update({parameter_name: parameter_transfer_functions})
        with open(self.mount_point / "gen-kw-priors.json", "w", encoding="utf-8") as f:
            json.dump(priors, f)

    def load_gen_kw_priors(
        self,
    ) -> Dict[str, List[Dict[str, Union[str, Dict[str, float]]]]]:
        with open(self.mount_point / "gen-kw-priors.json", "r", encoding="utf-8") as f:
            priors: Dict[
                str, List[Dict[str, Union[str, Dict[str, float]]]]
            ] = json.load(f)
        return priors

    def load_gen_kw_realization(
        self, key: str, realization: int
    ) -> Tuple[npt.NDArray[np.double], List[str]]:
        input_file = self.mount_point / "gen-kw.nc"
        if not input_file.exists():
            raise KeyError(f"Unable to load GEN_KW for key: {key}")
        with xr.open_dataset(input_file, engine="scipy") as ds_disk:
            np_data = ds_disk.sel(iens=realization)[key].to_numpy()
            keys = list(ds_disk[key][f"{key}_keys"].values)

        return np_data, keys

    def save_summary_data(
        self,
        data: npt.NDArray[np.double],
        keys: List[str],
        axis: List[Any],
        realization: int,
    ) -> None:
        output_path = self.mount_point / f"summary-{realization}"
        Path.mkdir(output_path, exist_ok=True)

        ds = xr.Dataset(
            {"data": (("key", "time"), data)},
            coords={
                "key": keys,
                "time": axis,
            },
        )

        ds.to_netcdf(output_path / "data.nc", engine="scipy")

    def load_summary_data(
        self, summary_keys: List[str], realizations: List[int]
    ) -> Tuple[npt.NDArray[np.double], List[datetime], List[int]]:
        result = []
        loaded = []
        dates: List[datetime] = []
        for realization in realizations:
            input_path = self.mount_point / f"summary-{realization}"
            if not input_path.exists():
                continue
            loaded.append(realization)
            with xr.open_dataset(input_path / "data.nc", engine="scipy") as ds_disk:
                np_data = ds_disk["data"].to_numpy()
                keys = list(ds_disk["data"]["key"].values)
                if not dates:
                    dates = list(ds_disk["data"]["time"].values)
            indices = [keys.index(summary_key) for summary_key in summary_keys]
            selected_data = np_data[indices, :]

            result.append(selected_data.reshape(1, len(selected_data) * len(dates)).T)
        if not result:
            return np.array([]), dates, loaded
        return np.concatenate(result, axis=1), dates, loaded

    def load_summary_data_as_df(
        self, summary_keys: List[str], realizations: List[int]
    ) -> pd.DataFrame:
        data, time_axis, realizations = self.load_summary_data(
            summary_keys, realizations
        )
        if not data.any():
            raise KeyError(f"Unable to load SUMMARY_DATA for keys: {summary_keys}")
        multi_index = pd.MultiIndex.from_product(
            [summary_keys, time_axis], names=["data_key", "axis"]
        )
        return pd.DataFrame(
            data=data,
            index=multi_index,
            columns=realizations,
        )

    def save_gen_data(self, data: Dict[str, List[float]], realization: int) -> None:
        output_path = self.mount_point / f"gen-data-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        ds = xr.Dataset(
            data,
        )

        ds.to_netcdf(output_path / "data.nc", engine="scipy")

    def load_gen_data(
        self, key: str, realizations: List[int]
    ) -> Tuple[npt.NDArray[np.double], List[int]]:
        result = []
        loaded = []
        for realization in realizations:
            input_path = self.mount_point / f"gen-data-{realization}"
            if not input_path.exists():
                continue

            with xr.open_dataset(input_path / "data.nc", engine="scipy") as ds_disk:
                np_data = ds_disk[key].as_numpy()
                result.append(np_data)
                loaded.append(realization)
        if not result:
            raise KeyError(f"Unable to load GEN_DATA for key: {key}")
        return np.stack(result).T, loaded

    def load_gen_data_as_df(
        self, keys: List[str], realizations: List[int]
    ) -> pd.DataFrame:
        dfs = []
        for key in keys:
            data, realizations = self.load_gen_data(key, realizations)
            x_axis = [*range(data.shape[0])]
            multi_index = pd.MultiIndex.from_product(
                [[key], x_axis], names=["data_key", "axis"]
            )
            dfs.append(
                pd.DataFrame(
                    data=data,
                    index=multi_index,
                    columns=realizations,
                )
            )
        return pd.concat(dfs)

    def save_field_info(  # pylint: disable=too-many-arguments
        self,
        key: str,
        grid_file: Optional[str],
        transfer_out: str,
        truncation_mode: EnkfTruncationType,
        trunc_min: float,
        trunc_max: float,
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        input_path = self.mount_point / "field-info"
        Path.mkdir(input_path, exist_ok=True)
        info = {
            key: {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "transfer_out": transfer_out,
                "truncation_mode": truncation_mode,
                "truncation_min": trunc_min,
                "truncation_max": trunc_max,
            }
        }
        if grid_file is not None and not (input_path / "field-info.egrid").exists():
            shutil.copy(grid_file, input_path / "field-info.egrid")

        existing_info = {}
        if (input_path / "field-info.json").exists():
            with open(input_path / "field-info.json", encoding="utf-8", mode="r") as f:
                existing_info = json.load(f)
        existing_info.update(info)

        with open(input_path / "field-info.json", encoding="utf-8", mode="w") as f:
            json.dump(existing_info, f)

    def save_field_data(
        self,
        parameter_name: str,
        realization: int,
        data: npt.ArrayLike,
    ) -> None:
        output_path = self.mount_point / f"field-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        np.save(f"{output_path}/{parameter_name}", data)

    def load_field(self, key: str, realizations: List[int]) -> npt.NDArray[np.double]:
        result = []
        for realization in realizations:
            input_path = self.mount_point / f"field-{realization}"
            if not input_path.exists():
                raise KeyError(f"Unable to load FIELD for key: {key}")
            data = np.load(input_path / f"{key}.npy")
            result.append(data)
        return np.stack(result).T  # type: ignore

    def field_has_data(self, key: str, realization: int) -> bool:
        path = self.mount_point / f"field-{realization}/{key}.npy"
        return path.exists()

    def field_has_info(self, key: str) -> bool:
        path = self.mount_point / "field-info" / "field-info.json"
        if not path.exists():
            return False
        with open(path, encoding="utf-8", mode="r") as f:
            info = json.load(f)
        return key in info

    def export_field(
        self, key: str, realization: int, output_path: Path, fformat: str
    ) -> None:
        info_path = self.mount_point / "field-info"

        with open(info_path / "field-info.json", encoding="utf-8", mode="r") as f:
            info = json.load(f)[key]

        if (info_path / "field-info.egrid").exists():
            grid = xtgeo.grid_from_file(info_path / "field-info.egrid")
        else:
            grid = None

        input_path = self.mount_point / f"field-{realization}"

        if not input_path.exists():
            raise KeyError(
                f"Unable to load FIELD for key: {key}, realization: {realization} "
            )
        data = np.load(input_path / f"{key}.npy")

        data_transformed = _field_transform(data, transform_name=info["transfer_out"])
        data_truncated = _field_truncate(
            data_transformed,
            info["truncation_mode"],
            info["truncation_min"],
            info["truncation_max"],
        )

        gp = xtgeo.GridProperty(
            ncol=info["nx"],
            nrow=info["ny"],
            nlay=info["nz"],
            values=data_truncated,
            grid=grid,
            name=key,
        )

        os.makedirs(Path(output_path).parent, exist_ok=True)

        gp.to_file(output_path, fformat=fformat)

    def export_field_many(
        self,
        key: str,
        realizations: List[int],
        output_path: str,
        fformat: str,
    ) -> None:
        for realization in realizations:
            file_name = Path(output_path % realization)
            try:
                self.export_field(key, realization, file_name, fformat)
                print(f"{key}[{realization:03d}] -> {file_name}")
            except ValueError:
                sys.stderr.write(
                    f"ERROR: Could not load field: {key}, "
                    f"realization:{realization} - export failed"
                )
                pass
