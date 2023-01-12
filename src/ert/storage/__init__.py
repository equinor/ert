from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np
import pandas as pd
import xtgeo

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert._c_wrappers.enkf.config import FieldConfig


class Storage:
    def __init__(self, mount_point: Path) -> None:
        self.mount_point = mount_point

    def _has_parameters(self) -> bool:
        """
        Checks if a parameter folder has been created
        """
        for path in self.mount_point.iterdir():
            if "gen-kw" in str(path):
                return True
        return False

    def save_gen_kw(
        self,
        parameter_name: str,
        parameter_keys: List[str],
        realization: int,
        data: npt.ArrayLike,
    ) -> None:
        output_path = self.mount_point / f"gen-kw-{realization}"
        Path.mkdir(output_path, exist_ok=True)

        np.save(output_path / parameter_name, data)
        with open(output_path / f"{parameter_name}-keys", "w", encoding="utf-8") as f:
            f.write("\n".join(parameter_keys))

    def save_summary_data(
        self,
        data: npt.NDArray[np.double],
        keys: List[str],
        axis: List[Any],
        realization: int,
    ) -> None:
        output_path = self.mount_point / f"summary-{realization}"
        Path.mkdir(output_path, exist_ok=True)

        np.save(output_path / "data", data)
        with open(output_path / "keys", "w", encoding="utf-8") as f:
            f.write("\n".join(keys))

        with open(output_path / "time_map", "w", encoding="utf-8") as f:
            f.write("\n".join([t.strftime("%Y-%m-%d") for t in axis]))

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
            np_data = np.load(input_path / "data.npy")

            keys = []
            with open(input_path / "keys", "r", encoding="utf-8") as f:
                keys = [k.strip() for k in f.readlines()]
            if not dates:
                with open(input_path / "time_map", "r", encoding="utf-8") as f:
                    dates = [
                        datetime.strptime(k.strip(), "%Y-%m-%d") for k in f.readlines()
                    ]
            indices = [keys.index(summary_key) for summary_key in summary_keys]
            selected_data = np_data[indices, :]

            result.append(selected_data.reshape(1, len(indices) * len(dates)).T)
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

    def save_gen_data(
        self, key: str, data: List[List[float]], realization: int
    ) -> None:
        output_path = self.mount_point / f"gen-data-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        np_data = np.array(data)
        np.save(output_path / key, np_data)

    def load_gen_data(
        self, key: str, realizations: List[int]
    ) -> Tuple[npt.NDArray[np.double], List[int]]:
        result = []
        loaded = []
        for realization in realizations:
            input_path = self.mount_point / f"gen-data-{realization}" / f"{key}.npy"
            if not input_path.exists():
                continue

            np_data = np.load(input_path)

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

    def export_field(
        self, config_node: FieldConfig, realization: int, output_path: str, fformat: str
    ) -> None:
        input_path = self.mount_point / f"field-{realization}"
        key = config_node.get_key()

        if not input_path.exists():
            raise KeyError(
                f"Unable to load FIELD for key: {key}, realization: {realization} "
            )
        data = np.load(input_path / f"{key}.npy")

        transform_name = config_node.get_output_transform_name()
        data_transformed = config_node.transform(transform_name, data)
        data_truncated = config_node.truncate(data_transformed)

        gp = xtgeo.GridProperty(
            ncol=config_node.get_nx(),
            nrow=config_node.get_ny(),
            nlay=config_node.get_nz(),
            values=data_truncated,
            grid=config_node.get_grid(),
            name=key,
        )

        os.makedirs(Path(output_path).parent, exist_ok=True)

        gp.to_file(output_path, fformat=fformat)

    def export_field_many(
        self,
        config_node: FieldConfig,
        realizations: List[int],
        output_path: str,
        fformat: str,
    ) -> None:
        for realization in realizations:
            file_name = output_path % realization
            try:
                self.export_field(config_node, realization, file_name, fformat)
                print(f"{config_node.get_key()}[{realization:03d}] -> {file_name}")
            except ValueError:
                sys.stderr.write(
                    f"ERROR: Could not load realisation:{realization} - export failed"
                )
                pass
