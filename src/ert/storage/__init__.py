from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd


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
        data: "npt.ArrayLike",
    ) -> None:
        output_path = self.mount_point / f"gen-kw-{realization}"
        Path.mkdir(output_path, exist_ok=True)

        np.save(output_path / parameter_name, data)
        with open(output_path / f"{parameter_name}-keys", "w", encoding="utf-8") as f:
            f.write("\n".join(parameter_keys))

    def save_summary_data(
        self,
        data: "npt.NDArray[np.double]",
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
    ) -> Tuple["npt.NDArray[np.double]", List[datetime], List[int]]:
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
    ) -> Tuple["npt.NDArray[np.double]", List[int]]:
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
