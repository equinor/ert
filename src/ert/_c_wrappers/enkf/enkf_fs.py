import logging
import math
import shutil
from datetime import datetime
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from cwrap import BaseCClass

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config import EnkfConfigNode
from ert._c_wrappers.enkf.enums import EnKFFSType, RealizationStateEnum
from ert._c_wrappers.enkf.model_callbacks import LoadStatus
from ert._c_wrappers.enkf.ert_config import EnsembleConfig
from ert._c_wrappers.enkf.time_map import TimeMap
from ert._clib import update
from ert.ensemble_evaluator.callbacks import forward_model_ok

if TYPE_CHECKING:
    from ecl.summary import EclSum
    from ecl.util.util import IntVector

    from ert._c_wrappers.enkf.config import GenKwConfig
    from ert._c_wrappers.enkf.res_config import ModelConfig
    from ert._c_wrappers.enkf.run_arg import RunArg
    from ert._c_wrappers.enkf.state_map import StateMap

logger = logging.getLogger(__name__)


def _load_realization(
    sim_fs: "EnkfFs",
    realisation: int,
    ensemble_config: "EnsembleConfig",
    model_config: "ModelConfig",
    run_args: List["RunArg"],
) -> Tuple["LoadStatus", str]:
    state_map = sim_fs.getStateMap()

    state_map.update_matching(
        realisation,
        RealizationStateEnum.STATE_UNDEFINED,
        RealizationStateEnum.STATE_INITIALIZED,
    )
    status = forward_model_ok(run_args[realisation], ensemble_config, model_config)
    state_map._set(
        realisation,
        RealizationStateEnum.STATE_HAS_DATA
        if status[0] == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStateEnum.STATE_LOAD_FAILURE,
    )
    return status, realisation


class EnkfFs(BaseCClass):
    TYPE_NAME = "enkf_fs"

    _mount = ResPrototype("void* enkf_fs_mount(char*, int, bool)", bind=False)
    _sync = ResPrototype("void enkf_fs_sync(enkf_fs)")
    _fsync = ResPrototype("void  enkf_fs_fsync(enkf_fs)")
    _create = ResPrototype(
        "void*   enkf_fs_create_fs(char* , enkf_fs_type_enum ,int, bool)",
        bind=False,
    )
    _umount = ResPrototype("void enkf_fs_umount(enkf_fs)")

    def __init__(
        self,
        mount_point: Union[str, Path],
        ensemble_config: EnsembleConfig,
        ensemble_size: int,
        read_only: bool = False,
        refcase: Optional["EclSum"] = None,
    ):
        self.mount_point = Path(mount_point).absolute()
        self.case_name = self.mount_point.stem
        self.refcase = refcase
        c_ptr = self._mount(self.mount_point.as_posix(), ensemble_size, read_only)
        super().__init__(c_ptr)
        if self.refcase:
            time_map = self.getTimeMap()
            time_map.attach_refcase(self.refcase)
        self._ensemble_config = ensemble_config
        self._ensemble_size = ensemble_size

    def getTimeMap(self) -> TimeMap:
        return _clib.enkf_fs.get_time_map(self)

    def getStateMap(self) -> "StateMap":
        return _clib.enkf_fs.get_state_map(self)

    def getCaseName(self) -> str:
        return self.case_name

    @property
    def is_initalized(self) -> bool:
        return (
            _clib.enkf_fs.is_initialized(
                self,
                self._ensemble_config,
                self._ensemble_config.parameters,
                self._ensemble_size,
            )
            or self._has_parameters()
        )

    @classmethod
    def createFileSystem(
        cls,
        path: Union[str, Path],
        ensemble_config: "EnsembleConfig",
        ensemble_size: int,
        read_only: bool = False,
        refcase: Optional["EclSum"] = None,
    ) -> "EnkfFs":
        path = Path(path).absolute()
        fs_type = EnKFFSType.BLOCK_FS_DRIVER_ID
        cls._create(path.as_posix(), fs_type, ensemble_size, False)
        return cls(
            path, ensemble_config, ensemble_size, read_only=read_only, refcase=refcase
        )

    def sync(self):
        self._sync()

    def free(self):
        self._umount()

    def __repr__(self):
        return f"EnkfFs(case_name = {self.getCaseName()}) {self._ad_str()}"

    def fsync(self):
        self._fsync()

    def getSummaryKeySet(self) -> List[str]:
        summary_folders = list(self.mount_point.glob("summary-*"))
        if not summary_folders:
            return []
        summary_path = summary_folders[0]
        with open(summary_path / "keys", "r", encoding="utf-8") as f:
            keys = [k.strip() for k in f.readlines()]
        return sorted(keys)

    def realizationList(self, state: "RealizationStateEnum") -> "IntVector":
        """
        Will return list of realizations with state == the specified state.
        """
        state_map = self.getStateMap()
        return state_map.realizationList(state)

    def _has_parameters(self) -> bool:
        """
        Checks if a parameter folder has been created
        """
        for path in self.mount_point.iterdir():
            if "gen-kw" in str(path):
                return True

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
        with open(output_path / f"{parameter_name}-keys", "w") as f:
            f.write("\n".join(parameter_keys))

        self.getStateMap().update_matching(
            realization,
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_INITIALIZED,
        )

    def _load_gen_kw_realization(
        self, key: str, realization: int
    ) -> Tuple["npt.NDArray[np.double]", List[str]]:
        input_path = self.mount_point / f"gen-kw-{realization}"
        if not input_path.exists():
            raise KeyError(f"Unable to load GEN_KW for key: {key}")

        np_data = np.load(input_path / f"{key}.npy")
        with open(input_path / f"{key}-keys", "r") as f:
            keys = [k.strip() for k in f.readlines()]

        return np_data, keys

    def load_gen_kw_as_dict(
        self, key: str, realization: int, gen_kw_config: "GenKwConfig"
    ) -> Dict[str, float]:
        data, keys = self._load_gen_kw_realization(key, realization)

        transformed = {
            parameter_key: gen_kw_config.transform(index, value)
            for index, (parameter_key, value) in enumerate(zip(keys, data))
        }

        result = {key: transformed}

        log10 = {
            parameter_key: math.log(value, 10)
            for index, (parameter_key, value) in enumerate(transformed.items())
            if gen_kw_config.shouldUseLogScale(index)
        }
        if log10:
            result.update({f"LOG10_{key}": log10})
        return result

    def load_gen_kw(
        self, key: str, realizations: List[int]
    ) -> "npt.NDArray[np.double]":
        result = []
        for realization in realizations:
            data, _ = self._load_gen_kw_realization(key, realization)
            result.append(data)
        return np.stack(result).T

    def save_parameters(
        self,
        config_node: "EnkfConfigNode",
        iens_active_index: List[int],
        parameter: update.Parameter,
        values: npt.ArrayLike,
    ) -> None:
        update.save_parameter(self, config_node, iens_active_index, parameter, values)

    def load_parameter(
        self,
        config_node: EnkfConfigNode,
        iens_active_index: List[int],
        parameter: update.Parameter,
    ) -> np.ndarray:
        return update.load_parameter(self, config_node, iens_active_index, parameter)

    def copy_from_case(
        self, other: "EnkfFs", report_step: int, nodes: List[str], active: List[bool]
    ) -> None:
        """
        This copies parameters from self into other, checking if nodes exists
        in self before performing copy.
        """
        _clib.enkf_fs.copy_from_case(
            self,
            self._ensemble_config,
            other,
            report_step,
            nodes,
            active,
        )
        self._copy_parameter_files(other, nodes, [i for i, b in enumerate(active) if b])

    def _copy_parameter_files(
        self, to: "EnkfFs", parameter_keys: List[str], realizations: List[int]
    ) -> None:
        """
        Copies selected parameter files from one storage to another.
        Filters on realization and parameter keys
        """
        for gen_kw_folder in self.mount_point.glob("gen-kw-*"):
            files_to_copy = []
            realization = int(str(gen_kw_folder).rsplit("-", maxsplit=1)[-1])
            if realization in realizations:
                for parameter_file in gen_kw_folder.iterdir():
                    base_name = str(parameter_file.stem)
                    if base_name in parameter_keys:
                        files_to_copy.append(parameter_file.name)
                        files_to_copy.append(f"{base_name}-keys")

            if not files_to_copy:
                continue

            Path.mkdir(to.mount_point / gen_kw_folder.stem)
            for f in files_to_copy:
                shutil.copy(
                    src=self.mount_point / gen_kw_folder.stem / f,
                    dst=to.mount_point / gen_kw_folder.stem / f,
                )

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
    ) -> Tuple["npt.NDArray[np.double]", List[str], List[int]]:
        result = []
        loaded = []
        dates = []
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

    def load_from_run_path(
        self,
        ensemble_size: int,
        ensemble_config: "EnsembleConfig",
        model_config: "ModelConfig",
        run_args: List["RunArg"],
        active_realizations: List[bool],
    ) -> int:
        """Returns the number of loaded realizations"""
        pool = ThreadPool(processes=8)

        async_result = [
            pool.apply_async(
                _load_realization, (self, iens, ensemble_config, model_config, run_args)
            )
            for iens in range(ensemble_size)
            if active_realizations[iens]
        ]

        loaded = 0
        for t in async_result:
            ((status, message), iens) = t.get()

            if status == LoadStatus.LOAD_SUCCESSFUL:
                loaded += 1
            else:
                logger.error(f"Realization: {iens}, load failure: {message}")

        return loaded
