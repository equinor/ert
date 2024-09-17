from __future__ import annotations

import logging
import mmap
import re
import struct
import warnings
import zlib
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Type,
    TypeVar,
)

import numpy as np
import xarray as xr
import xtgeo
import xtgeo.surface
from typing_extensions import Self

from ert.config import (
    EnsembleConfig,
    Field,
    GenDataConfig,
    GenKwConfig,
    ParameterConfig,
    ResponseConfig,
    SurfaceConfig,
    field_transform,
)
from ert.storage import Ensemble, Storage
from ert.storage.local_storage import LocalStorage, local_storage_get_ert_config
from ert.storage.mode import Mode

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy.typing as npt
    from xtgeo.surface import RegularSurface

    DType = TypeVar("DType", np.float32, np.float64)


_SIZEOF_I32 = 4
_SIZEOF_I64 = 8
_SIZEOF_F64 = 8


class Kind(IntEnum):
    FIELD = 104
    GEN_KW = 107
    SUMMARY = 110
    GEN_DATA = 113
    SURFACE = 114
    EXT_PARAM = 116


@dataclass
class Block:
    kind: Kind
    name: str
    report_step: int
    realization_index: int
    pos: int
    len: int
    count: int


def parse_name(name: str, kind: Kind) -> tuple[str, int, int]:
    if (index := name.rfind(".")) < 0:
        raise ValueError(f"Key '{name}' has no realization index")
    if kind == Kind.SUMMARY:
        return (name[:index], 0, int(name[index + 1 :]))
    if (index_ := name.rfind(".", 0, index - 1)) < 0:
        raise ValueError(f"Key '{name}' has no report step")
    return (name[:index_], int(name[index_ + 1 : index]), int(name[index + 1 :]))


class DataFile:
    def __init__(self, path: Path) -> None:
        self.blocks: dict[Kind, list[Block]] = defaultdict(list)
        self.realizations: set[int] = set()
        self._file = path.open("rb")
        self._pos = 0

        try:
            self._mmap = mmap.mmap(
                self._file.fileno(), 0, mmap.PROT_READ, mmap.MAP_SHARED
            )
        except ValueError:
            # "cannot mmap an empty file"
            return

        self._build_index()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if hasattr(self, "_mmap"):
            self._mmap.close()
        self._file.close()

    def _build_index(self) -> None:
        try:
            while True:
                self._seek_until_marker()

                name_length = self._read_u32()
                name = self._readn(name_length).decode("ascii")
                self._pos += 1  # NULL terminator

                # Skip node_size
                self._pos += _SIZEOF_I32
                data_size = self._read_i32()

                self._pos += _SIZEOF_I64
                data_size -= 8
                count = 0

                kind = Kind(self._read_i32())
                data_size -= _SIZEOF_I32
                if kind == Kind.SUMMARY:
                    # Read count
                    count = self._read_u32()
                    data_size -= _SIZEOF_I32

                    # Skip default value
                    self._pos += _SIZEOF_F64
                    data_size -= _SIZEOF_F64
                elif kind == Kind.GEN_DATA:
                    # Read count
                    count = self._read_u32()

                    # Skip report_step
                    self._pos += _SIZEOF_I32
                    data_size -= _SIZEOF_I32
                elif kind in (Kind.SURFACE, Kind.GEN_KW):
                    # The count is given in the config and not available in the
                    # data file, but we can make an informed guess by looking at
                    # the size of the whole data section
                    count = data_size // _SIZEOF_F64
                elif kind == Kind.FIELD:
                    # The count is given in the config
                    pass
                elif kind == Kind.EXT_PARAM:
                    raise RuntimeError("Migrating EXT_PARAM is not supported")
                else:
                    # Unknown Kind, continue
                    continue

                name_, report_step, realization_index = parse_name(name, kind)
                self.realizations.add(realization_index)
                self.blocks[kind].append(
                    Block(
                        kind=kind,
                        name=name_,
                        report_step=report_step,
                        realization_index=realization_index,
                        pos=self._pos,
                        len=data_size,
                        count=count,
                    )
                )

        except IndexError:
            # We have read the entire file, return
            return

    def load_field(self, block: Block, count_hint: int) -> npt.NDArray[np.float32]:
        return self._load_vector_compressed(block, np.float32, count_hint)

    def load(
        self, block: Block, count_hint: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        if (
            block.kind in (Kind.GEN_KW, Kind.SURFACE, Kind.EXT_PARAM)
            and count_hint is not None
            and count_hint != block.count
        ):
            raise ValueError(
                f"On-disk vector has {block.count} elements, but ERT config expects {count_hint}"
            )

        if block.kind == Kind.GEN_DATA:
            return self._load_vector_compressed(block, np.float64, block.count)
        else:
            return self._load_vector(block, block.count)

    def _load_vector_compressed(
        self, block: Block, dtype: Type[DType], count: int
    ) -> npt.NDArray[DType]:
        compressed = self._mmap[block.pos : block.pos + block.len]
        return np.frombuffer(zlib.decompress(compressed), dtype=dtype, count=count)

    def _load_vector(self, block: Block, count: int) -> npt.NDArray[np.float64]:
        return np.frombuffer(
            self._mmap[block.pos : block.pos + block.len], dtype=np.float64, count=count
        )

    def _read(self, fmt: str) -> Any:
        len = struct.calcsize(fmt)
        value = struct.unpack(fmt, self._mmap[self._pos : self._pos + len])
        self._pos += len
        return value

    def _readn(self, nbytes: int) -> bytes:
        data = self._mmap[self._pos : self._pos + nbytes]
        self._pos += nbytes
        return data

    def _read_u32(self) -> int:
        return self._read("I")[0]

    def _read_i32(self) -> int:
        return self._read("i")[0]

    def _seek_until_marker(self) -> None:
        count = 0
        while count < 4:
            char = self._mmap[self._pos]
            self._pos += 1
            count = count + 1 if char == 0x55 else 0


def migrate(path: Path) -> None:
    logger.info(f"Outdated storage detected at '{path}'. Migrating...")
    block_storage_path = _backup(path)

    statuses: List[bool] = []
    with LocalStorage(path, Mode.WRITE, ignore_migration_check=True) as storage:
        for casedir in sorted(block_storage_path.iterdir()):
            if (casedir / "ert_fstab").is_file():
                statuses.append(_migrate_case_ignoring_exceptions(storage, casedir))
    failures = len(statuses) - sum(statuses)
    logger.info(f"Migration from BlockFs completed with {failures} failure(s)")
    logger.info(
        "Note: ERT 4 and lower is not compatible with the migrated storage. "
        "To restore your old data, copy the contents of "
        f"'{block_storage_path}' to '{path}'."
    )


def _migrate_case_ignoring_exceptions(storage: Storage, casedir: Path) -> bool:
    try:
        with warnings.catch_warnings(record=True), ExitStack() as stack:
            migrate_case(storage, casedir, stack)
        return True
    except Exception as exc:
        logger.warning(
            (
                "Exception occurred during migration of BlockFs case "
                f"'{casedir.name}': {exc}"
            ),
            exc_info=exc,
        )
        return False


def migrate_case(storage: Storage, path: Path, stack: ExitStack) -> None:
    logger.info(f"Migrating case '{path.name}'")
    time_map = _load_timestamps(path / "files/time-map")

    parameter_files = [
        stack.push(DataFile(x)) for x in path.glob("Ensemble/mod_*/PARAMETER.data_0")
    ]
    response_files = [
        stack.push(DataFile(x)) for x in path.glob("Ensemble/mod_*/FORECAST.data_0")
    ]

    ensemble_size = _guess_ensemble_size(*parameter_files, *response_files)

    ert_config = local_storage_get_ert_config()
    ens_config = ert_config.ensemble_config

    if ensemble_size == 0:
        return

    # Copy experiment parameter data
    parameter_configs: List[ParameterConfig] = []
    for data_file in parameter_files:
        parameter_configs.extend(_migrate_field_info(data_file, ens_config))
        parameter_configs.extend(_migrate_gen_kw_info(data_file, ens_config))
        parameter_configs.extend(_migrate_surface_info(data_file, ens_config))

    # Copy experiment response data
    response_configs: List[ResponseConfig] = []
    for data_file in response_files:
        response_configs.extend(_migrate_summary_info(data_file, ens_config))
        response_configs.extend(_migrate_gen_data_info(data_file, ens_config))

    # Guess iteration number
    iteration = 0
    if (match := re.search(r"_(\d+)$", path.name)) is not None:
        iteration = int(match[1])

    experiment = storage.create_experiment(
        parameters=parameter_configs,
        responses=response_configs,
        observations=ert_config.observations,
        name="migrate-case",
    )
    ensemble = experiment.create_ensemble(
        name=path.name,
        ensemble_size=ensemble_size,
        iteration=iteration,
    )

    # Copy parameter data
    for data_file in parameter_files:
        _migrate_field(ensemble, data_file, ens_config)
        _migrate_gen_kw(ensemble, data_file, ens_config)
        _migrate_surface(ensemble, data_file, ens_config)

    # Copy response data
    for data_file in response_files:
        _migrate_summary(ensemble, data_file, time_map)
        _migrate_gen_data(ensemble, data_file)


def _backup(path: Path) -> Path:
    backup_path = path / f"_backup_{datetime.now().isoformat()}"
    backup_path.mkdir(parents=False, exist_ok=False)

    logger.info(f"Backing up BlockFs storage\n\t{path} -> {backup_path}")

    for subpath in path.iterdir():
        # Don't copy backup directory or the storage lock
        if subpath.name in (backup_path.name, "storage.lock"):
            continue
        subpath.rename(backup_path / subpath.name)

    return backup_path


def _guess_ensemble_size(*files: DataFile) -> int:
    try:
        return max(real for file in files for real in file.realizations) + 1
    except ValueError:
        return 0


def _load_timestamps(path: Path) -> npt.NDArray[np.datetime64]:
    if not path.exists():
        return np.ndarray((0,), dtype=np.float64)

    sizeof_time_t = 8

    with path.open("rb") as f:
        size = struct.unpack("I", f.read(4))[0]
        f.read(sizeof_time_t)  # time_t default_value; (unused)
        return np.frombuffer(f.read(size * sizeof_time_t), dtype="datetime64[s]")


def _migrate_surface_info(
    data_file: DataFile, ens_config: EnsembleConfig
) -> List[ParameterConfig]:
    seen = set()
    configs: List[ParameterConfig] = []
    for block in data_file.blocks[Kind.SURFACE]:
        if block.name in seen:
            continue
        seen.add(block.name)
        config = ens_config[block.name]
        assert isinstance(config, SurfaceConfig)
        configs.append(config)
    return configs


def _migrate_surface(
    ensemble: Ensemble,
    data_file: DataFile,
    ens_config: EnsembleConfig,
) -> None:
    surfaces: dict[str, RegularSurface] = {}
    for block in data_file.blocks[Kind.SURFACE]:
        config = ens_config[block.name]
        assert isinstance(config, SurfaceConfig)
        try:
            surface = surfaces[block.name]
        except KeyError:
            surface = surfaces[block.name] = xtgeo.surface_from_file(
                config.base_surface_path, fformat="irap_ascii", dtype=np.float32
            )
        array = data_file.load(block, np.prod(surface.dimensions)).reshape(
            surface.dimensions
        )
        ensemble.save_parameters(
            block.name,
            block.realization_index,
            xr.DataArray(array, name="values").to_dataset(),
        )


def _migrate_field_info(
    data_file: DataFile,
    ens_config: EnsembleConfig,
) -> List[ParameterConfig]:
    seen = set()
    configs: List[ParameterConfig] = []
    for block in data_file.blocks[Kind.FIELD]:
        if block.name in seen:
            continue
        seen.add(block.name)
        config = ens_config[block.name]
        assert isinstance(config, Field)

        configs.append(config)
    return configs


def _migrate_field(
    ensemble: Ensemble,
    data_file: DataFile,
    ens_config: EnsembleConfig,
) -> None:
    for block in data_file.blocks[Kind.FIELD]:
        config = ens_config[block.name]
        assert isinstance(config, Field)

        data_size = config.nx * config.ny * config.nz
        data = data_file.load_field(block, int(data_size))
        if config.output_transformation:
            data = field_transform(data, config.output_transformation)
        m_data = np.ma.MaskedArray(
            data.reshape((config.nx, config.ny, config.nz)),
            config.mask,
            fill_value=np.nan,
        )  # type: ignore
        ds = xr.Dataset({"values": (["x", "y", "z"], m_data)})
        ensemble.save_parameters(block.name, block.realization_index, ds)


def _migrate_summary_info(
    data_file: DataFile,
    ens_config: EnsembleConfig,
) -> List[ResponseConfig]:
    seen = set()
    for block in data_file.blocks[Kind.SUMMARY]:
        if block.name in seen:
            continue
        seen.add(block.name)
    return [ens_config["summary"]] if seen else []  # type: ignore


def _migrate_summary(
    ensemble: Ensemble,
    data_file: DataFile,
    time_map: npt.NDArray[np.datetime64],
) -> None:
    if len(time_map) == 0:
        return

    time_mask = time_map != np.datetime64(-1, "s")
    time_mask[0] = False  # report_step 0 is invalid

    data: dict[int, tuple[list[npt.NDArray[np.float64]], list[str]]] = defaultdict(
        lambda: ([], [])
    )
    for block in data_file.blocks[Kind.SUMMARY]:
        if block.name == "TIME":
            continue

        array, keys = data[block.realization_index]
        vector = data_file.load(block, 0)[time_mask]

        # Old Ert uses -9999 as stand-in for NaN
        NAN_STAND_IN = -9999.0
        vector[vector == -NAN_STAND_IN] = np.nan

        array.append(vector)
        keys.append(block.name)

    for realization_index, (array, keys) in data.items():
        ds = xr.Dataset(
            {"values": (["name", "time"], array)},
            coords={"time": time_map[time_mask], "name": keys},
        )
        ensemble.save_response("summary", ds, realization_index)


def _migrate_gen_data_info(
    data_file: DataFile,
    ens_config: EnsembleConfig,
) -> List[ResponseConfig]:
    seen = set()
    configs: List[ResponseConfig] = []
    for block in data_file.blocks[Kind.GEN_DATA]:
        if block.name in seen:
            continue
        seen.add(block.name)
        config = ens_config[block.name]
        assert isinstance(config, GenDataConfig)

        configs.append(config)
    return configs


def _migrate_gen_data(
    ensemble: Ensemble,
    data_file: DataFile,
) -> None:
    realizations = defaultdict(lambda: defaultdict(list))  # type: ignore
    for block in data_file.blocks[Kind.GEN_DATA]:
        realizations[block.realization_index][block.name].append(
            {"values": data_file.load(block, 0), "report_step": block.report_step}
        )
    for iens, gen_data in realizations.items():
        datasets = []
        for name, values in gen_data.items():
            dataset_fragments = []
            for value in values:
                dataset_fragments.append(
                    xr.Dataset(
                        {
                            "values": (
                                ["report_step", "index"],
                                [value["values"]],
                            )
                        },
                        coords={
                            "index": range(len(value["values"])),
                            "report_step": [value["report_step"]],
                        },
                    )
                )

            datasets.append(
                xr.combine_by_coords(dataset_fragments).expand_dims(name=[name])
            )

        ensemble.save_response(
            "gen_data",
            xr.concat(datasets, dim="name"),  # type: ignore
            iens,
        )


def _migrate_gen_kw_info(
    data_file: DataFile,
    ens_config: EnsembleConfig,
) -> List[ParameterConfig]:
    seen = set()
    configs: List[ParameterConfig] = []
    for block in data_file.blocks[Kind.GEN_KW]:
        if block.name in seen:
            continue
        seen.add(block.name)
        config = ens_config[block.name]
        assert isinstance(config, GenKwConfig)

        configs.append(config)
    return configs


def _migrate_gen_kw(
    ensemble: Ensemble, data_file: DataFile, ens_config: EnsembleConfig
) -> None:
    for block in data_file.blocks[Kind.GEN_KW]:
        config = ens_config[block.name]
        assert isinstance(config, GenKwConfig)

        priors = config.get_priors()
        array = data_file.load(block, len(priors))
        dataset = xr.Dataset(
            {
                "values": ("names", array),
                "transformed_values": ("names", config.transform(array)),
                "names": [x["key"] for x in priors],
            }
        )
        ensemble.save_parameters(block.name, block.realization_index, dataset)
