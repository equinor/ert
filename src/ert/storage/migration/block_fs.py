from __future__ import annotations

import logging
import struct
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np
import xarray as xr
import xtgeo
import xtgeo.surface

from ert._c_wrappers.enkf.config.field_config import Field, field_transform
from ert._c_wrappers.enkf.config.gen_kw_config import GenKwConfig
from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig
from ert._c_wrappers.enkf.config.surface_config import SurfaceConfig
from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
from ert._c_wrappers.enkf.enums.realization_state_enum import RealizationStateEnum
from ert.storage import EnsembleAccessor, StorageAccessor
from ert.storage.local_storage import LocalStorageAccessor, local_storage_get_ert_config
from ert.storage.migration._block_fs_native import (  # pylint: disable=E0401
    DataFile,
    Kind,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy.typing as npt
    from xtgeo.surface import RegularSurface


def migrate(path: Path) -> None:
    logger.info(f"Outdated storage detected at '{path}'. Migrating...")
    block_storage_path = _backup(path)

    statuses: List[bool] = []
    with LocalStorageAccessor(path, ignore_migration_check=True) as storage:
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


def _migrate_case_ignoring_exceptions(storage: StorageAccessor, casedir: Path) -> bool:
    try:
        migrate_case(storage, casedir)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            (
                "Exception occurred during migration of BlockFs case "
                f"'{casedir.name}': {exc}"
            ),
            exc_info=exc,
        )
        return False


def migrate_case(storage: StorageAccessor, path: Path) -> None:
    logger.info(f"Migrating case '{path.name}'")
    time_map = _load_timestamps(path / "files/time-map")
    state_map = _load_states(path / "files/state-map")

    parameter_files = [
        DataFile(x) for x in path.glob("Ensemble/mod_*/PARAMETER.data_0")
    ]
    response_files = [DataFile(x) for x in path.glob("Ensemble/mod_*/FORECAST.data_0")]
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

    experiment = storage.create_experiment(parameters=parameter_configs)
    ensemble = experiment.create_ensemble(name=path.name, ensemble_size=ensemble_size)

    _copy_state_map(ensemble, state_map)

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


def _load_states(path: Path) -> List[RealizationStateEnum]:
    if not path.exists():
        return []

    sizeof_int = 4

    with path.open("rb") as f:
        size = struct.unpack("I", f.read(4))[0]
        f.read(sizeof_int)  # int default_value; (unused)
        return [
            RealizationStateEnum(x)
            for x in np.frombuffer(f.read(size * sizeof_int), dtype=np.int32)
        ]


def _copy_state_map(
    ensemble: EnsembleAccessor, states: Sequence[RealizationStateEnum]
) -> None:
    for index, state in enumerate(states):
        ensemble.state_map[index] = state


def _migrate_surface_info(
    data_file: DataFile, ens_config: EnsembleConfig
) -> List[ParameterConfig]:
    seen = set()
    configs: List[ParameterConfig] = []
    for block in data_file.blocks(Kind.SURFACE):
        if block.name in seen:
            continue
        seen.add(block.name)
        config = ens_config[block.name]
        assert isinstance(config, SurfaceConfig)
        configs.append(config)
    return configs


def _migrate_surface(
    ensemble: EnsembleAccessor,
    data_file: DataFile,
    ens_config: EnsembleConfig,
) -> None:
    surfaces: Dict[str, RegularSurface] = {}
    for block in data_file.blocks(Kind.SURFACE):
        config = ens_config[block.name]
        assert isinstance(config, SurfaceConfig)
        try:
            surface = surfaces[block.name]
        except KeyError:
            surface = surfaces[block.name] = xtgeo.surface_from_file(
                config.base_surface_path, fformat="irap_ascii"
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
    for block in data_file.blocks(Kind.FIELD):
        if block.name in seen:
            continue
        seen.add(block.name)
        config = ens_config[block.name]
        assert isinstance(config, Field)

        configs.append(config)
    return configs


def _migrate_field(
    ensemble: EnsembleAccessor,
    data_file: DataFile,
    ens_config: EnsembleConfig,
) -> None:
    for block in data_file.blocks(Kind.FIELD):
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


def _migrate_summary(
    ensemble: EnsembleAccessor,
    data_file: DataFile,
    time_map: npt.NDArray[np.datetime64],
) -> None:
    data: Dict[int, Tuple[List[npt.NDArray[np.float64]], List[str]]] = defaultdict(
        lambda: ([], [])
    )
    for block in data_file.blocks(Kind.SUMMARY):
        if block.name == "TIME":
            continue

        array, keys = data[block.realization_index]
        array.append(data_file.load(block, 0))
        keys.append(block.name)

    for realization_index, (array, keys) in data.items():
        ds = xr.Dataset(
            {"values": (["name", "time"], array)},
            coords={"time": time_map, "name": keys},
        )
        ensemble.save_response("summary", ds, realization_index)


def _migrate_gen_data(
    ensemble: EnsembleAccessor,
    data_file: DataFile,
) -> None:
    realizations = defaultdict(lambda: defaultdict(list))  # type: ignore
    for block in data_file.blocks(Kind.GEN_DATA):
        realizations[block.realization_index][block.name].append(
            {"values": data_file.load(block, 0), "report_step": block.report_step}
        )
    for iens, gen_data in realizations.items():
        for name, values in gen_data.items():
            datasets = []
            for value in values:
                datasets.append(
                    xr.Dataset(
                        {"values": (["report_step", "index"], [value["values"]])},
                        coords={
                            "index": range(len(value["values"])),
                            "report_step": [value["report_step"]],
                        },
                    )
                )
            ensemble.save_response(
                name, xr.combine_by_coords(datasets), iens  # type: ignore
            )


def _migrate_gen_kw_info(
    data_file: DataFile,
    ens_config: EnsembleConfig,
) -> List[ParameterConfig]:
    seen = set()
    configs: List[ParameterConfig] = []
    for block in data_file.blocks(Kind.GEN_KW):
        if block.name in seen:
            continue
        seen.add(block.name)
        config = ens_config[block.name]
        assert isinstance(config, GenKwConfig)

        configs.append(config)
    return configs


def _migrate_gen_kw(
    ensemble: EnsembleAccessor, data_file: DataFile, ens_config: EnsembleConfig
) -> None:
    for block in data_file.blocks(Kind.GEN_KW):
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
