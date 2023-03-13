import os
import stat
from argparse import ArgumentParser
from textwrap import dedent
from typing import Optional

import numpy as np
import pytest
import xtgeo
from ecl.util.geometry import Surface
from pandas.testing import assert_frame_equal

from ert.__main__ import ert_parser
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.storage import EnsembleAccessor, open_storage


def create_runpath(
    storage, config, active_mask=None, *, ensemble: Optional[EnsembleAccessor] = None
):
    active_mask = [True] if active_mask is None else active_mask
    ert_config = ErtConfig.from_file(config)
    ert = EnKFMain(ert_config)

    if ensemble is None:
        experiment_id = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment_id, name="default", ensemble_size=ert.getEnsembleSize()
        )

    prior = ert.ensemble_context(
        ensemble,
        active_mask,
        0,
    )

    ert.sample_prior(prior.sim_fs, prior.active_realizations)
    ert.createRunPath(prior)
    return ert, ensemble


def write_grid_property(name, grid, filename, file_format, shape, buffer):
    arr = np.ndarray(shape=shape, buffer=buffer, dtype=buffer.dtype)
    prop = xtgeo.GridProperty(
        ncol=shape[0], nrow=shape[1], nlay=shape[2], values=arr, grid=grid, name=name
    )
    prop.to_file(filename, fformat=file_format)
    return arr


def test_copy_surface(
    tmpdir,
    storage,
):
    with tmpdir.as_cwd():
        config = dedent(
            """
        SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf.irap BASE_SURFACE:surf0.irap
        NUM_REALIZATIONS 10
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        expect_surface = Surface(
            nx=2, ny=2, xinc=1, yinc=1, xstart=1, ystart=1, angle=0
        )
        for i in range(4):
            expect_surface[i] = float(i)
        expect_surface.write("surf.irap")
        expect_surface.write("surf0.irap")

        ert, fs = create_runpath(
            storage, "config.ert", active_mask=[True for _ in range(10)]
        )
        assert fs

        # Assert that the data has been internalised to storage
        new_fs = storage.create_ensemble(
            storage.create_experiment(),
            name="copy",
            ensemble_size=ert.getEnsembleSize(),
        )

        fs.copy_from_case(
            new_fs,
            ["MY_PARAM"],
            [True if x in range(5) else False for x in range(10)],
        )

        arr_old = fs.load_surface_data("MY_PARAM", [0, 2, 3])
        arr_new = new_fs.load_surface_data("MY_PARAM", [0, 2, 3])
        assert np.array_equal(arr_old, arr_new)

        with pytest.raises(KeyError, match="No parameter: MY_PARAM in storage"):
            new_fs.load_surface_data("MY_PARAM", [6, 7, 8])


def test_copy_field(storage, tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_0.grdecl
        NUM_REALIZATIONS 10
        GRID MY_EGRID.EGRID
        """
        )
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(4, 4, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        write_grid_property(
            "MY_PARAM",
            grid,
            "my_param_0.grdecl",
            "grdecl",
            (4, 4, 1),
            np.arange(start=0, stop=4 * 4),
        )

        ert, fs = create_runpath(
            storage, "config.ert", active_mask=[True for _ in range(10)]
        )
        assert fs

        # Assert that the data has been internalised to storage
        new_fs = storage.create_ensemble(
            storage.create_experiment(),
            name="copy",
            ensemble_size=ert.getEnsembleSize(),
        )

        fs.copy_from_case(
            new_fs,
            ["MY_PARAM"],
            [True if x in range(5) else False for x in range(10)],
        )

        arr_old = fs.load_field("MY_PARAM", [0, 2, 3])
        arr_new = new_fs.load_field("MY_PARAM", [0, 2, 3])
        assert np.array_equal(arr_old, arr_new)

        with pytest.raises(KeyError, match="Unable to load FIELD for key: MY_PARAM"):
            new_fs.load_field("MY_PARAM", [6, 7, 8])


def test_copy_gen_kw(storage, tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 10
        GEN_KW KW_NAME template.txt gen_kw.txt prior.txt
        """
        )
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)

        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")

        ert, fs = create_runpath(
            storage, "config.ert", active_mask=[True for _ in range(10)]
        )
        assert fs

        # Assert that the data has been internalised to storage
        new_fs = storage.create_ensemble(
            storage.create_experiment(),
            name="copy",
            ensemble_size=ert.getEnsembleSize(),
        )

        fs.copy_from_case(
            new_fs,
            ["KW_NAME"],
            [True if x in range(5) else False for x in range(10)],
        )

        arr_old = fs.load_gen_kw("KW_NAME", [0, 2, 3])
        arr_new = new_fs.load_gen_kw("KW_NAME", [0, 2, 3])

        assert np.array_equal(arr_old, arr_new)


def test_copy_state_map(storage, tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 10
        """
        )
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)

        active_mask = [True for _ in range(10)]
        active_mask[1] = False
        active_mask[6] = False
        ert, fs = create_runpath(storage, "config.ert", active_mask=active_mask)
        assert fs

        # Assert that the data has been internalised to storage
        new_fs = storage.create_ensemble(
            storage.create_experiment(),
            name="copy",
            ensemble_size=ert.getEnsembleSize(),
        )

        assert fs.state_map == [
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_INITIALIZED,
        ]

        fs.copy_from_case(
            new_fs, [], [True if x in range(5) else False for x in range(10)]
        )

        assert new_fs.state_map == [
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_INITIALIZED,
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_UNDEFINED,
        ]


@pytest.mark.integration_test
def test_copy_gen_data(tmpdir):
    """Full update with a surface parameter, it mirrors the poly example,
    except it uses SURFACE instead of GEN_KW.
    """
    with tmpdir.as_cwd():
        config = dedent(
            """
NUM_REALIZATIONS 10
QUEUE_OPTION LOCAL MAX_RUNNING 5
OBS_CONFIG observations
SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf.irap BASE_SURFACE:surf.irap FORWARD_INIT:True
GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
INSTALL_JOB poly_eval POLY_EVAL
SIMULATION_JOB poly_eval
TIME_MAP time_map
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        expect_surface = Surface(
            nx=1, ny=3, xinc=1, yinc=1, xstart=1, ystart=1, angle=0
        )
        expect_surface.write("surf.irap")

        with open("forward_model", "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    """#!/usr/bin/env python
from ecl.util.geometry import Surface
import numpy as np
import os

if __name__ == "__main__":
    if not os.path.exists("surf.irap"):
        surf = Surface(nx=1, ny=3, xinc=1, yinc=1, xstart=1, ystart=1, angle=0)
        values = np.random.standard_normal(surf.getNX() * surf.getNY())
        for i, value in enumerate(values):
            surf[i] = value
            surf.write(f"surf.irap")
    a, b, c = list(Surface(filename="surf.irap"))
    print(list(Surface(filename="surf.irap")))
    output = [a * x**2 + b * x + c for x in range(10)]
    with open("gen_data_0.out", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, output)))
        """
                )
            )
        os.chmod(
            "forward_model",
            os.stat("forward_model").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )
        with open("POLY_EVAL", "w", encoding="utf-8") as fout:
            fout.write("EXECUTABLE forward_model")
        with open("observations", "w", encoding="utf-8") as fout:
            fout.write(
                dedent(
                    """
            GENERAL_OBSERVATION MY_OBS {
                DATA       = MY_RESPONSE;
                INDEX_LIST = 0,2,4,6,8;
                RESTART    = 0;
                OBS_FILE   = obs.txt;
            };"""
                )
            )

        with open("obs.txt", "w", encoding="utf-8") as fobs:
            fobs.write(
                dedent(
                    """
            2.1457049781272213 0.6
            8.769219841380755 1.4
            12.388014786122742 3.0
            25.600464531354252 5.4
            42.35204755970952 8.6"""
                )
            )

        with open("time_map", "w", encoding="utf-8") as fobs:
            fobs.write("2014-09-10")

        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--current-case",
                "prior",
                "--target-case",
                "smoother_update",
                "config.ert",
                "--port-range",
                "1024-65535",
            ],
        )

        run_cli(parsed)
        with open_storage(tmpdir / "storage", mode="w") as storage:

            new_fs = storage.create_ensemble(
                storage.create_experiment(),
                name="copy",
                ensemble_size=10,
            )

            fs = storage.get_ensemble_by_name("smoother_update")

            fs.copy_from_case(
                new_fs,
                ["MY_RESPONSE@0"],
                [True if x in range(5) else False for x in range(10)],
            )

            ret_old = fs.load_gen_data("MY_RESPONSE@0", [0, 1, 2])
            ret_new = new_fs.load_gen_data("MY_RESPONSE@0", [0, 1, 2])
            assert (ret_new[0] == ret_old[0]).all()
            assert len(ret_old[1]) == 3
            assert len(ret_new[1]) == 3

            ret_old = fs.load_gen_data("MY_RESPONSE@0", [1])
            ret_new = new_fs.load_gen_data("MY_RESPONSE@0", [1, 7])
            assert (ret_new[0] == ret_old[0]).all()
            # should only return one array since 7 is inactive
            assert len(ret_new[1]) == 1


def test_copy_summary(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.ert_config.ens_path, mode="w") as storage:
        default_0 = storage.get_ensemble_by_name("default_0")

        new_ensemble = storage.create_ensemble(
            storage.create_experiment(),
            name="new-ensemble",
            ensemble_size=default_0.ensemble_size,
        )

        keys = ["FOPR", "WGPR:OP2", "BPR:1,3,8"]

        # copy only the three first realizations
        default_0.copy_from_case(new_ensemble, keys, [True, True, True, False, False])

        df_0 = default_0.load_summary_data_as_df(keys, [0, 1, 2])
        df_new = new_ensemble.load_summary_data_as_df(keys, [0, 1, 2])
        assert_frame_equal(df_0, df_new)

        df_0 = default_0.load_summary_data_as_df(keys, [1, 4])
        df_new = new_ensemble.load_summary_data_as_df(keys, [1, 4])
        assert (df_0[1] == df_new[1]).all()
        assert df_new.shape == (600, 1)
