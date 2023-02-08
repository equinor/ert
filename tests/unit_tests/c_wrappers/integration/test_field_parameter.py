import math
import os
import stat
from argparse import ArgumentParser
from pathlib import Path
from textwrap import dedent
from typing import Optional
import numpy as np
import numpy.testing
import pytest
import xtgeo

from ert.__main__ import ert_parser
from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.libres_facade import LibresFacade
from ert.storage import EnsembleAccessor, open_storage


def create_runpath(
    storage, config, active_mask=None, *, ensemble: Optional[EnsembleAccessor] = None
):
    active_mask = [True] if active_mask is None else active_mask
    res_config = ResConfig(config)
    ert = EnKFMain(res_config)

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


def load_from_forward_model(ert, ensemble):
    facade = LibresFacade(ert)
    realizations = [True] * facade.get_ensemble_size()
    return facade.load_from_forward_model(ensemble, realizations, 0)


def write_grid_property(name, grid, filename, file_format, shape, buffer):
    arr = np.ndarray(shape=shape, buffer=buffer, dtype=buffer.dtype)
    prop = xtgeo.GridProperty(
        ncol=shape[0], nrow=shape[1], nlay=shape[2], values=arr, grid=grid, name=name
    )
    prop.to_file(filename, fformat=file_format)
    return arr


@pytest.fixture
def storage(tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        yield storage


def test_unknown_file_extension(storage, tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        FIELD PARAM PARAMETER param.wrong INIT_FILES:../../../param.grdecl FORWARD_INIT:True
        GRID MY_EGRID.EGRID
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        write_grid_property(
            "PARAM",
            grid,
            "param.grdecl",
            "grdecl",
            (10, 10, 1),
            np.random.random_sample(100),
        )

        ert, ensemble = create_runpath(storage, "config.ert")
        load_from_forward_model(ert, ensemble)

        with pytest.raises(ValueError, match="Cannot export, invalid fformat: wrong"):
            create_runpath(storage, "config.ert", ensemble=ensemble)


def test_load_two_parameters_forward_init(storage, tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        FIELD PARAM_A PARAMETER param_a.grdecl INIT_FILES:../../../param_a.grdecl FORWARD_INIT:True
        FIELD PARAM_B PARAMETER param_b.grdecl INIT_FILES:../../../param_b.grdecl FORWARD_INIT:True
        GRID MY_EGRID.EGRID
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        param_a = write_grid_property(
            "PARAM_A", grid, "param_a.grdecl", "grdecl", (10, 10, 1), np.full((100), 22)
        )
        param_b = write_grid_property(
            "PARAM_B", grid, "param_b.grdecl", "grdecl", (10, 10, 1), np.full((100), 77)
        )

        ert, fs = create_runpath(storage, "config.ert")

        assert ert.ensembleConfig()["PARAM_A"].getUseForwardInit()
        assert ert.ensembleConfig()["PARAM_B"].getUseForwardInit()
        assert not Path("simulations/realization-0/iter-0/param_a.grdecl").exists()
        assert not Path("simulations/realization-0/iter-0/param_b.grdecl").exists()

        # should not be loaded yet
        with pytest.raises(KeyError, match="Unable to load FIELD for key: PARAM_A"):
            fs.load_field("PARAM_A", [0])

        with pytest.raises(KeyError, match="Unable to load FIELD for key: PARAM_B"):
            fs.load_field("PARAM_B", [0])

        assert load_from_forward_model(ert, fs) == 1

        create_runpath(storage, "config.ert", ensemble=fs)

        prop_a = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_a.grdecl",
            name="PARAM_A",
            grid=grid,
        )
        numpy.testing.assert_equal(prop_a.values.data, param_a)

        prop_b = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_b.grdecl",
            name="PARAM_B",
            grid=grid,
        )
        numpy.testing.assert_equal(prop_b.values.data, param_b)

        # should be loaded now
        loaded_a = fs.load_field("PARAM_A", [0])
        for e in range(0, loaded_a.shape[0]):
            assert loaded_a[e][0] == 22

        loaded_b = fs.load_field("PARAM_B", [0])
        for e in range(0, loaded_b.shape[0]):
            assert loaded_b[e][0] == 77


def test_load_two_parameters_roff(storage, tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        FIELD PARAM_A PARAMETER param_a.roff INIT_FILES:param_a_%d.roff
        FIELD PARAM_B PARAMETER param_b.roff INIT_FILES:param_b_%d.roff
        GRID MY_EGRID.EGRID
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        param_a = write_grid_property(
            "PARAM_A", grid, "param_a_0.roff", "roff", (10, 10, 1), np.full((100), 22)
        )
        param_b = write_grid_property(
            "PARAM_B", grid, "param_b_0.roff", "roff", (10, 10, 1), np.full((100), 77)
        )

        ert, fs = create_runpath(storage, "config.ert")
        assert not ert.ensembleConfig()["PARAM_A"].getUseForwardInit()
        assert not ert.ensembleConfig()["PARAM_B"].getUseForwardInit()

        loaded_a = fs.load_field("PARAM_A", [0])
        for e in range(0, loaded_a.shape[0]):
            assert loaded_a[e][0] == 22

        loaded_b = fs.load_field("PARAM_B", [0])
        for e in range(0, loaded_b.shape[0]):
            assert loaded_b[e][0] == 77

        prop_a = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_a.roff",
            name="PARAM_A",
        )
        numpy.testing.assert_equal(prop_a.values.data, param_a)

        prop_b = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_b.roff",
            name="PARAM_B",
        )
        numpy.testing.assert_equal(prop_b.values.data, param_b)


def test_load_two_parameters(storage, tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        FIELD PARAM_A PARAMETER param_a.grdecl INIT_FILES:param_a_%d.grdecl
        FIELD PARAM_B PARAMETER param_b.grdecl INIT_FILES:param_b_%d.grdecl
        GRID MY_EGRID.EGRID
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        param_a = write_grid_property(
            "PARAM_A",
            grid,
            "param_a_0.grdecl",
            "grdecl",
            (10, 10, 1),
            np.full((100), 22),
        )
        param_b = write_grid_property(
            "PARAM_B",
            grid,
            "param_b_0.grdecl",
            "grdecl",
            (10, 10, 1),
            np.full((100), 77),
        )

        ert, fs = create_runpath(storage, "config.ert")
        assert not ert.ensembleConfig()["PARAM_A"].getUseForwardInit()
        assert not ert.ensembleConfig()["PARAM_B"].getUseForwardInit()

        loaded_a = fs.load_field("PARAM_A", [0])
        for e in range(0, loaded_a.shape[0]):
            assert loaded_a[e][0] == 22

        loaded_b = fs.load_field("PARAM_B", [0])
        for e in range(0, loaded_b.shape[0]):
            assert loaded_b[e][0] == 77

        prop_a = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_a.grdecl",
            name="PARAM_A",
            grid=grid,
        )
        numpy.testing.assert_equal(prop_a.values.data, param_a)

        prop_b = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_b.grdecl",
            name="PARAM_B",
            grid=grid,
        )
        numpy.testing.assert_equal(prop_b.values.data, param_b)


@pytest.mark.parametrize(
    "min_, max_, field_config",
    [
        (
            0.5,
            None,
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_%d.grdecl MIN:0.5",  # pylint: disable=line-too-long  # noqa: E501
        ),
        (
            None,
            0.8,
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_%d.grdecl MAX:0.8",  # pylint: disable=line-too-long  # noqa: E501
        ),
        (
            0.5,
            0.8,
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_%d.grdecl MIN:0.5 MAX:0.8",  # pylint: disable=line-too-long  # noqa: E501
        ),
    ],
)
def test_min_max(storage, tmpdir, min_: int, max_: int, field_config: str):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GRID MY_EGRID.EGRID
        """
        )
        config += field_config
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        buffer = np.random.random_sample(100)
        buffer[56] = 0.001
        buffer[34] = 1.001
        write_grid_property(
            "MY_PARAM", grid, "my_param_0.grdecl", "grdecl", (10, 10, 1), buffer
        )

        create_runpath(storage, "config.ert", [True])

        my_prop = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/my_param.grdecl",
            name="MY_PARAM",
            grid=grid,
        )
        if min_ and max_:
            vfunc = np.vectorize(
                lambda x: ((x + 0.0001) >= min_) and ((x - 0.0001) <= max_)
            )
            assert vfunc(my_prop.values.data).all()
        elif min_:
            vfunc = np.vectorize(lambda x: (x + 0.0001) >= min_)
            assert vfunc(my_prop.values.data).all()
        elif max_:
            vfunc = np.vectorize(lambda x: (x - 0.0001) <= max_)
            assert vfunc(my_prop.values.data).all()


def test_transformation(storage, tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 2
        FIELD PARAM_A PARAMETER param_a.grdecl INIT_FILES:param_a_%d.grdecl INIT_TRANSFORM:LN OUTPUT_TRANSFORM:EXP
        GRID MY_EGRID.EGRID
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        param_a_1 = write_grid_property(
            "PARAM_A",
            grid,
            "param_a_0.grdecl",
            "grdecl",
            (10, 10, 1),
            np.full((100), math.exp(2.5), dtype=float),
        )
        param_a_2 = write_grid_property(
            "PARAM_A",
            grid,
            "param_a_1.grdecl",
            "grdecl",
            (10, 10, 1),
            np.full((100), math.exp(1.5), dtype=float),
        )

        _, fs = create_runpath(storage, "config.ert", [True, True])

        # stored internally as 2.5, 1.5
        loaded_a = fs.load_field("PARAM_A", [0, 1])
        for e in range(0, loaded_a.shape[0]):
            assert loaded_a[e][0] == pytest.approx(2.5)
            assert loaded_a[e][1] == pytest.approx(1.5)

        prop_a_1 = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/param_a.grdecl",
            name="PARAM_A",
            grid=grid,
        )
        numpy.testing.assert_almost_equal(prop_a_1.values.data, param_a_1, decimal=5)

        prop_a_2 = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-1/iter-0/param_a.grdecl",
            name="PARAM_A",
            grid=grid,
        )
        numpy.testing.assert_almost_equal(prop_a_2.values.data, param_a_2, decimal=5)


@pytest.mark.parametrize(
    "config_str, expect_forward_init",
    [
        (
            "FIELD MY_PARAM PARAMETER my_param.grdecl "
            "INIT_FILES:../../../my_param_0.grdecl FORWARD_INIT:True",
            True,
        ),
        (
            "FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param_%d.grdecl",
            False,
        ),
    ],
)
def test_forward_init(storage, tmpdir, config_str, expect_forward_init):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GRID MY_EGRID.EGRID
        """
        )
        config += config_str
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(4, 4, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        expect_param = write_grid_property(
            "MY_PARAM",
            grid,
            "my_param_0.grdecl",
            "grdecl",
            (4, 4, 1),
            np.arange(start=0, stop=4 * 4),
        )

        ert, fs = create_runpath(storage, "config.ert")
        assert (
            ert.ensembleConfig()["MY_PARAM"].getUseForwardInit() is expect_forward_init
        )

        # Assert that the data has been written to runpath
        if expect_forward_init:
            # FORWARD_INIT: True means that ERT waits until the end of the
            # forward model to internalise the data
            assert not Path("simulations/realization-0/iter-0/my_param.grdecl").exists()

            with pytest.raises(
                KeyError, match="Unable to load FIELD for key: MY_PARAM"
            ):
                fs.load_field("MY_PARAM", [0])

            # We try to load the parameters from the forward model, this would fail if
            # forward init was not set correctly
            assert load_from_forward_model(ert, fs) == 1

            # Once data has been internalised, ERT will generate the
            # parameter files
            create_runpath(storage, "config.ert", ensemble=fs)

        prop = xtgeo.gridproperty_from_file(
            pfile="simulations/realization-0/iter-0/my_param.grdecl",
            name="MY_PARAM",
            grid=grid,
        )
        numpy.testing.assert_equal(prop.values.data, expect_param)

        if expect_forward_init:
            arr = fs.load_field("MY_PARAM", [0])
            assert len(arr) == 16


@pytest.mark.integration_test
def test_paramerter_update(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
NUM_REALIZATIONS 5
OBS_CONFIG observations

FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param.grdecl FORWARD_INIT:True
GRID MY_EGRID.EGRID

GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
INSTALL_JOB poly_eval POLY_EVAL
SIMULATION_JOB poly_eval
TIME_MAP time_map
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        grid = xtgeo.create_box_grid(dimension=(10, 10, 1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        write_grid_property(
            "MY_PARAM",
            grid,
            "my_param.grdecl",
            "grdecl",
            (10, 10, 1),
            np.full((100), math.exp(2.5), dtype=float),
        )

        with open("forward_model", "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    """#!/usr/bin/env python
import xtgeo
import numpy as np
import os

if __name__ == "__main__":
    if not os.path.exists("my_param.grdecl"):
        grid= xtgeo.create_box_grid(dimension=(10,10,1))
        grid.to_file("MY_EGRID.EGRID", "egrid")

        my_param = np.ndarray(shape=(10,10,1), buffer=np.random.random_sample(100))
        gp = xtgeo.GridProperty(
            ncol=10,
            nrow=10,
            nlay=1,
            values=my_param,
            grid=grid,
            name="MY_PARAM",
        )
        gp.to_file("my_param.grdecl", fformat="grdecl")

    a= np.random.standard_normal()
    b= np.random.standard_normal()
    c= np.random.standard_normal()
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
        ert = EnKFMain(ResConfig("config.ert"))
        with open_storage(ert.resConfig().ens_path) as storage:
            prior = storage.get_ensemble_by_name("prior")
            posterior = storage.get_ensemble_by_name("smoother_update")

            prior_param = prior.load_field("MY_PARAM", list(range(5)))
            posterior_param = posterior.load_field("MY_PARAM", list(range(5)))

        assert prior_param.shape == (100, 5)
        assert posterior_param.shape == (100, 5)

        pp0 = posterior_param[:, 0]
        pp1 = posterior_param[:, 1]
        pp2 = posterior_param[:, 2]
        pp3 = posterior_param[:, 3]
        pp4 = posterior_param[:, 4]

        assert not np.equal(pp0, pp1).all()
        assert not np.equal(pp1, pp2).all()
        assert not np.equal(pp2, pp3).all()
        assert not np.equal(pp3, pp4).all()
