import os
import stat
from multiprocessing import Process
from textwrap import dedent
from typing import Tuple

import numpy as np
import pytest
import xtgeo

from ert.libres_facade import LibresFacade
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage

from .run_cli import run_cli


def load_from_forward_model(ert_config, ensemble):
    facade = LibresFacade.from_config_file(ert_config)
    realizations = [True] * facade.get_ensemble_size()
    return facade.load_from_forward_model(ensemble, realizations)


@pytest.mark.usefixtures("set_site_config")
def test_surface_param_update(tmpdir):
    """Full update with a surface parameter, it mirrors the poly example,
    except it uses SURFACE instead of GEN_KW.
    """
    ensemble_size = 5
    with tmpdir.as_cwd():
        config = f"""
NUM_REALIZATIONS {ensemble_size}
QUEUE_OPTION LOCAL MAX_RUNNING {ensemble_size}
OBS_CONFIG observations
SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf.irap BASE_SURFACE:surf.irap FORWARD_INIT:True
GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
INSTALL_JOB poly_eval POLY_EVAL
FORWARD_MODEL poly_eval
"""
        base_surface = xtgeo.RegularSurface(
            ncol=2,
            nrow=3,
            xinc=1,
            yinc=1,
            xori=1,
            yori=1,
            yflip=1,
            rotation=1,
        )
        base_surface.to_file("surf.irap", fformat="irap_ascii")

        with open("forward_model", "w", encoding="utf-8") as f:
            f.write(
                """#!/usr/bin/env python
import os

import xtgeo
import numpy as np

if __name__ == "__main__":
    if not os.path.exists("surf.irap"):
        nx = 2
        ny = 3
        values = np.random.standard_normal(nx * ny)
        surf = xtgeo.RegularSurface(ncol=nx,
                                    nrow=ny,
                                    xinc=1,
                                    yinc=1,
                                    rotation=0,
                                    values=values)
        surf.to_file("surf.irap", fformat="irap_ascii")

    surf_fs = xtgeo.surface_from_file("surf.irap", fformat="irap_ascii",
                                    dtype=np.float32)
    a, b, c, *_ = surf_fs.values.data.ravel()

    output = [a * x**2 + b * x + c for x in range(10)]

    with open("gen_data_0.out", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, output)))
        """
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

        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        run_cli(
            ENSEMBLE_SMOOTHER_MODE,
            "--disable-monitor",
            "config.ert",
        )
        with open_storage(tmpdir / "storage") as storage:
            experiment = storage.get_experiment_by_name("es")
            prior = experiment.get_ensemble_by_name("iter-0")
            posterior = experiment.get_ensemble_by_name("iter-1")
            prior_param = (
                prior.load_parameters("MY_PARAM", range(5))["values"]
                .values.reshape(5, 2 * 3)
                .T
            )
            posterior_param = (
                posterior.load_parameters("MY_PARAM", range(5))["values"]
                .values.reshape(5, 2 * 3)
                .T
            )

            assert prior_param.dtype == np.float32
            assert posterior_param.dtype == np.float32
            assert len(prior.load_parameters("MY_PARAM", 0)["values"].x) == 2
            assert len(prior.load_parameters("MY_PARAM", 0)["values"].y) == 3

            assert np.linalg.det(np.cov(prior_param[:3])) > np.linalg.det(
                np.cov(posterior_param[:3])
            )

        rng = np.random.default_rng()
        realizations_to_test = rng.choice(range(ensemble_size), size=2, replace=False)
        surf = xtgeo.surface_from_file(
            f"simulations/realization-{realizations_to_test[0]}/iter-1/surf.irap",
            fformat="irap_ascii",
            dtype=np.float32,
        )

        assert base_surface.ncol == surf.ncol
        assert base_surface.nrow == surf.nrow
        assert base_surface.xinc == surf.xinc
        assert base_surface.yinc == surf.yinc
        assert base_surface.xori == surf.xori
        assert base_surface.yori == surf.yori
        assert base_surface.yflip == surf.yflip
        assert base_surface.rotation == surf.yflip

        surf2 = xtgeo.surface_from_file(
            f"simulations/realization-{realizations_to_test[1]}/iter-1/surf.irap",
            fformat="irap_ascii",
            dtype=np.float32,
        )

        assert not (surf.values == surf2.values).any()


@pytest.mark.limit_memory("130 MB")
@pytest.mark.flaky(reruns=5)
def test_field_param_memory(tmpdir):
    with tmpdir.as_cwd():
        # Setup is done in a subprocess so that memray does not pick up the allocations
        p = Process(target=create_poly_with_field, args=((2000, 1000, 1), 2))
        p.start()
        p.join()  # this blocks until the process terminates

        run_poly()


@pytest.mark.usefixtures("set_site_config")
def create_poly_with_field(field_dim: Tuple[int, int, int], realisations: int):
    """
    This replicates the poly example, only it uses FIELD parameter
    """
    grid_size = field_dim[0] * field_dim[1] * field_dim[2]
    config = dedent(
        f"""
            NUM_REALIZATIONS {realisations}
            OBS_CONFIG observations

            FIELD MY_PARAM PARAMETER my_param.bgrdecl INIT_FILES:my_param.bgrdecl FORWARD_INIT:True
            GRID MY_EGRID.EGRID

            GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
            INSTALL_JOB poly_eval POLY_EVAL
            FORWARD_MODEL poly_eval
            """
    )
    with open("config.ert", "w", encoding="utf-8") as fh:
        fh.writelines(config)

    grid = xtgeo.create_box_grid(dimension=field_dim)
    grid.to_file("MY_EGRID.EGRID", "egrid")
    del grid

    with open("forward_model", "w", encoding="utf-8") as f:
        f.write(
            f"""#!/usr/bin/env python
import numpy as np
import os
import resfo

if __name__ == "__main__":
    if not os.path.exists("my_param.bgrdecl"):
        values = np.random.standard_normal({grid_size})
        resfo.write("my_param.bgrdecl", [("MY_PARAM", values)])
    datas = resfo.read("my_param.bgrdecl")
    assert datas[0][0] == "MY_PARAM"
    a,b,c,*_ = datas[0][1]

    output = [float(a) * x**2 + float(b) * x + float(c) for x in range(10)]
    with open("gen_data_0.out", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, output)))
            """
        )
    os.chmod(
        "forward_model",
        os.stat("forward_model").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
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


def run_poly():
    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitor",
        "config.ert",
    )
