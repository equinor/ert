import os
import stat
from textwrap import dedent

import numpy as np
import pytest
import xtgeo

from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage

from .run_cli import run_cli


@pytest.mark.usefixtures("set_site_config")
def test_surface_param_update(tmpdir):
    """Full update with a surface parameter, it mirrors the poly example,
    except it uses SURFACE instead of GEN_KW.
    """
    ensemble_size = 5
    with tmpdir.as_cwd():
        config = f"""
NUM_REALIZATIONS {ensemble_size}
QUEUE_OPTION LOCAL MAX_RUNNING 2
OBS_CONFIG observations
SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf.irap \
    BASE_SURFACE:surf.irap FORWARD_INIT:True
GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:0
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
            values=6 * [0.0],
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
            "--disable-monitoring",
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
