import logging
import os
from contextlib import ExitStack as does_not_raise
from hashlib import sha256
from pathlib import Path
from textwrap import dedent
from typing import Optional, Tuple

import numpy as np
import pytest
from resdata.geometry import Surface

from ert.config import ErtConfig, GenKwConfig
from ert.enkf_main import create_run_path, ensemble_context, sample_prior
from ert.libres_facade import LibresFacade
from ert.storage import EnsembleAccessor, open_storage


def create_runpath(
    storage,
    config,
    active_mask=None,
    *,
    ensemble: Optional[EnsembleAccessor] = None,
    iteration=0,
    random_seed: Optional[int] = 1234,
) -> Tuple[ErtConfig, EnsembleAccessor]:
    active_mask = [True] if active_mask is None else active_mask
    ert_config = ErtConfig.from_file(config)

    if ensemble is None:
        experiment_id = storage.create_experiment(
            ert_config.ensemble_config.parameter_configuration
        )
        ensemble = storage.create_ensemble(
            experiment_id,
            name="default",
            ensemble_size=ert_config.model_config.num_realizations,
        )

    prior = ensemble_context(
        ensemble,
        active_mask,
        iteration,
        None,
        "",
        ert_config.model_config.runpath_format_string,
        "name",
    )

    sample_prior(
        ensemble,
        [i for i, active in enumerate(active_mask) if active],
        random_seed=random_seed,
    )
    create_run_path(prior, ert_config.substitution_list, ert_config)
    return ert_config.ensemble_config, ensemble


def load_from_forward_model(ert_config, ensemble):
    facade = LibresFacade.from_config_file(ert_config)
    realizations = [True] * facade.get_ensemble_size()
    return facade.load_from_forward_model(ensemble, realizations, 0)


@pytest.fixture
def storage(tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        yield storage


@pytest.mark.parametrize(
    "config_str, expect_forward_init, expect_num_loaded, error",
    [
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap   INIT_FILES:surf%d.irap   "
            "BASE_SURFACE:surf0.irap",
            False,
            1,
            "",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:../../../surf%d.irap "
            "BASE_SURFACE:surf0.irap FORWARD_INIT:True",
            True,
            1,
            "",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:../../../surf.irap "
            "BASE_SURFACE:surf0.irap FORWARD_INIT:True",
            True,
            1,
            "",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf%d.irap "
            "BASE_SURFACE:surf0.irap FORWARD_INIT:True",
            True,
            0,
            "Failed to initialize parameter 'MY_PARAM' in file surf0.irap",
        ),
        (
            "SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf.irap "
            "BASE_SURFACE:surf0.irap FORWARD_INIT:True",
            True,
            0,
            "Failed to initialize parameter 'MY_PARAM' in file surf.irap",
        ),
    ],
)
def test_surface_param(
    storage,
    tmpdir,
    config_str,
    expect_forward_init,
    expect_num_loaded,
    error,
    caplog,
):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        """
        )
        config += config_str
        expect_surface = Surface(
            nx=2, ny=2, xinc=1, yinc=1, xstart=1, ystart=1, angle=0
        )
        for i in range(4):
            expect_surface[i] = float(i)
        expect_surface.write("surf.irap")
        expect_surface.write("surf0.irap")

        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        ensemble_config, fs = create_runpath(storage, "config.ert")
        assert ensemble_config["MY_PARAM"].forward_init is expect_forward_init
        # We try to load the parameters from the forward model, this would fail if
        # forward init was not set correctly
        assert load_from_forward_model("config.ert", fs) == expect_num_loaded
        assert error in "".join(caplog.messages)

        # Assert that the data has been written to runpath
        if expect_num_loaded > 0:
            if expect_forward_init:
                # FORWARD_INIT: True means that ERT waits until the end of the
                # forward model to internalise the data
                assert not Path("simulations/realization-0/iter-0/surf.irap").exists()

                # Once data has been internalised, ERT will generate the
                # parameter files
                create_runpath(storage, "config.ert", ensemble=fs, iteration=1)
            expected_iter = 1 if expect_forward_init else 0
            actual_surface = Surface(
                f"simulations/realization-0/iter-{expected_iter}/surf.irap"
            )
            assert actual_surface == expect_surface

        # Assert that the data has been internalised to storage
        if expect_num_loaded > 0:
            arr = fs.load_parameters("MY_PARAM", 0)["values"].values.T
            assert arr.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
        else:
            with pytest.raises(
                KeyError, match="No dataset 'MY_PARAM' in storage for realization 0"
            ):
                fs.load_parameters("MY_PARAM", 0)["values"]


@pytest.mark.parametrize(
    "check_random_seed, expectation",
    [
        pytest.param(
            True,
            does_not_raise(),
            id=(
                "The second initialization we extract the random seed from the first "
                "case and set that in the second case to make sure the sampling can "
                "be reproduced"
            ),
        ),
    ],
)
def test_initialize_random_seed(
    tmpdir, storage, caplog, check_random_seed, expectation
):
    """
    This test initializes a case twice, the first time without a random
    seed.
    """
    with caplog.at_level(logging.INFO), tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
        )
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        create_runpath(storage, "config.ert", random_seed=None)
        # We read the first parameter value as a reference value
        expected = Path("simulations/realization-0/iter-0/kw.txt").read_text("utf-8")

        # Make a clean directory for the second case, which is identical
        # to the first, except that it uses the random seed from the first
        os.makedirs("second")
        os.chdir("second")
        random_seed = next(
            message for message in caplog.messages if message.startswith("RANDOM_SEED")
        ).split()[1]
        if check_random_seed:
            config += f"RANDOM_SEED {random_seed}"
        with open("config_2.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")

        create_runpath(storage, "config_2.ert", random_seed=int(random_seed))
        with expectation:
            assert (
                Path("simulations/realization-0/iter-0/kw.txt").read_text("utf-8")
                == expected
            )


def test_that_first_three_parameters_sampled_snapshot(tmpdir, storage):
    """
    Nothing special about the first three, but there was a regression
    in nr. 2, so added one extra.
    """
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        RANDOM_SEED 1234
        """
        )
        with open("config.ert", mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")
        _, fs = create_runpath(storage, "config.ert", [True] * 3)
        prior = fs.load_parameters("KW_NAME", range(3))["values"].values.ravel()
        expected = np.array([-0.8814228, 1.5847818, 1.009956])
        np.testing.assert_almost_equal(prior, expected)


@pytest.mark.parametrize(
    "num_realisations",
    [4, 5, 10],
)
@pytest.mark.parametrize(
    "template, prior",
    [
        (
            "MY_KEYWORD <MY_KEYWORD>\nMY_SECOND_KEYWORD <MY_SECOND_KEYWORD>",
            ["MY_KEYWORD NORMAL 0 1", "MY_SECOND_KEYWORD NORMAL 0 1"],
        ),
        (
            "MY_KEYWORD <MY_KEYWORD>",
            ["MY_KEYWORD NORMAL 0 1"],
        ),
        (
            "MY_FIRST_KEYWORD <MY_FIRST_KEYWORD>\nMY_KEYWORD <MY_KEYWORD>",
            ["MY_FIRST_KEYWORD NORMAL 0 1", "MY_KEYWORD NORMAL 0 1"],
        ),
    ],
)
def test_that_sampling_is_fixed_from_name(
    tmpdir, storage, template, prior, num_realisations
):
    """
    Testing that the order and number of parameters is not relevant for the values,
    only that name of the parameter and the global seed determine the values.
    """
    with tmpdir.as_cwd():
        conf = GenKwConfig(
            name="KW_NAME",
            forward_init=False,
            template_file="template.txt",
            transfer_function_definitions=prior,
            output_file="kw.txt",
        )
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines(template)
        fs = storage.create_ensemble(
            storage.create_experiment(parameters=[conf]),
            name="prior",
            ensemble_size=num_realisations,
        )
        sample_prior(fs, range(num_realisations), random_seed=1234)

        key_hash = sha256(b"1234" + b"KW_NAME:MY_KEYWORD")
        seed = np.frombuffer(key_hash.digest(), dtype="uint32")
        expected = np.random.default_rng(seed).standard_normal(num_realisations)
        assert fs.load_parameters("KW_NAME").sel(names="MY_KEYWORD")[
            "values"
        ].values.ravel().tolist() == list(expected)


@pytest.mark.parametrize(
    "mask, expected",
    [
        pytest.param(
            [True] * 5,
            [
                -0.8814227775506998,
                1.5847817694032422,
                1.009956004559659,
                -0.3614874716984976,
                0.12143084130052884,
            ],
            id="Sampling all values, checking that we get length of 5",
        ),
        pytest.param(
            [False, True, False, True],
            [
                1.5847817694032422,
                -0.3614874716984976,
            ],
            id=(
                "Sampling a subset of parameters (at index 1 and 4), checking"
                "that those values match the corresponding values from the full"
                "sample at the same index"
            ),
        ),
    ],
)
def test_that_sub_sample_maintains_order(tmpdir, storage, mask, expected):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 5
        RANDOM_SEED 1234
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("template.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with open("prior.txt", "w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD NORMAL 0 1")

        ert_config = ErtConfig.from_file("config.ert")

        fs = storage.create_ensemble(
            storage.create_experiment(
                ert_config.ensemble_config.parameter_configuration
            ),
            name="prior",
            ensemble_size=5,
        )
        sample_prior(
            fs, [i for i, active in enumerate(mask) if active], random_seed=1234
        )

        assert (
            fs.load_parameters("KW_NAME")["values"]
            .sel(names="MY_KEYWORD")
            .values.ravel()
            .tolist()
            == expected
        )
