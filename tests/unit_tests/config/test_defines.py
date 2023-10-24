import pytest

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain, create_run_path, ensemble_context
from ert.storage import open_storage


def read_jobname(config_file):
    ert_config = ErtConfig.from_file(config_file)
    with open_storage(ert_config.ens_path, mode="w") as storage:
        prior = storage.create_experiment().create_ensemble(
            name="prior", ensemble_size=ert_config.model_config.num_realizations
        )
        run_context = ensemble_context(
            prior,
            [True] * ert_config.model_config.num_realizations,
            0,
            substitution_list=ert_config.substitution_list,
            jobname_format=ert_config.model_config.jobname_format_string,
            runpath_format=ert_config.model_config.runpath_format_string,
            runpath_file="name",
        )
        create_run_path(run_context, ert_config.substitution_list, ert_config)
    return run_context[0].job_name


@pytest.mark.parametrize(
    "defines, expected",
    [
        pytest.param(
            """
NUM_REALIZATIONS 1
DEFINE <A> <B><C>
DEFINE <B> my
DEFINE <C> name
JOBNAME <A>%d""",
            "myname0",
            id="Testing of use before declaration works for defines",
        ),
        pytest.param(
            """
NUM_REALIZATIONS 1
DATA_KW <A> <B><C>
DATA_KW <B> my
DATA_KW <C> name
JOBNAME <A>%d""",
            "myname0",
            id="Testing of use before declaration works for data_kw",
        ),
        pytest.param(
            """
NUM_REALIZATIONS 1
DEFINE <B> my
DEFINE <C> name
DEFINE <A> <B><C>
JOBNAME <A>%d""",
            "myname0",
            id="Testing of declaration before use works for defines",
        ),
        pytest.param(
            """
NUM_REALIZATIONS 1
DATA_KW <B> my
DATA_KW <C> name
DATA_KW <A> <B><C>
JOBNAME <A>%d""",
            "myname0",
            id="Testing of declaration before use works for data_kw",
        ),
    ],
)
def test_that_user_defined_substitution_works_as_expected(
    defines, expected, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config_file.ert"
    config_file.write_text(defines)

    assert read_jobname(str(config_file)) == expected
