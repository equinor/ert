import pytest

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain
from storage import open_storage


def read_jobname(config_file):
    ert_config = ErtConfig.from_file(config_file)
    ert = EnKFMain(ert_config)
    with open_storage(ert_config.ens_path, mode="w") as storage:
        prior = storage.create_experiment().create_ensemble(
            name="prior", ensemble_size=ert.getEnsembleSize()
        )
        run_context = ert.ensemble_context(prior, [True] * ert.getEnsembleSize(), 0)
        ert.createRunPath(run_context)
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
