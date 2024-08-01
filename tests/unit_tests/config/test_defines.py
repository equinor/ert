import pytest

from ert.config import ErtConfig


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
    defines, expected, tmp_path, monkeypatch, run_args, prior_ensemble
):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config_file.ert"
    config_file.write_text(defines)
    assert (
        run_args(ErtConfig.from_file(config_file), prior_ensemble)[0].job_name
        == expected
    )
