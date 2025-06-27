from textwrap import dedent

import pytest
from pandas.core.frame import DataFrame

from ert.libres_facade import LibresFacade
from ert.storage import open_storage


def test_keyword_type_checks(snake_oil_default_storage):
    assert (
        "BPR:1,3,8"
        in snake_oil_default_storage.experiment.response_type_to_response_keys[
            "summary"
        ]
    )


def test_keyword_type_checks_missing_key(snake_oil_default_storage):
    assert (
        "nokey"
        not in snake_oil_default_storage.experiment.response_type_to_response_keys[
            "summary"
        ]
    )


@pytest.mark.filterwarnings("ignore:.*Use load_responses.*:DeprecationWarning")
def test_data_fetching_missing_key(snake_oil_case):
    with open_storage(snake_oil_case.ens_path, mode="w") as storage:
        experiment = storage.create_experiment()
        empty_case = experiment.create_ensemble(name="new_case", ensemble_size=25)

        data = [
            empty_case.load_all_gen_kw_data("nokey", None),
        ]

        for dataframe in data:
            assert isinstance(dataframe, DataFrame)
            assert dataframe.empty


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_kw_log_appended_extra():
    with open("config_file.ert", "w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 1
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
            )
        )
    with open("template.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD LOGNORMAL 1 2")


def test_misfit_collector(snake_oil_case_storage, snake_oil_default_storage, snapshot):
    facade = LibresFacade(snake_oil_case_storage)
    data = facade.load_all_misfit_data(snake_oil_default_storage)
    snapshot.assert_match(data.round(8).to_csv(), "misfit_collector.csv")

    with pytest.raises(KeyError):
        # realization 60:
        _ = data.loc[60]
