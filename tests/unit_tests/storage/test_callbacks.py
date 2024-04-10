from pathlib import Path
from unittest.mock import MagicMock

import hypothesis.strategies as st
import pytest
from hypothesis import given

from ert.callbacks import _write_responses_to_storage
from ert.config import SummaryConfig
from ert.load_status import LoadStatus
from ert.run_arg import RunArg
from ert.storage import Ensemble
from ert.substitution_list import SubstitutionList

from ..config.summary_generator import summaries


@pytest.mark.parametrize(
    "iens, geo_id, itr",
    [
        (4, 42, 2),
        (24, 4, 2),
    ],
)
@given(summaries(summary_keys=st.just(["FOPR"])))
@pytest.mark.usefixtures("use_tmpdir")
def test_stuff(iens, geo_id, itr, fopr_summary):
    smspec, unsmry = fopr_summary

    run_path = f"test/batch_{itr}/geo_realization_{geo_id}/simulation_{iens}"

    Path(run_path).mkdir(parents=True, exist_ok=True)
    smspec.to_file(f"{run_path}/CASE_{geo_id}.SMSPEC")
    unsmry.to_file(f"{run_path}/CASE_{geo_id}.UNSMRY")
    smspec.to_file(f"{run_path}/CASE_{iens}.SMSPEC")
    unsmry.to_file(f"{run_path}/CASE_{iens}.UNSMRY")
    summary_config = [
        SummaryConfig("summary", "CASE_<GEO_ID>", ["FOPR"], None),
        SummaryConfig("summary", "CASE_<IENS>", ["FOPR"], None),
    ]
    substitutions = SubstitutionList()
    substitutions[f"<GEO_ID_{iens}_{itr}>"] = str(geo_id)
    run_args = RunArg(
        "test",
        MagicMock(spec=Ensemble),
        iens,
        itr,
        run_path,
        "test",
        substitution_list=substitutions,
    )
    write_result = _write_responses_to_storage(run_args, summary_config)
    assert write_result.status == LoadStatus.LOAD_SUCCESSFUL
