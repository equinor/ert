from pathlib import Path

import pytest

from ert.run_context import RunContext
from ert.runpaths import Runpaths


@pytest.mark.usefixtures("use_tmpdir")
def test_create(storage):
    mask = [True] * 100
    mask[50] = False
    itr = 0
    realizations = list(range(len(mask)))
    run_context1 = RunContext(
        storage.create_experiment().create_ensemble(
            name="test", ensemble_size=len(realizations)
        ),
        Runpaths(
            "path/to/sim%d",
            "job%d",
            Path("runpath_file_name"),
        ),
        mask,
        iteration=itr,
    )
    run_id1 = run_context1.run_id

    run_arg0 = run_context1[0]
    with pytest.raises(ValueError):
        run_arg0.getQueueIndex()

    assert run_arg0.iter_id == itr
    assert run_arg0.get_run_id() == str(run_id1)

    run_context2 = RunContext(
        storage.create_experiment().create_ensemble(
            name="test", ensemble_size=len(realizations)
        ),
        Runpaths(
            "path/to/sim%d",
            "job%d",
            Path("runpath_file_name"),
        ),
        mask,
        iteration=itr,
    )
    run_id2 = run_context2.run_id

    assert run_id1 != run_id2

    assert run_context1.is_active(49)
    assert not run_context1.is_active(50)
