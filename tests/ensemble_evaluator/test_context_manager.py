from ert_shared.ensemble_evaluator.context_manager import attach_ensemble_evaluator
from contextlib import ExitStack


def test_context_manager_not_enabled_by_default():
    with attach_ensemble_evaluator(
        run_context=None, run_path_list=None, forward_model=None
    ) as context:
        assert isinstance(context, ExitStack)
