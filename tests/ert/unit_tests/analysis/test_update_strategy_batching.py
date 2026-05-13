from ert.analysis._update_strategies._batching import calculate_localization_batch_size


def test_that_batch_size_equals_num_params_when_no_observations() -> None:
    assert calculate_localization_batch_size(num_params=7, num_obs=0) == 7
