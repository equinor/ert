from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.everest_storage import EverestStorage


def test_save_after_one_batch(copy_math_func_test_data_to_tmp):
    num_batches = 1
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = config.to_dict()
    config_dict["optimization"]["max_batch_num"] = num_batches

    config = EverestConfig(**config_dict)

    n_invocations = 0

    original_write_to_output_dir = None

    def write_to_output_dir_intercept(*args, **kwargs):
        nonlocal n_invocations
        assert original_write_to_output_dir is not None
        result = original_write_to_output_dir(*args, **kwargs)
        n_invocations += 1
        return result

    # We "catch" the everest storage through __setattr__
    # then assert that its .write_to_output_dir is invoked once
    # per batch + one final write for adding merit values
    class MockEverestRunModel(EverestRunModel):
        def __setattr__(self, key, value):
            nonlocal original_write_to_output_dir
            if isinstance(value, EverestStorage):
                ever_storage = value
                original_write_to_output_dir = ever_storage.write_to_output_dir
                ever_storage.write_to_output_dir = write_to_output_dir_intercept

            object.__setattr__(self, key, value)

    run_model = MockEverestRunModel.create(config)
    run_model.run_experiment(EvaluatorServerConfig())

    # Expect one per batch, + one final write after the entire experiment is done
    assert n_invocations == num_batches + 1
