import ert_shared.ensemble_evaluator.entity.ensemble as ee

from unittest.mock import Mock, MagicMock


def test_build_ensemble():
    ensemble = ee.create_ensemble_builder().add_realization(
        ee.create_realization_builder()
        .set_iens(0)
        .add_stage(
            ee.create_stage_builder()
            .add_step(
                ee.create_step_builder()
                .add_job(
                    ee.create_legacy_job_builder()
                    .set_id(0)
                    .set_name("echo_command")
                    .set_ext_job(Mock())
                )
                .set_id(0)
                .set_dummy_io()
            )
            .set_id(0)
            .set_status("unknown")
        )
        .active(True)
    )
    ensemble = ensemble.build()
    real = ensemble.get_reals()[0]
    assert real.is_active()


def test_build_ensemble_legacy():

    run_context = MagicMock()
    run_context.__len__.return_value = 1
    run_context.is_active = lambda i: True if i == 0 else False

    ext_job = MagicMock()
    ext_job.get_executable = MagicMock(return_value="junk.exe")
    ext_job.name = MagicMock(return_value="junk")
    ext_job.get_arglist = MagicMock(return_value=("arg1", "arg2", "arg3"))

    forward_model = MagicMock()
    forward_model.__len__.return_value = 1
    forward_model.iget_job = lambda i: ext_job if i == 0 else None

    analysis_config = MagicMock()
    analysis_config.get_max_runtime = MagicMock(return_value=0)

    queue_config = MagicMock()

    res_config = MagicMock()

    ensemble_builder = ee.create_ensemble_builder_from_legacy(
        run_context=run_context,
        forward_model=forward_model,
        queue_config=queue_config,
        analysis_config=analysis_config,
        res_config=res_config,
    )

    ensemble = ensemble_builder.build()

    real = ensemble.get_reals()[0]
    assert real.is_active()
