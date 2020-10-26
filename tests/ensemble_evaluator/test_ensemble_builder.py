import ert_shared.ensemble_evaluator.entity.ensemble as ee


def test_build_ensemble():
    ensemble = ee.create_ensemble_builder().add_realization(
        ee.create_realization_builder()
        .set_iens(0)
        .add_stage(
            ee.create_stage_builder()
            .add_step(
                ee.create_step_builder()
                .add_job(
                    ee.create_script_job_builder()
                    .set_executable("cmd.exe")
                    .set_args(("echo", "bar"))
                    .set_id(0)
                    .set_name("echo_command")
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


# def test_build_ensemble_legacy
