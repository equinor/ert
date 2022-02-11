from ert.exceptions import ConfigValidationError, ExperimentError


def report_validation_errors(error: ConfigValidationError) -> None:
    if error.source:
        print(f"Error while loading {error.source} configuration data:")
    print(error.message)


def report_experiment_error(error: ExperimentError) -> None:
    print(error.message)
