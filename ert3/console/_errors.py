from ert.exceptions import ConfigValidationError


def report_validation_errors(error: ConfigValidationError) -> None:
    if error.source:
        print(f"Error while loading {error.source} configuration data:")
    print(error.message)
