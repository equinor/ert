from .ert_config import ErtConfig
from .parsing import ConfigValidationError, ErrorInfo


def lint_file(file: str) -> None:
    def formatter(info: ErrorInfo) -> str:
        return ":".join(
            [
                str(key)
                for key in [
                    info.filename,
                    info.line,
                    info.column,
                    info.end_column,
                    info.message,
                ]
            ]
        )

    try:
        ErtConfig.from_file(file)
        print("Found no errors")

    except ConfigValidationError as err:
        print("\n".join(m.replace("\n", " ") for m in err.messages(formatter)))
