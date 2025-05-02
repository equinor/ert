import os

from .config_errors import ConfigValidationError, ErrorInfo
from .file_context_token import FileContextToken


def read_file(file: str, token: FileContextToken | None = None) -> str:
    file = os.path.normpath(os.path.abspath(file))
    try:
        with open(file, encoding="utf-8") as f:
            return f.read()
    except OSError as err:
        raise ConfigValidationError.with_context(str(err), token or file) from err
    except UnicodeDecodeError as e:
        error_words = str(e).split(" ")
        hex_str = error_words[error_words.index("byte") + 1]
        try:
            unknown_char = chr(int(hex_str, 16))
        except ValueError:
            unknown_char = f"hex:{hex_str}"

        # Find the first line in the file with decode error
        bad_byte_lines: list[int] = []
        with open(file, "rb") as f:
            all_lines = []
            for line in f:
                all_lines.append(line)

        for i, line in enumerate(all_lines):
            try:
                line.decode("utf-8")
            except UnicodeDecodeError:
                # The error occurs on this line, so make this entire line red
                # (Figuring column if it is not 0 is tricky and prob not necessary)
                # Use 1-indexed lines like lark and for errors in ert
                bad_byte_lines.append(i + 1)

        assert len(bad_byte_lines) != -1

        raise ConfigValidationError(
            [
                ErrorInfo(
                    message=(
                        f"Unsupported non UTF-8 character {unknown_char!r} "
                        f"found in file: {file!r}"
                    ),
                    filename=str(file),
                    column=0,
                    line=bad_line,
                    end_column=-1,
                    end_line=bad_line,
                )
                for bad_line in bad_byte_lines
            ]
        ) from e
