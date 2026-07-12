import tempfile
from unittest.mock import MagicMock


class RaisingWriteNamedTemporaryFile:
    entered = False

    def __init__(self, *args, **kwargs) -> None:
        self.wrapped = tempfile.NamedTemporaryFile(*args, **kwargs)  # noqa
        RaisingWriteNamedTemporaryFile.entered = False

    def __enter__(self, *args, **kwargs) -> MagicMock:
        self.actual_handle = self.wrapped.__enter__(*args, **kwargs)
        mock_handle = MagicMock()
        RaisingWriteNamedTemporaryFile.entered = True

        def ctrlc(_):
            raise RuntimeError

        mock_handle.write = ctrlc
        return mock_handle

    def __exit__(self, *args, **kwargs) -> None:
        self.wrapped.__exit__(*args, **kwargs)
