from typing import Optional

from ert.gui.ertwidgets.models.valuemodel import ValueModel


class PathModel(ValueModel):
    def __init__(
        self,
        default_path: str,
        is_required: bool = True,
        must_be_a_directory: bool = False,
        must_be_a_file: bool = True,
        must_exist: bool = False,
        must_be_absolute: bool = False,
        must_be_executable: bool = False,
    ):
        ValueModel.__init__(self, default_path)

        self._path_is_required = is_required
        self._path_must_be_a_directory = must_be_a_directory
        self._path_must_be_a_file = must_be_a_file
        self._path_must_be_executable = must_be_executable
        self._path_must_exist = must_exist
        self._path_must_be_absolute = must_be_absolute

    def pathIsRequired(self) -> bool:
        return self._path_is_required

    def pathMustBeADirectory(self) -> bool:
        return self._path_must_be_a_directory

    def pathMustBeAFile(self) -> bool:
        return self._path_must_be_a_file

    def pathMustBeExecutable(self) -> bool:
        return self._path_must_be_executable

    def pathMustExist(self) -> bool:
        return self._path_must_exist

    def pathMustBeAbsolute(self) -> bool:
        return self._path_must_be_absolute

    def getPath(self) -> Optional[str]:
        return self.getValue()

    def setPath(self, value: str) -> None:
        self.setValue(value)
