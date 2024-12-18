from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QToolButton, QWidget

from .validationsupport import ValidationSupport

if TYPE_CHECKING:
    from .models.path_model import PathModel


class PathChooser(QWidget):
    """
    PathChooser: shows, enables choosing of and validates paths.  The data
    structure expected and sent to the models getValue and setValue is a string.
    """

    PATH_DOES_NOT_EXIST_MSG = "The specified path does not exist."
    FILE_IS_NOT_EXECUTABLE_MSG = "The specified file is not an executable."
    PATH_IS_NOT_A_FILE_MSG = "The specified path must be a file."
    PATH_IS_NOT_ABSOLUTE_MSG = "The specified path must be an absolute path."
    PATH_IS_NOT_A_DIRECTORY_MSG = "The specified path must be a directory."
    REQUIRED_FIELD_MSG = "A path is required."

    def __init__(self, model: PathModel) -> None:
        QWidget.__init__(self)
        self._validation_support = ValidationSupport(self)

        self._editing = True

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self._path_line = QLineEdit()
        self._path_line.setMinimumWidth(250)

        layout.addWidget(self._path_line)

        dialog_button = QToolButton(self)
        dialog_button.setIcon(QIcon("img:folder_open.svg"))
        dialog_button.setIconSize(QSize(16, 16))
        dialog_button.clicked.connect(self.selectPath)
        layout.addWidget(dialog_button)

        self.valid_color = self._path_line.palette().color(
            self._path_line.backgroundRole()
        )

        self._path_line.setText(os.getcwd())
        self._editing = False

        self._model = model
        self._model.valueChanged.connect(self.getPathFromModel)

        self._path_line.editingFinished.connect(self.validatePath)
        self._path_line.editingFinished.connect(self.contentsChanged)
        self._path_line.textChanged.connect(self.validatePath)

        self.setLayout(layout)
        self.getPathFromModel()

    def isPathValid(self, path: str) -> tuple[bool, str]:
        path = path.strip()
        path_exists = os.path.exists(path)
        is_file = os.path.isfile(path)
        is_directory = os.path.isdir(path)
        is_executable = os.access(path, os.X_OK)
        is_absolute = os.path.isabs(path)

        valid = True
        message = ""

        if not path:
            if self._model.pathIsRequired():
                valid = False
                message = PathChooser.REQUIRED_FIELD_MSG
        elif not path_exists:
            if self._model.pathMustExist():
                valid = False
                message = PathChooser.PATH_DOES_NOT_EXIST_MSG
            # todo: check if new (non-existing) file has directory or file format?
        elif path_exists:
            if self._model.pathMustBeExecutable() and is_file and not is_executable:
                valid = False
                message = PathChooser.FILE_IS_NOT_EXECUTABLE_MSG
            elif self._model.pathMustBeADirectory() and not is_directory:
                valid = False
                message = PathChooser.PATH_IS_NOT_A_DIRECTORY_MSG
            elif self._model.pathMustBeAbsolute() and not is_absolute:
                valid = False
                message = PathChooser.PATH_IS_NOT_ABSOLUTE_MSG
            elif self._model.pathMustBeAFile() and not is_file:
                valid = False
                message = PathChooser.PATH_IS_NOT_A_FILE_MSG

        return valid, message

    def validatePath(self) -> None:
        """Called whenever the path is modified"""
        palette = self._path_line.palette()

        valid, message = self.isPathValid(self.getPath())

        validity_type = ValidationSupport.WARNING

        color = ValidationSupport.ERROR_COLOR if not valid else self.valid_color

        self._validation_support.setValidationMessage(message, validity_type)
        self._path_line.setToolTip(message)
        palette.setColor(self._path_line.backgroundRole(), color)

        self._path_line.setPalette(palette)

    def getPath(self) -> str:
        """Returns the path"""
        return os.path.expanduser(str(self._path_line.text()).strip())

    def selectPath(self) -> None:
        """Pops up the 'select a file/directory' dialog"""
        # todo: This probably needs some reworking to work properly with
        # different scenarios... (file + dir)
        self._editing = True
        current_directory = self.getPath()

        if self._model.pathMustBeAFile():
            current_directory = QFileDialog.getOpenFileName(
                self, "Select a file path", current_directory
            )[0]
        else:
            current_directory = QFileDialog.getExistingDirectory(
                self, "Select a directory", current_directory
            )

        if current_directory:
            if not self._model.pathMustBeAbsolute():
                cwd = os.getcwd()
                match = re.match(cwd + "/(.*)", current_directory)
                if match:
                    current_directory = match.group(1)

            self._path_line.setText(current_directory)
            self._model.setPath(self.getPath())

        self._editing = False

    def contentsChanged(self) -> None:
        """Called whenever the path is changed."""
        path_is_valid, _ = self.isPathValid(self.getPath())

        if not self._editing and path_is_valid:
            self._model.setPath(self.getPath())

    def getPathFromModel(self) -> None:
        """Retrieves data from the model and inserts it into the edit line"""
        self._editing = True

        path = self._model.getPath()
        if path is None:
            path = ""

        self._path_line.setText(str(path))
        self._editing = False

    def getValidationSupport(self) -> ValidationSupport:
        return self._validation_support

    def isValid(self) -> bool:
        return self._validation_support.isValid()
