from qtpy.QtWidgets import QFileDialog, QMessageBox


def mock_dialogs(
    mocker,
    existing_dir=None,
    open_file_name=None,
    open_file_names=None,
    save_file_name=None,
    information_button=None,
    warning_button=None,
    critical_button=None,
    question_button=None,
):
    """Mock some methods that shows modal dialogs in Qt.

    Parameter mocker is expected to be the result of the mocker fixture
    from the pytest-mock package.
    When testing the Qt GUI, modal dialogs block also the execution of
    the test as well. This method mocks some of the methods that open
    modal dialogs, and set the given arguments as return values. If the
    given argument for a certain method is None, the method is not
    mocked.
    Other possible workarounds for this problem are discussed at
      https://github.com/pytest-dev/pytest-qt/issues/18
    """
    if existing_dir is not None:
        mocker.patch.object(
            QFileDialog, "getExistingDirectory", return_value=(existing_dir, None)
        )
    if open_file_name is not None:
        mocker.patch.object(
            QFileDialog, "getOpenFileName", return_value=(open_file_name, None)
        )
    if open_file_names is not None:
        mocker.patch.object(
            QFileDialog, "getOpenFileNames", return_value=(open_file_names, None)
        )
    if save_file_name is not None:
        mocker.patch.object(
            QFileDialog, "getSaveFileName", return_value=(save_file_name, None)
        )

    if information_button is not None:
        mocker.patch.object(QMessageBox, "information", return_value=information_button)
    if warning_button is not None:
        mocker.patch.object(QMessageBox, "warning", return_value=warning_button)
    if critical_button is not None:
        mocker.patch.object(QMessageBox, "critical", return_value=critical_button)
    if question_button is not None:
        mocker.patch.object(QMessageBox, "question", return_value=question_button)


def mock_dialogs_all(
    mocker,
    existing_dir="",
    open_file_name="",
    open_file_names="",
    save_file_name="",
    information_button=QMessageBox.Ok,
    warning_button=QMessageBox.Ok,
    critical_button=QMessageBox.Ok,
    question_button=QMessageBox.Ok | QMessageBox.Yes,
):
    """Same as mock_dialog but with some acceptable default"""
    mock_dialogs(
        mocker,
        existing_dir,
        open_file_name,
        open_file_names,
        save_file_name,
        information_button,
        warning_button,
        critical_button,
        question_button,
    )
