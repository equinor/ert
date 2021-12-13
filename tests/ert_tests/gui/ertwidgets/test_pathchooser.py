from qtpy.QtWidgets import QFileDialog
from ert_gui.ertwidgets.pathchooser import PathChooser
from ert_gui.ertwidgets.models.path_model import PathModel


def test_selectfile(qtbot, tmpdir, monkeypatch):
    model = PathModel(tmpdir, must_be_a_file=True)
    widget = PathChooser(model)
    qtbot.addWidget(widget)

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("foo", "bar"))
    with qtbot.waitExposed(widget):
        widget.show()
    widget.selectPath()
    assert "foo" == model.getPath(), f"Unexpected path {model.getPath()}"


def test_selectfile_cancel(qtbot, tmpdir, monkeypatch):
    model = PathModel(tmpdir, must_be_a_file=True)
    widget = PathChooser(model)
    qtbot.addWidget(widget)

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    with qtbot.waitExposed(widget):
        widget.show()
    widget.selectPath()
    assert tmpdir == model.getPath(), f"Unexpected path {model.getPath()}"


def test_selectdirectory(qtbot, tmpdir, monkeypatch):
    model = PathModel(tmpdir, must_be_a_file=False)
    widget = PathChooser(model)
    qtbot.addWidget(widget)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: "foo")

    with qtbot.waitExposed(widget):
        widget.show()
    widget.selectPath()
    assert "foo" == model.getPath(), f"Unexpected path {model.getPath()}"


def test_selectdirectory_cancel(qtbot, tmpdir, monkeypatch):
    model = PathModel(tmpdir, must_be_a_file=False)
    widget = PathChooser(model)
    qtbot.addWidget(widget)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: "")

    with qtbot.waitExposed(widget):
        widget.show()
    widget.selectPath()
    assert tmpdir == model.getPath(), f"Unexpected path {model.getPath()}"
