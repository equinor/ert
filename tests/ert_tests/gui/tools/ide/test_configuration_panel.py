# -*- coding: utf-8 -*-
import io

from qtpy.QtWidgets import QMessageBox, QFileDialog

from ert_gui.tools.ide import ConfigurationPanel
import os

UNICODE_TEXT = """ᚠᛇᚻ᛫ᛒᛦᚦ᛫ᚠᚱᚩᚠᚢᚱ᛫ᚠᛁᚱᚪ᛫ᚷᛖᚻᚹᛦᛚᚳᚢᛗ
ᛋᚳᛖᚪᛚ᛫ᚦᛖᚪᚻ᛫ᛗᚪᚾᚾᚪ᛫ᚷᛖᚻᚹᛦᛚᚳ᛫ᛗᛁᚳᛚᚢᚾ᛫ᚻᛦᛏ᛫ᛞᚫᛚᚪᚾ
ᚷᛁᚠ᛫ᚻᛖ᛫ᚹᛁᛚᛖ᛫ᚠᚩᚱ᛫ᛞᚱᛁᚻᛏᚾᛖ᛫ᛞᚩᛗᛖᛋ᛫ᚻᛚᛇᛏᚪᚾ᛬"""


def test_init(qtbot):
    """Initialise a ConfigurationPanel"""
    w = ConfigurationPanel(__file__)
    qtbot.addWidget(w)

    w.close()


def test_load_unicode(tmpdir, qtbot):
    """Attempt to open a config that contains unicode characters."""
    with tmpdir.as_cwd():
        with io.open("poly.ert", "w") as f:
            f.write(UNICODE_TEXT)

        w = ConfigurationPanel("poly.ert")
        qtbot.addWidget(w)

        assert w.ide_panel.getText() == UNICODE_TEXT


def test_save_unicode(tmpdir, qtbot, monkeypatch):
    """Attempt to save config that contains unicode characters. This is an issue
    that exists in Python 2's implementation of `open`, which supports only
    ASCII (Latin-1) encoding.

    """

    @staticmethod
    def information(*args, **kwargs):
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "information", information)

    with tmpdir.as_cwd():
        with io.open("poly.ert", "w") as f:
            f.write("Hello World\n")

        widget = ConfigurationPanel("poly.ert")
        qtbot.addWidget(widget)

        widget.ide_panel.setPlainText(UNICODE_TEXT)
        widget.save()

        with io.open("poly.ert") as f:
            assert f.read() == UNICODE_TEXT


def test_saveas(tmpdir, qtbot, monkeypatch):
    @staticmethod
    def information(*args, **kwargs):
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "information", information)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args: ("poly_copy.ert", "")
    )

    with tmpdir.as_cwd():
        with io.open("poly.ert", "w") as f:
            f.write("Hello World\n")

        widget = ConfigurationPanel("poly.ert")
        widget.show()
        qtbot.addWidget(widget)
        qtbot.waitExposed(widget)
        widget.saveAs()

        with io.open("poly.ert") as orig, io.open("poly_copy.ert") as cpy:
            assert orig.read() == cpy.read()


def test_saveas_cancel(tmpdir, qtbot, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: ("", ""))

    with tmpdir.as_cwd():
        with io.open("poly.ert", "w") as f:
            f.write("Hello World\n")

        widget = ConfigurationPanel("poly.ert")
        widget.show()
        qtbot.addWidget(widget)
        qtbot.waitExposed(widget)
        widget.saveAs()

        assert len(os.listdir(tmpdir)) == 1, "Should not created new file"
