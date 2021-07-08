# -*- coding: utf-8 -*-
import io

from qtpy.QtWidgets import QAction, QMessageBox

from ert_gui.tools.ide import ConfigurationPanel

UNICODE_TEXT = u"""ᚠᛇᚻ᛫ᛒᛦᚦ᛫ᚠᚱᚩᚠᚢᚱ᛫ᚠᛁᚱᚪ᛫ᚷᛖᚻᚹᛦᛚᚳᚢᛗ
ᛋᚳᛖᚪᛚ᛫ᚦᛖᚪᚻ᛫ᛗᚪᚾᚾᚪ᛫ᚷᛖᚻᚹᛦᛚᚳ᛫ᛗᛁᚳᛚᚢᚾ᛫ᚻᛦᛏ᛫ᛞᚫᛚᚪᚾ
ᚷᛁᚠ᛫ᚻᛖ᛫ᚹᛁᛚᛖ᛫ᚠᚩᚱ᛫ᛞᚱᛁᚻᛏᚾᛖ᛫ᛞᚩᛗᛖᛋ᛫ᚻᛚᛇᛏᚪᚾ᛬"""


class MockHelpTool(object):
    def getAction(self):
        return QAction(None)


def test_init(qtbot):
    """Initialise a ConfigurationPanel"""
    w = ConfigurationPanel(__file__, MockHelpTool())
    qtbot.addWidget(w)

    w.close()


def test_load_unicode(tmpdir, qtbot):
    """Attempt to open a config that contains unicode characters."""
    with tmpdir.as_cwd():
        with io.open("poly.ert", "w") as f:
            f.write(UNICODE_TEXT)

        w = ConfigurationPanel("poly.ert", MockHelpTool())
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
            f.write(u"Hello World\n")

        widget = ConfigurationPanel("poly.ert", MockHelpTool())
        qtbot.addWidget(widget)

        widget.ide_panel.setPlainText(UNICODE_TEXT)
        widget.save()

        with io.open("poly.ert") as f:
            assert f.read() == UNICODE_TEXT
