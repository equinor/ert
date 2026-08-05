from __future__ import annotations

import pytest

from ert.gui.theme_manager import design_token as design_token_mod
from ert.gui.theme_manager import qss_processing as qss_mod
from ert.gui.theme_manager import theme_utils
from ert.gui.theme_manager.design_token import read_design_token_file
from ert.gui.theme_manager.qss_processing import read_qss_stylesheet_file
from ert.gui.theme_manager.theme_utils import ColorTheme, read_theming_resource


class _MissingResource:
    def is_file(self) -> bool:
        return False

    def __str__(self) -> str:
        return "<missing>"


class _MissingPackage:
    def joinpath(self, _name: str) -> _MissingResource:
        return _MissingResource()


@pytest.mark.parametrize("theme", list(ColorTheme))
def test_that_read_theming_resource_returns_file_content(theme: ColorTheme) -> None:
    content = read_theming_resource(
        filename=f"themes/{theme.value}.json",
        resource_kind=f"{theme.value}-theme design tokens",
    )
    assert isinstance(content, str)
    assert len(content) > 0


def test_that_read_theming_resource_raises_file_not_found_for_missing_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(theme_utils, "files", lambda _pkg: _MissingPackage())

    with pytest.raises(FileNotFoundError, match="my-tokens"):
        read_theming_resource(filename="nope.json", resource_kind="my-tokens")


@pytest.mark.parametrize("theme", list(ColorTheme))
def test_that_read_design_token_file_requests_the_theme_json_resource(
    monkeypatch: pytest.MonkeyPatch, theme: ColorTheme
) -> None:
    monkeypatch.setattr(
        design_token_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: f"content-of-{filename}",
    )
    assert read_design_token_file(theme) == f"content-of-themes/{theme.value}.json"


def test_that_read_design_token_file_raises_file_not_found_for_missing_theme_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*, filename: str, resource_kind: str) -> str:
        raise FileNotFoundError(f"not found: {resource_kind}")

    monkeypatch.setattr(design_token_mod, "read_theming_resource", _raise)

    with pytest.raises(FileNotFoundError, match=r"not found.*design token"):
        read_design_token_file(ColorTheme.DARK)


def test_that_read_qss_stylesheet_file_returns_template_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: f"content-of-{filename}",
    )
    result = read_qss_stylesheet_file("main")
    assert result == "content-of-qss_stylesheet/main.qss.in"


def test_that_read_qss_stylesheet_file_raises_file_not_found_for_missing_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*, filename: str, resource_kind: str) -> str:
        raise FileNotFoundError(f"not found: {resource_kind}")

    monkeypatch.setattr(qss_mod, "read_theming_resource", _raise)

    with pytest.raises(FileNotFoundError, match=r"not found.*QSS template"):
        read_qss_stylesheet_file("missing")
