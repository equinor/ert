from __future__ import annotations

import pytest

from ert.gui.theme_manager import qss_processing as qss_mod
from ert.gui.theme_manager.qss_processing import (
    QssProcessingError,
    process_qss,
    read_qss_stylesheet_file,
    substitute_tokens,
)
from ert.gui.theme_manager.theme_utils import ColorTheme


def test_that_substitute_tokens_replaces_single_token() -> None:
    result = substitute_tokens("color: {{bg}};", {"bg": "#fff"})
    assert result == "color: #fff;"


@pytest.mark.parametrize(
    ("template", "tokens", "expected"),
    [
        (
            "{{fg}} on {{bg}}",
            {"fg": "#000", "bg": "#fff"},
            "#000 on #fff",
        ),
        (
            "a:{{x}};b:{{y}};c:{{z}}",
            {"x": "1", "y": "2", "z": "3"},
            "a:1;b:2;c:3",
        ),
    ],
    ids=["two-tokens", "three-tokens"],
)
def test_that_substitute_tokens_replaces_multiple_distinct_tokens(
    template: str, tokens: dict[str, str], expected: str
) -> None:
    assert substitute_tokens(template, tokens) == expected


def test_that_substitute_tokens_replaces_repeated_occurrences_of_same_token() -> None:
    result = substitute_tokens("{{c}} and {{c}} again", {"c": "red"})
    assert result == "red and red again"


def test_that_substitute_tokens_returns_template_unchanged_when_no_placeholders() -> (
    None
):
    template = "QWidget { color: black; }"
    assert substitute_tokens(template, {}) == template


def test_that_substitute_tokens_ignores_extra_tokens_not_in_template() -> None:
    result = substitute_tokens("{{a}}", {"a": "1", "unused": "2"})
    assert result == "1"


def test_that_substitute_tokens_raises_for_undefined_token() -> None:
    with pytest.raises(QssProcessingError, match=r"undefined tokens.*missing"):
        substitute_tokens("{{missing}}", {})


def test_that_substitute_tokens_raises_with_sorted_deduplicated_token_names() -> None:
    template = "{{z-tok}} {{a-tok}} {{z-tok}}"
    with pytest.raises(QssProcessingError, match=r"a-tok,z-tok"):
        substitute_tokens(template, {})


def test_that_substitute_tokens_raises_for_placeholder_with_uppercase_letters() -> None:
    with pytest.raises(QssProcessingError, match="Bg"):
        substitute_tokens("color: {{Bg}};", {"bg": "#fff"})


def test_that_substitute_tokens_raises_for_placeholder_with_underscores() -> None:
    with pytest.raises(QssProcessingError, match="bg_canvas"):
        substitute_tokens("color: {{bg_canvas}};", {"bg-canvas": "#fff"})


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


def test_that_process_qss_returns_fully_resolved_stylesheet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: "bg:{{primary}};fg:{{secondary}}",
    )
    monkeypatch.setattr(
        qss_mod,
        "load_tokens",
        lambda theme: {"primary": "#111", "secondary": "#222"},
    )

    result = process_qss(ColorTheme.DARK)
    assert result == "bg:#111;fg:#222"


def test_that_process_qss_raises_file_not_found_when_template_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*, filename: str, resource_kind: str) -> str:
        raise FileNotFoundError(resource_kind)

    monkeypatch.setattr(qss_mod, "read_theming_resource", _raise)

    with pytest.raises(FileNotFoundError):
        process_qss(ColorTheme.LIGHT)


def test_that_process_qss_raises_processing_error_for_undefined_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: "{{defined}} {{undefined}}",
    )
    monkeypatch.setattr(
        qss_mod,
        "load_tokens",
        lambda theme: {"defined": "ok"},
    )

    with pytest.raises(QssProcessingError, match="undefined"):
        process_qss(ColorTheme.DARK)


def test_that_qss_processing_error_is_an_exception() -> None:
    assert issubclass(QssProcessingError, Exception)
    err = QssProcessingError("boom")
    assert str(err) == "boom"
