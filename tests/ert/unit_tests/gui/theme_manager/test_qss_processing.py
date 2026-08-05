from __future__ import annotations

import pytest

from ert.gui.theme_manager import qss_processing as qss_mod
from ert.gui.theme_manager.qss_processing import (
    QssProcessingError,
    process_qss,
    resolve_includes,
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


def test_that_resolve_includes_replaces_include_directive_with_file_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: "/* sidebar styles */",
    )
    template = 'before\n@include "sidebar.qss.in"\nafter'
    result = resolve_includes(template)
    assert result == "before\n/* sidebar styles */\nafter"


def test_that_resolve_includes_handles_multiple_includes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = {
        "qss_stylesheet/sidebar.qss.in": "sidebar",
        "qss_stylesheet/nav.qss.in": "nav",
    }
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: contents[filename],
    )
    template = '@include "sidebar.qss.in"\n@include "nav.qss.in"'
    result = resolve_includes(template)
    assert result == "sidebar\nnav"


def test_that_resolve_includes_supports_nested_includes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = {
        "qss_stylesheet/outer.qss.in": '@include "inner.qss.in"',
        "qss_stylesheet/inner.qss.in": "inner-content",
    }
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: contents[filename],
    )
    template = '@include "outer.qss.in"'
    result = resolve_includes(template)
    assert result == "inner-content"


def test_that_resolve_includes_detects_circular_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = {
        "qss_stylesheet/a.qss.in": '@include "b.qss.in"',
        "qss_stylesheet/b.qss.in": '@include "a.qss.in"',
    }
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: contents[filename],
    )
    template = '@include "a.qss.in"'
    with pytest.raises(QssProcessingError, match="Circular @include"):
        resolve_includes(template)


def test_that_resolve_includes_returns_template_unchanged_when_no_includes() -> None:
    template = "QWidget { color: black; }"
    assert resolve_includes(template) == template


def test_that_resolve_includes_raises_file_not_found_for_missing_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*, filename: str, resource_kind: str) -> str:
        raise FileNotFoundError(f"not found: {resource_kind}")

    monkeypatch.setattr(qss_mod, "read_theming_resource", _raise)

    template = '@include "missing.qss.in"'
    with pytest.raises(FileNotFoundError):
        resolve_includes(template)


def test_that_resolve_includes_resolves_indented_include_directive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: "nav",
    )
    template = 'before\n    @include "nav.qss.in"\nafter'
    assert resolve_includes(template) == "before\nnav\nafter"


def test_that_resolve_includes_resolves_include_with_trailing_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: "nav",
    )
    template = '@include "nav.qss.in"   \nafter'
    assert resolve_includes(template) == "nav\nafter"


@pytest.mark.parametrize(
    "template",
    [
        '@include "nav.css"',
        '@include "nav"',
    ],
    ids=["wrong-suffix", "no-suffix"],
)
def test_that_resolve_includes_raises_for_include_of_non_qss_in_file(
    template: str,
) -> None:
    with pytest.raises(QssProcessingError, match="Malformed @include directive"):
        resolve_includes(template)


def test_that_resolve_includes_raises_for_include_without_quoted_filename() -> None:
    with pytest.raises(QssProcessingError, match="Malformed @include directive"):
        resolve_includes("@include nav.qss.in")


def test_that_resolve_includes_raises_for_trailing_content_after_include() -> None:
    with pytest.raises(QssProcessingError, match="Malformed @include directive"):
        resolve_includes('@include "nav.qss.in" extra')


def test_that_resolve_includes_error_message_names_the_offending_directive() -> None:
    with pytest.raises(QssProcessingError, match=r'@include "nav\.css"'):
        resolve_includes('    @include "nav.css"   ')


def test_that_resolve_includes_ignores_include_word_inside_a_rule_body() -> None:
    template = 'QWidget { qproperty-name: "@include nav"; }'
    assert resolve_includes(template) == template


def test_that_resolve_includes_raises_for_malformed_include_in_included_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = {
        "qss_stylesheet/outer.qss.in": '@include "inner.css"',
    }
    monkeypatch.setattr(
        qss_mod,
        "read_theming_resource",
        lambda *, filename, resource_kind: contents[filename],
    )

    with pytest.raises(QssProcessingError, match=r'@include "inner\.css"'):
        resolve_includes('@include "outer.qss.in"')
