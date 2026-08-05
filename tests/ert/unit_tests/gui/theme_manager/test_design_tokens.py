from __future__ import annotations

import pytest

from ert.gui.theme_manager import theme_utils
from ert.gui.theme_manager.design_token import (
    _HEX_COLOR,
    load_tokens,
    parse_design_tokens,
    validate_design_tokens,
)
from ert.gui.theme_manager.theme_utils import ColorTheme, read_theming_resource


class _MissingResource:
    def is_file(self) -> bool:
        return False

    def __str__(self) -> str:
        return "<missing>"


class _MissingPackage:
    def joinpath(self, _name: str) -> _MissingResource:
        return _MissingResource()


class _FakeResource:
    """Return controlled text content so we can inject bad JSON / bad tokens."""

    def __init__(self, text: str) -> None:
        self._text = text

    def is_file(self) -> bool:
        return True

    def read_text(self, encoding: str = "utf-8") -> str:
        return self._text


class _FakePackage:
    def __init__(self, text: str) -> None:
        self._text = text

    def joinpath(self, _name: str) -> _FakeResource:
        return _FakeResource(self._text)


@pytest.mark.parametrize(
    "value",
    [
        "#abc",
        "#ABC",
        "#AaBbCc",
        "#aabbcc",
        "#AABBCC",
        "#aabbccdd",
        "#AABBCCDD",
        "#000",
        "#000000",
        "#00000000",
    ],
    ids=lambda v: f"valid:{v}",
)
def test_that_hex_regex_matches_valid_colors(value: str) -> None:
    assert _HEX_COLOR.fullmatch(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "aabbcc",
        "#abcd",
        "#abcde",
        "#abcdeff",
        "#gggggg",
        "#12345",
        "#1234567",
        "",
        "rgb(0,0,0)",
        "#",
    ],
    ids=lambda v: f"invalid:{v!r}",
)
def test_that_hex_regex_rejects_invalid_colors(value: str) -> None:
    assert _HEX_COLOR.fullmatch(value) is None


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


@pytest.mark.parametrize(
    ("json_text", "theme", "expected"),
    [
        ('{"bg": "#fff"}', ColorTheme.DARK, {"bg": "#fff"}),
        ("{}", ColorTheme.LIGHT, {}),
    ],
)
def test_that_parse_design_tokens_returns_expected_dict(
    json_text: str, theme: ColorTheme, expected: dict
) -> None:
    assert parse_design_tokens(json_text, theme) == expected


@pytest.mark.parametrize(
    ("json_text", "theme", "match"),
    [
        ("{bad", ColorTheme.DARK, "not valid JSON"),
        ("[1, 2]", ColorTheme.LIGHT, "must be a JSON object"),
    ],
)
def test_that_parse_design_tokens_raises_on_invalid_input(
    json_text: str, theme: ColorTheme, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_design_tokens(json_text, theme)


@pytest.mark.parametrize("theme", list(ColorTheme))
def test_that_parse_design_tokens_error_includes_theme_name(
    theme: ColorTheme,
) -> None:
    with pytest.raises(ValueError, match=f"{theme.value}-theme"):
        parse_design_tokens("{bad", theme)


@pytest.mark.parametrize(
    ("tokens", "theme"),
    [
        ({"a": "#abc", "b": "#aabbcc", "c": "#aabbccdd"}, ColorTheme.DARK),
        ({"x": "#fff"}, ColorTheme.LIGHT),
        ({}, ColorTheme.DARK),
    ],
)
def test_that_validate_design_tokens_accepts_valid_input_and_returns_same_object(
    tokens: dict, theme: ColorTheme
) -> None:
    result = validate_design_tokens(tokens, theme)
    assert result == tokens
    assert result is tokens


@pytest.mark.parametrize(
    ("tokens", "theme", "match"),
    [
        ({"bad-token": "not-a-color"}, ColorTheme.DARK, "bad-token"),
        ({"num": 123}, ColorTheme.LIGHT, "num"),
        (
            {"z-token": "bad", "a-token": "bad", "m-token": "#fff"},
            ColorTheme.DARK,
            "a-token, z-token",
        ),
    ],
)
def test_that_validate_design_tokens_raises_for_invalid_values(
    tokens: dict, theme: ColorTheme, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_design_tokens(tokens, theme)


@pytest.mark.parametrize("theme", list(ColorTheme))
def test_that_load_tokens_returns_non_empty_dict_for_each_theme(
    theme: ColorTheme,
) -> None:
    tokens = load_tokens(theme)
    assert isinstance(tokens, dict)
    assert len(tokens) > 0
    for name, value in tokens.items():
        assert _HEX_COLOR.fullmatch(value), f"Token {name!r} has invalid hex: {value!r}"


def test_that_load_tokens_raises_file_not_found_when_theme_json_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(theme_utils, "files", lambda _pkg: _MissingPackage())

    with pytest.raises(FileNotFoundError):
        load_tokens(ColorTheme.DARK)


def test_that_load_tokens_raises_value_error_for_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(theme_utils, "files", lambda _pkg: _FakePackage("{not json"))

    with pytest.raises(ValueError, match="not valid JSON"):
        load_tokens(ColorTheme.LIGHT)


def test_that_load_tokens_raises_value_error_for_invalid_hex_colors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        theme_utils,
        "files",
        lambda _pkg: _FakePackage('{"bad-tok": "not-hex"}'),
    )

    with pytest.raises(ValueError, match="bad-tok"):
        load_tokens(ColorTheme.DARK)
