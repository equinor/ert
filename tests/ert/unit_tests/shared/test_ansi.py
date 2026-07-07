import pytest

from _ert import ansi


@pytest.mark.parametrize("code", ansi.ALL_CODES)
@pytest.mark.parametrize("isatty", [True, False])
def test_that_ansi_print_keeps_or_strips_each_known_code_depending_on_isatty(
    isatty, code, mocker, capsys
):
    mocker.patch("sys.stdout.isatty", return_value=isatty)
    ansi.ansi_print(f"{code}text{code}")
    captured = capsys.readouterr()
    if isatty:
        assert captured.out == f"{code}text{code}\n"
    else:
        assert captured.out == "text\n"
        assert code not in captured.out
