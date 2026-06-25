from ert.config.lint_file import lint_file


def test_that_lint_file_catches_errors(tmp_path, capsys):
    file = tmp_path / "test_config.txt"
    file.write_text("", encoding="utf-8")

    lint_file(str(file))
    captured = capsys.readouterr()

    errors = [m for m in captured.out.splitlines() if "interactive backend" not in m]
    assert len(errors) == 1
    error = errors[0]
    assert "NUM_REALIZATIONS must be set" in error
    assert len(error.split(":")) == 5
