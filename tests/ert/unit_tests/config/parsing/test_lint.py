from ert.config.lint_file import lint_file


def test_that_empty_file_yields_linting_error_on_num_realizations(tmp_path, capsys):
    file = tmp_path / "test_config.txt"
    file.write_text("", encoding="utf-8")

    lint_file(str(file))
    captured = capsys.readouterr()

    errors = captured.out.splitlines()
    assert len(errors) == 1

    error = errors[0]
    assert "NUM_REALIZATIONS must be set" in error
    assert len(error.split(":")) == 5


def test_that_lint_file_prints_when_no_errors(tmp_path, capsys):
    file = tmp_path / "test_config.txt"
    file.write_text("NUM_REALIZATIONS 10", encoding="utf-8")

    lint_file(str(file))
    captured = capsys.readouterr()

    assert captured.out.splitlines().count("Found no errors") == 1
