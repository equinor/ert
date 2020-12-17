# Contributing

The following is a set of guidelines for contributing to ERT.

## Ground Rules

1. We use Black code formatting
1. All code must be testable and unit tested.
1. Commit messages should follow the format as described here https://chris.beams.io/posts/git-commit/

## Pull Request Process

1. Work on your own fork of the main repo
1. Push your commits and make a draft pull request using the pull request template.
1. Check that your pull request passes all tests.
1. When all tests have passed and your are happy with your changes, change your pull request to "ready for review"
   and ask for a code review.
1. When your code has been approved—rebase, squash and merge your changes.

### Build documentation

You can build the documentation after installation by running
```bash
pip install -r dev-requirements.txt
sphinx-build -n -v -E -W ./docs/rst/manual ./tmp/ert_docs
```
and then open the generated `./tmp/ert_docs/index.html` in a browser.
