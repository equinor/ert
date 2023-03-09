# Contributing

The following is a set of guidelines for contributing to ERT.

## Ground Rules

1. We use Black code formatting
1. All code must be testable and unit tested.

## Commits

We strive to keep a consistent and clean git history and all contributions should adhere to the following:

1. All tests should pass on all commits(*)
1. A commit should do one atomic change on the repository
1. The commit message should be descriptive.



We expect commit messages to follow this style:

1. Separate subject from body with a blank line
1. Limit the subject line to 50 characters
1. Capitalize the subject line
1. Do not end the subject line with a period
1. Use the imperative mood in the subject line
1. Wrap the body at 72 characters
1. Use the body to explain what and why vs. how

This list is taken from [here](https://chris.beams.io/posts/git-commit/).

Also, focus on making clear the reasons why you made the change in the first
place—the way things worked before the change (and what was wrong with that),
the way they work now, and why you decided to solve it the way you did. A
commit body is required for anything except very small changes.

(*) Tip for making sure all tests passes, try out --exec while rebasing. You
can then have all tests run per commit in a single command.

## Pull Request Process

1. Work on your own fork of the main repo
1. Squash/organize your work into meaningful atomic commits, if possible.
1. Push your commits and make a draft pull request using the pull request template.
1. Check that your pull request passes all tests.
1. While you wait, carefully review the diff yourself.
1. When all tests have passed and your are happy with your changes, change your
   pull request to "ready for review" and ask for a code review.
1. When your code has been approved
  * rebase onto the base branch
  * squash whatever still needs squashing, and
  * merge.

### Build documentation

You can build the documentation after installation by running
```bash
pip install -r dev-requirements.txt
sphinx-build -n -v -E -W ./docs ./tmp/ert_docs
```
and then open the generated `./tmp/ert_docs/index.html` in a browser.

To automatically reload on changes you may use

```bash
sphinx-autobuild docs docs/_build/html
```
