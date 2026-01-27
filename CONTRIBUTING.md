# Contributing

The following is a set of guidelines for contributing to ERT.

## Ground Rules

1. Automatic code formatting is applied via pre-commit hooks. You
   can see how to set that up [here](https://pre-commit.com/).
2. All code must be testable and unit tested.


## Test Naming

We strive to use test names that are meaningful. Ideally,
the output of `pytest --collect-only tests/` should tell you all you need to know
about why the test is present and what it tests for. We do this for three reasons:


1. So that failure log messages are easy to understand;
2. So that the tests purpose is not lost when the code is updated;
3. To keep a record of intent for changes to the code

Good name examples:
```
<Dir ert>
  <Package tests>
    <Package ert>
      <Package ui_tests>
        <Package cli>
          <Package analysis>
            <Module test_adaptive_localization.py>
              <Function test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior>
              <Function test_that_adaptive_localization_works_with_a_single_observation>
              <Function test_that_adaptive_localization_works_with_multiple_observations>
              <Function test_that_adaptive_localization_with_cutoff_0_equals_ESupdate>
              <Function test_that_posterior_generalized_variance_increases_in_cutoff>
            ...
        <Package config>
          <Package parsing>
              ...
              <Function test_that_missing_arglist_does_not_affect_subsequent_calls>
              <Function test_that_setenv_does_not_expand_envvar>
              <Function test_that_realisation_is_a_alias_of_realization>
              <Function test_that_new_line_can_be_escaped>
              <Function test_that_unknown_queue_option_gives_error_message>
```

Bad name examples:

```
<Dir ert>
  <Package tests>
    <Package ert>
      <Package unit_tests>
        <Package cli>
          <Module test_cli_monitor.py>
            <Function test_color_always>
            <Function test_legends>
            <Function test_result_success>
            <Function test_result_failure>
            <Function test_print_progress>
        <Package config>
          <Package parsing>
            <Module test_config_schema_deprecations.py>
              <Function test_is_angle_bracketed>
          <Module test_summary_config.py>
            <Function test_bad_user_config_file_error_message>
```

Some tips and tricks:

* Avoid smells such as "works", "correctly", "as_expected", "are_handled", etc.

```
            <Function test_forward_model_job[This style of args works without infinite substitution loop.]>
            <Function test_that_arglist_is_parsed_correctly>
            <Function test_that_history_observation_errors_are_calculated_correctly>
            <Function test_that_double_comments_are_handled>
            <Function test_that_quotations_in_forward_model_arglist_are_handled_correctly>
            <Function test_that_the_manage_experiments_tool_can_be_used>
            <Function test_that_parsing_workflows_gives_expected>
            <Function test_that_history_observation_errors_are_calculated_correctly>
```

These fillers fail to add meaning. What does "working correctly" entail? What
is "expected"? How is it "handled"?

* Answers the question: what is correct behavior?

```
            <Function test_that_config_path_substitution_is_the_name_of_the_configs_directory>
            <Function test_when_forward_model_contains_multiple_steps_just_one_checksum_status_is_given>
```

* You can rely on file name (and directory name) to provide context

```
          <Module test_create_forward_model_json.py>
            <Function test_that_values_with_brackets_are_omitted>
            <Function test_that_executables_in_path_are_not_made_realpath>
```

* You can provide context of parameters with `id=`:

```
            <Function test_get_number_of_active_realizations_varies_when_rerun_or_new_iteration[rerun_so_total_realization_count_is_not_affected_by_previous_failed_realizations]>
            <Function test_get_number_of_active_realizations_varies_when_rerun_or_new_iteration[new_iteration_so_total_realization_count_is_only_previously_successful_realizations]>
```

However, sometimes it is best to split those up:

```
            <Function test_number_of_active_realizations_for_reruns_is_unaffected_by_previous_failed_realizations>
            <Function test_number_of_active_realizations_for_new_iteration_is_previously_successful_realizations>
```

## Test categories

Tests that are in the `tests/ert/unit_tests` directory and are not marked with
`slow`, `unreliable`, or `high_utilization` are meant to be exceptionally fast,
reliable and use a limited amount of resources. This is so that one can run
those while iterating on the code. This means special care has to be made when
placing tests here.

### UI tests

These tests are meant to test behavior from a user interaction view to
ensure that the application behaves the way the user expects independently
of code changes. We have two user interfaces, the cli and the gui so those
are subdirectories.

## Performance tests

Tests that runtime and memory performance does not degrade.

## Type hints

mypy is used to check type hints of all code in src/. This is to discover
easily avoidable bugs. A study estimated that about ~15% of bugs can be
discovered by typechecking (https://rebels.cs.uwaterloo.ca/papers/tse2021_khan.pdf).

The following guidelines should be applied when adding type hints:

1. Prefer not to use the `Any` type when possible. The `Any` type can
   be convenient, but anything of type `Any` is equivalent to being
   untyped so essentially not type checked.
1. Except for dunder methods (`__repr__`, `__eq__` etc.),  overridden methods
   should be decorated with the `@override` decorator.
1. Prefer use of `cast` or `assert` (as a type guard) over using the `#type: ignore`
   to ignore type errors. This is to make the assumption of what types are used
   explicit.

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

(*) Tip for making sure all tests pass, try out --exec while rebasing. You
can then have all tests run per commit in a single command.

## Pull Request Scoping

Ideally a pull request will be small in scope, and atomic, addressing precisely
one issue, and mostly result in a single commit. It is however permissible to
fix minor details (formatting, linting, moving, simple refactoring ...) in the
vicinity of your work.

If you find that you want to do lots of changes that are not directly related
to the issue you're working on, create a seperate PR for them so as to avoid
noise in the review process.

## Pull Request Process

1. Work on your own fork of the main repo
1. Squash/organize your work into meaningful atomic commits, if possible.
1. Push your commits and make a draft pull request using the pull request template.
1. Check that your pull request passes all tests.
1. While you wait, carefully review the diff yourself.
1. When all tests have passed and your are happy with your changes, change your
   pull request to "ready for review" and ask for a code review.
1. As a courtesy to the reviewer(s), you may mark commits that react to review
   comments with `fixup` (check out `git commit --fixup`) rather than
   immediately squashing / fixing up and force pushing
1. When the review is concluded
  * rebase onto base branch if necessary,
  * squash whatever still needs squashing, and
  * [fast-forward](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-linear-history) merge.

## Docstrings

Avoid adding trivial documentation but where warranted, docstrings should follow the
[google style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
