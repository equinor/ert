The following instructions are only to be applied when performing a code review.

# Testing

- [] Make sure all is unit tested.
- [] test names should follow a `test_that` executable specification style. Ideally,
the output of `pytest --collect-only tests/` should tell you all you need to know
about why the test is present and what it tests for.

Good name examples:

 * test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior
 * test_that_adaptive_localization_works_with_a_single_observation
 * test_that_adaptive_localization_works_with_multiple_observations
 * test_that_adaptive_localization_with_cutoff_0_equals_ESupdate
 * test_that_posterior_generalized_variance_increases_in_cutoff
 * test_that_missing_arglist_does_not_affect_subsequent_calls
 * test_that_setenv_does_not_expand_envvar
 * test_that_realisation_is_a_alias_of_realization
 * test_that_new_line_can_be_escaped
 * test_that_unknown_queue_option_gives_error_message

Bad name examples:
  * test_color_always
  * test_legends
  * test_result_success
  * test_result_failure
  * test_print_progress
  * test_bad_user_config_file_error_message

- [] Make sure test name smells such as "works", "correctly", "as_expected", "are_handled", etc. are not in the PR

examples:
 * test_that_arglist_is_parsed_correctly
 * test_that_history_observation_errors_are_calculated_correctly
 * test_that_double_comments_are_handled
 * test_that_quotations_in_forward_model_arglist_are_handled_correctly
 * test_that_the_manage_experiments_tool_can_be_used
 * test_that_parsing_workflows_gives_expected
 * test_that_history_observation_errors_are_calculated_correctly

- [] Make sure test names answer the question: what is correct behavior?

good examples:

 * test_that_config_path_substitution_is_the_name_of_the_configs_directory
 * test_when_forward_model_contains_multiple_steps_just_one_checksum_status_is_given

- [] Make sure that the tests that are in the `tests/ert/unit_tests` directory and not marked with `integration_test` should are fast and reliable.

By "integration test" we simply mean unit tests that is either too slow, too unreliable, have difficult
to understand error messages, etc.

- [] Make sure UI Tests describe behavior from a user interaction point of view. These are located in `tests/ert/ui_tests`.


# Commit message

- [] Make sure each commit does one atomic change
- [] Make sure the commit message is descriptive

Make sure commit messages to follow the following style:

- [] Separate subject from body with a blank line
- [] Limit the subject line to 50 characters
- [] Capitalize the subject line
- [] Do not end the subject line with a period
- [] Use the imperative mood in the subject line
- [] Wrap the body at 72 characters
- [] Use the body to explain what and why vs. how


# Documentation


- [] Code should not contain trivial documentation; ie. one that is self-explanatory from the function name for example `get_count()` 
- [] There should not be any commented out code
- [] User facing changes should be documented in an .rst file in the `docs/` directory
