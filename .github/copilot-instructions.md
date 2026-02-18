# Copilot instructions for `ert`

## Build, test, lint, and type-check commands

Use `uv` for all local commands (this repo is managed with `uv sync` groups).

```bash
# Dev environment
uv sync --all-groups

# Fast local regression checks (same target used by pre-push hook)
uv run just rapid-tests

# Full local check bundle (tests + typing + docs builds)
uv run just check-all

# ERT-focused suites
uv run just ert-unit-tests
uv run just ert-cli-tests
uv run just ert-gui-tests

# Everest suite
uv run just everest-tests

# Type checking
uv run just check-types
# (equivalent to: uv run mypy src)

# Style/lint hooks used in CI
SKIP=no-commit-to-branch uv run pre-commit run --all-files --hook-stage pre-push

# Docs
uv run just build-ert-docs
uv run just build-everest-docs
```

Run a single test:

```bash
uv run pytest tests/ert/unit_tests/<path>/test_<file>.py::test_<name>
uv run pytest tests/everest/test_<file>.py::test_<name>
```

## High-level architecture

- `src/ert`: main application package (CLI + GUI + analysis workflows + storage integration).
  - Entry point: `ert` script -> `src/ert/__main__.py`.
  - `ert` subcommands include GUI, lint, API server, and running in CLI.
  - `ert.services._storage_main` starts the storage server (uvicorn).
  - `ert.dark_storage` contains the FastAPI app/endpoints used by the storage/API layer.
- `src/everest`: optimization tool built on top of ERT.
  - Entry point for `everest`: `src/everest/bin/main.py`.
  - `everest/optimizer` converts the Everest model/configuration into `ropt`.
- `src/_ert`: shared low-level runtime helpers (e.g., threading/forward-model runner support).
- Plugin system lives under `src/ert/plugins` (hook specs + implementations + runtime plugin loading); `ert.__main__` executes within runtime plugin context.
- Tests are split by intent:
  - `tests/ert/unit_tests` (fast/reliable), `tests/ert/ui_tests` (cli/gui behavior), `tests/ert/performance_tests`, and `tests/everest`.

## Key repository conventions

- Prefer `just` targets for standardized test groupings and CI parity (`rapid-tests`, `check-all`, `ert-*`, `everest-tests`).
- Keep unit tests in `tests/ert/unit_tests` exceptionally fast; slower/broader cases should be marked with `@pytest.mark.integration_test` or moved.
- Test naming convention in this repo is explicit behavior-driven names (often `test_that_...`), not vague names like `test_works`.
- Type-hint policy from `CONTRIBUTING.md`:
  - avoid `Any` when possible,
  - use `@override` for overridden non-dunder methods,
  - prefer `cast`/`assert` over blanket `# type: ignore`.
- Pre-commit is the source of truth for formatting/lint hooks (`ruff-check --fix`, `ruff-format`, yaml/json checks, actionlint, lockfile checks).
- Some test data requires LFS/submodules (`git lfs install` and `git submodule update --init --recursive`) for representative local runs.

---

# Copilot Code Review Instructions

Apply these instructions only when performing a code review for this repository.
Focus on: correctness, clarity, reliability, and maintainability.

## Quick Checklist

- [ ] Code should not have any critical security flaws or bugs
- [ ] All new or changed logic is covered by appropriate automated tests.
- [ ] Test names follow the `test_that_<behavior_or_condition>` specification style.
- [ ] Test names clearly state the expected behavior or invariant (they answer: “What is correct behavior?”).
- [ ] Test names avoid vague terms: `works`, `correctly`, `as_expected`, `are_handled`, `handles`, `success`, `failure`, etc.
- [ ] Unit tests in `tests/ert/unit_tests` (not marked `integration_test`) are fast, reliable, and produce clear error messages.
- [ ] UI tests (in `tests/ert/ui_tests`) describe user-visible interactions and outcomes.
- [ ] Each commit performs one atomic, logically isolated change.
- [ ] Commit messages follow the prescribed format and explain the *what* and *why*, not the detailed *how*.
- [ ] Code does not contain trivial or redundant documentation.
- [ ] There is no commented-out (dead) code.
- [ ] User-facing changes include/update relevant `.rst` documentation under `docs/`.

---

## 1. Critical flaws and incorrect code


Ensure that there are no issues that would cause observable failures, including

* Runtime errors (crashes, exceptions, undefined behavior)
* Breaking changes (API changes, data structure changes)
* Security vulnerabilities (exploitable, not theoretical)

Also ensure that the code does not have any inconsistencies such as

* Incorrect or vague type annotations
* Comment vs code discrepancies


## 2. Testing

### 2.1 Coverage
Ensure all new functional paths or behaviors introduced by the PR are covered with unit tests or integration/UI tests as appropriate.

### 2.2 Naming Style
Test names MUST:
- Start with `test_that_` (or `test_when_` if describing conditional sequences) and then explicitly describe the behavior, condition, or invariant.
- Read like an executable specification: someone running `pytest --collect-only tests/` should infer purpose without opening the test file.

Good examples:
- `test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior`
- `test_that_adaptive_localization_with_cutoff_0_equals_ESupdate`
- `test_that_posterior_generalized_variance_increases_in_cutoff`
- `test_that_missing_arglist_does_not_affect_subsequent_calls`
- `test_that_setenv_does_not_expand_envvar`
- `test_that_new_line_can_be_escaped`
- `test_that_unknown_queue_option_gives_error_message`
- `test_when_forward_model_contains_multiple_steps_just_one_checksum_status_is_given`
- `test_that_config_path_substitution_is_the_name_of_the_configs_directory`

Poor examples (too vague, not behavior-focused, etc.):
- `test_color_always`
- `test_legends`
- `test_result_success`
- `test_result_failure`
- `test_print_progress`
- `test_bad_user_config_file_error_message`

### 2.3 Avoid Vague Terms (“Name Smells”)
Reject test names containing ambiguous fillers, e.g.:
- `works`, `correctly`, `as_expected`, `are_handled`, `handles`, `success`, `failure`
These words state *judgment* rather than *behavior*. Replace with the explicit condition or outcome.

Instead of: `test_that_arglist_is_parsed_correctly`
Prefer: `test_that_arglist_parsing_preserves_quoted_values` (be precise about the correctness criterion).

Instead of: `test_that_history_observation_errors_are_calculated_correctly`
Prefer: `test_that_history_observation_relative_error_is_a_percentage_of_the_value`

Instead of: `test_that_double_comments_are_handled`
Prefer: `test_that_double_comments_are_ignored`

(NOTE: If the PR contains any of the vague forms above, recommend renaming.)

### 2.4 Fast, Reliable Unit Tests
Tests in `tests/ert/unit_tests` not marked `integration_test` MUST:
- Execute quickly (aim: sub-second or minimal dependency overhead).
- Have deterministic outcomes (no flaky timing, random seeds un-fixed, or external service reliance).
- Produce clear, concise assertion failure messages.

Definition of “integration_test” marker: Use it only when a test is slow,
interacts with external systems/resources, involves complex multi-component
orchestration, or commonly yields opaque errors. If a test fails any of the
fast/reliable criteria, ensure it is marked appropriately or refactored.

### 2.5 UI Tests
Tests in `tests/ert/ui_tests` SHOULD:
- Reflect user-visible workflows (actions + expected UI states).
- Avoid duplicating pure logic assertions that are already covered in unit tests.

---

## 3. Commit Messages

Each commit SHOULD represent one atomic concern (e.g., “Refactor parameter parsing”, “Add adaptive localization cutoff test”).

Commit message format:
1. Subject line:
   - Limit to 50 characters
   - Imperative mood (e.g., “Add…”, “Refactor…”, “Remove…”).
   - Capitalized first letter.
   - No trailing period.
2. Blank line separating subject from body (if body exists).
3. Body (wrap at ~72 chars):
   - Explain WHY and WHAT changed (focus on rationale + scope).
   - Avoid detailing HOW unless unusual design decisions require justification.
   - Reference related tests or docs if helpful.

Reject commits that bundle unrelated changes (e.g., test addition + API rename + lint fixes) unless explicitly justified.

---

## 4. Documentation

- Avoid trivial docstrings that restate the obvious (`get_count()` does not need “Return count”).
- Docstrings should follow the google style guide.
- Remove commented-out code blocks; if something is temporarily disabled, use version control (or explain in commit message) rather than comments.
- For user-facing changes (new features, changed behaviors, configuration adjustments), ensure an `.rst` file under `docs/` is added or updated:
  - Include usage examples.
  - State backward compatibility or migration notes if applicable.

---

## 5. Type Hints

All code should have type hints checked by mypy.

1. Prefer not to use the `Any` type when possible.
1. Except for dunder methods (`__repr__`, `__eq__` etc.),  overridden methods
   should be decorated with the `@override` decorator.
1. Prefer use of `cast` or `assert` (as a type guard) over using the `#type: ignore`
   to ignore type errors.

---

## 6. Prioritization

Address in order:
1. Critical flaws
2. Incorrect or missing tests for critical logic.
3. Flaky or slow unit tests not marked as integration.
4. Incorrect or missing type hints.
5. Poorly named tests (vague or non-spec style).
6. Commit message policy violations.
7. Documentation gaps.

Provide concise, actionable suggestions—avoid generic praise or ungrounded criticism.

Do not include any of the following suggestions:

* Style suggestions outside the established guidelines
* Micro-optimizations without measurable impact

---

End of instructions.
