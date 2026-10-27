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
  - `ert.services._ert_server_main` starts the server (uvicorn).
  - `ert.server` contains the FastAPI app/endpoints for the storage/API layer, and is evolving to also orchestrate ERT and Everest experiments.
- `src/everest`: optimization tool built on top of ERT.
  - Entry point for `everest`: `src/everest/bin/main.py`.
  - `everest/optimizer` converts the Everest model/configuration into `ropt`.
- `src/_ert`: shared low-level runtime helpers (e.g., threading/forward-model runner support).
- Plugin system lives under `src/ert/plugins` (hook specs + implementations + runtime plugin loading); `ert.__main__` executes within runtime plugin context.
- Tests are split by intent:
  - `tests/ert/unit_tests` (fast/reliable), `tests/ert/ui_tests` (cli/gui behavior), `tests/ert/performance_tests`, and `tests/everest`.

## Key repository conventions

- Prefer `just` targets for standardized test groupings and CI parity (`rapid-tests`, `check-all`, `ert-*`, `everest-tests`).
- Test conventions (naming, organization, categories, mocking) are defined in `.github/instructions/coding-standards/python-tests.instructions.md`.
- Type-hint policy from `CONTRIBUTING.md`:
  - avoid `Any` when possible,
  - use `@override` for overridden non-dunder methods,
  - prefer `cast`/`assert` over blanket `# type: ignore`.
- Pre-commit is the source of truth for formatting/lint hooks (`ruff-check --fix`, `ruff-format`, yaml/json checks, actionlint, lockfile checks).
- Some test data requires LFS/submodules (`git lfs install` and `git submodule update --init --recursive`) for representative local runs.
- Keep code comments minimal: prefer readable variable/function names and clear code structure over comments that explain what the code does. A comment should not argue for the current implementation over a previous one (e.g., "changed from X to Y because..."); that rationale belongs in the commit message body, not in the code.
- Do not add a `Co-authored-by: Copilot` trailer (or similar) to git commit messages in this repository.

---

# Copilot Code Review Instructions

Apply these instructions only when performing a code review for this repository.
Focus on: correctness, clarity, reliability, and maintainability.

## Quick Checklist

- [ ] Code should not have any critical security flaws or bugs
- [ ] All new or changed logic is covered by appropriate automated tests.
- [ ] Tests follow `.github/instructions/coding-standards/python-tests.instructions.md`.
- [ ] Each commit performs one atomic, logically isolated change.
- [ ] Commit messages follow the prescribed format and explain the *what* and *why*, not the detailed *how*.
- [ ] Code does not contain trivial or redundant documentation.
- [ ] There is no commented-out (dead) code.
- [ ] Comments are minimal, and do not argue for the current code over an earlier version (that belongs in the commit body).
- [ ] Commit messages do not contain a `Co-authored-by: Copilot` trailer.
- [ ] User-facing changes include/update relevant `.rst` documentation under `docs/`.
- [ ] New code should prefer the spelling "runpath" over "run path" or "run_path"

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

Ensure all new functional paths or behaviors introduced by the PR are covered with unit tests or integration/UI tests as appropriate.

Review test code against `.github/instructions/coding-standards/python-tests.instructions.md`.

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

Commits MUST NOT include a `Co-authored-by: Copilot` trailer (or similar automated attribution trailer).

---

## 4. Documentation

- Avoid trivial docstrings that restate the obvious (`get_count()` does not need “Return count”).
- Docstrings should follow the google style guide.
- Remove commented-out code blocks; if something is temporarily disabled, use version control (or explain in commit message) rather than comments.
- Keep code comments minimal: favor clear, self-explanatory variable/function names and code structure so comments are superfluous. A comment must not justify the current code versus an earlier version (e.g., "previously this did X, now it does Y"); such rationale belongs in the commit message body.
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
6. Commit message policy violations (including `Co-authored-by: Copilot` trailers).
7. Excessive or comparative comments (arguing current vs. earlier code) instead of clear naming/structure.
8. Documentation gaps.

Provide concise, actionable suggestions—avoid generic praise or ungrounded criticism.

Do not include any of the following suggestions:

* Style suggestions outside the established guidelines
* Micro-optimizations without measurable impact

---

End of instructions.
