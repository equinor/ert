# configuration for for `just`

# run poly example test-case
poly:
    ert gui test-data/ert/poly_example/poly.ert

# run snake oil test-case
snake_oil:
    ert gui test-data/ert/snake_oil/snake_oil.ert

pytest_args := env("ERT_PYTEST_ARGS", "--quiet")

# execute rapid unittests
rapid-tests:
    nice pytest -n auto tests/ert/unit_tests tests/everest --hypothesis-profile=fast -m "not integration_test"

ert-gui-tests:
    pytest {{pytest_args}} --mpl tests/ert/ui_tests/gui

ert-cli-tests:
    pytest {{pytest_args}} tests/ert/ui_tests/cli

ert-unit-tests:
    pytest {{pytest_args}} -n 4 --dist loadgroup --benchmark-disable tests/ert/unit_tests tests/ert/performance_tests

ert-doc-tests:
    pytest {{pytest_args}} --doctest-modules src/ --ignore src/ert/dark_storage

everest-tests:
    pytest {{pytest_args}} tests/everest -n 4 --dist loadgroup

build-everest-docs:
    sphinx-build -n -v -E -W ./docs/everest ./everest_docs

build-ert-docs:
    sphinx-build -n -v -E -W ./docs/ert ./ert_docs

build-docs: build-ert-docs build-everest-docs

check-types:
    mypy src/ert src/everest

test-all:
    parallel -j4 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests' 'just everest-tests'

check-all:
    parallel -j8 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests' 'just ert-doc-tests' 'just everest-tests' 'just check-types' 'just build-everest-docs' 'just build-ert-docs'
