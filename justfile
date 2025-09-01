# configuration for for `just`

poly:
    ert gui test-data/ert/poly_example/poly.ert

snake_oil:
    ert gui test-data/ert/snake_oil/snake_oil.ert

heat_equation:
    ert gui test-data/ert/heat_equation/config.ert

pytest_args := env("ERT_PYTEST_ARGS", "--quiet")

# execute rapid unittests
rapid-tests:
    OMP_NUM_THREADS=1 pytest -n auto --benchmark-disable --dist loadgroup tests/ert/unit_tests tests/everest --hypothesis-profile=fast -m "not (integration_test or flaky or memory_test or limit_memory)"

ert-gui-tests:
    pytest {{pytest_args}} --mpl tests/ert/ui_tests/gui

ert-cli-tests:
    pytest {{pytest_args}} tests/ert/ui_tests/cli

ert-memory-tests:
    _RJEM_MALLOC_CONF="dirty_decay_ms:100,muzzy_decay_ms:100" pytest -n 2 {{pytest_args}} tests/ert -m "memory_test"
    _RJEM_MALLOC_CONF="dirty_decay_ms:100,muzzy_decay_ms:100" pytest -n 2 {{pytest_args}} tests/ert -m "limit_memory" --memray

ert-unit-tests:
    pytest {{pytest_args}} -n 4 --dist loadgroup --benchmark-disable tests/ert/unit_tests tests/ert/performance_tests -m "not (memory_test or limit_memory)"

ert-doc-tests:
    pytest {{pytest_args}} --doctest-modules src/ --ignore src/ert/dark_storage

everest-tests:
    pytest -n 4 --benchmark-disable --dist loadgroup {{pytest_args}} tests/everest

build-everest-docs:
    sphinx-build -n -v -E -W ./docs/everest ./everest_docs

build-ert-docs:
    sphinx-build -n -v -E -W ./docs/ert ./ert_docs

build-docs: build-ert-docs build-everest-docs

check-types:
    mypy src

test-all:
    parallel -j4 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests' 'just everest-tests'

ert-tests:
    parallel -j4 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests'

check-all:
    parallel -j8 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests' 'just ert-doc-tests' 'just everest-tests' 'just check-types' 'just build-everest-docs' 'just build-ert-docs'
