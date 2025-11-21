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
    OMP_NUM_THREADS=1 pytest -n auto --dist loadgroup tests/ert/unit_tests tests/everest \
    --hypothesis-profile=fast \
    --benchmark-disable \
    -m "not (integration_test or flaky or memory_test or limit_memory)" \
    --timeout=10 --session-timeout=120 \
    -p no:memray

ert-rapid-tests:
    OMP_NUM_THREADS=1 pytest --dist loadgroup tests/ert/unit_tests \
    --hypothesis-profile=fast \
    --ignore=tests/ert/unit_tests/gui \
    --ignore=tests/ert/unit_tests/dark_storage \
    --ignore=tests/ert/unit_tests/config/test_transfer_functions.py \
    --ignore=tests/ert/unit_tests/ensemble_evaluator/test_ensemble_client.py \
    -m "not (integration_test or flaky or memory_test or limit_memory or creates_tmpdir)" \
    -p no:memray -p no:doctest -p no:benchmark -p no:mpl -p no:cov -p no:pytest-qt

continuous_tests:
    fswatch -r -o --event Created --event Updated --event Removed --event Renamed src/ tests/ --exclude="\.egg-info|__pycache__" | while IFS= read -r _; do nice just ert-rapid-tests; done

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
    sphinx-build -v -E -W ./docs/everest ./everest_docs

build-ert-docs:
    sphinx-build -v -E -W ./docs/ert ./ert_docs

build-docs: build-ert-docs build-everest-docs

check-types:
    mypy src

test-all:
    parallel -j4 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests' 'just everest-tests'

ert-tests:
    parallel -j4 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests'

check-all:
    parallel -j8 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests' 'just ert-doc-tests' 'just everest-tests' 'just check-types' 'just build-everest-docs' 'just build-ert-docs'
