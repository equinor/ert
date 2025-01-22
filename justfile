# configuration for for `just`

# run poly example test-case
poly:
    ert gui test-data/ert/poly_example/poly.ert

# run snake oil test-case
snake_oil:
    ert gui test-data/ert/snake_oil/snake_oil.ert

# execute rapid unittests
rapid-tests:
    nice pytest -n auto tests/ert/unit_tests tests/everest --hypothesis-profile=fast -m "not integration_test"

check-all:
    mypy src/ert src/everest
    pre-commit run --all-files
    pytest tests/everest -n 4 -m everest_models_test --dist loadgroup
    pytest tests/everest -n 4 -m integration_test --dist loadgroup
    pytest tests/ert/ui_tests/ --mpl --dist loadgroup
    pytest tests/ert/unit_tests/ -n 4 --dist loadgroup
    pytest --doctest-modules src/ --ignore src/ert/dark_storage
    pytest tests/ert/performance_tests --benchmark-disable --dist loadgroup
